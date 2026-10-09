# Week 3 notes — LoRA, rank sweep, quantization

**Status: code written and the LoRA maths unit-tested. The training runs have NOT been done yet.**
Sections marked ⏳ are for you to fill in after running. Nothing in them is invented.

---

## 🔑 The short version

- **Question this week answers:** how many trainable parameters does fine-tuning actually need?
- **Deliverable:** one plot — trainable parameters vs validation loss, for rank 1, 2, 4, 8, 16, 64.
- **Model:** SmolLM2-360M. **Data:** 2,000 Alpaca-cleaned examples (+200 held out).
- **What I could verify without the GPU run:** the LoRA class passes 5 checks on a toy model (`src/test_lora_toy.py`).

---

## 📋 How to run (in this order)

```bash
# 0. one-off: install the 3 missing packages into the shared venv
/home/jacob/venv/bin/pip install "transformers>=4.56" peft datasets

cd /home/jacob/ai_ml_learning/week-03-adaptation

# 1. LoRA maths on a toy model (seconds, no downloads)
/home/jacob/venv/bin/python src/test_lora_toy.py

# 2. same checks on the real model (downloads SmolLM2-360M, ~1.5 GB)
/home/jacob/venv/bin/python src/verify.py

# 3. all training runs: base + rank sweep + peft comparison (resumable)
/home/jacob/venv/bin/python src/run_sweep.py

# 4. plots + table, then the 10 before/after answers
/home/jacob/venv/bin/python src/plot_results.py
/home/jacob/venv/bin/python src/generate.py
```

- If step 3 is interrupted, run it again. Finished runs are skipped.
- To run one job by hand: `python src/train.py --method mine --rank 8`.

---

## 📋 What each file does

| file | job |
|---|---|
| `src/config.py` | model name, target layers, paths (one place) |
| `src/lora.py` | **my LoRA**: `LoRALinear`, `apply_lora`, merge, param counting |
| `src/test_lora_toy.py` | 5 checks on a toy model — passes |
| `src/data.py` | Alpaca → tokens; loss only on the answer part |
| `src/train.py` | training loop for `mine`, `peft`, or `base` |
| `src/verify.py` | README step 2 on the real model |
| `src/run_sweep.py` | runs every job in order |
| `src/plot_results.py` | rank-sweep plot, mine-vs-peft plot, table |
| `src/generate.py` | 10 before/after completions |

---

## 🧮 Theory I worked out

### Trainable parameter count

- One patched layer `in → out` adds `A (r × in)` and `B (out × r)`.
- That is **r · (in + out)** parameters.
- For a square `d × d` layer this is `2·r·d`, the formula in the paper.

**This model is not all-square.** SmolLM2 uses grouped-query attention.

- `q_proj`, `o_proj`: 960 → 960.
- `k_proj`, `v_proj`: 960 → 320 (fewer key/value heads).
- Per layer, per unit of rank: (960+960) + (960+320) + (960+320) + (960+960) = **6,400**.
- Over 32 layers: **204,800 · r**. So r = 8 → ~1.64 M. r = 64 → ~13.1 M.

⚠️ These dimensions are from my memory of the model config. `verify.py` computes the real number from the actual layer shapes, so trust that output over this paragraph.

### Why zero-init B (and not A)

- If **B = 0**: the update is 0, so the model starts identical to the base. ✅ That is what we want.
- If **A = 0 and B random** the same is true — but then B gets no gradient at first.
- If **both are 0**: nothing ever learns. Both gradients are zero forever (each is multiplied by the other).
- With **B = 0, A random**: the first backward pass gives `grad(B) ≠ 0` and `grad(A) = 0`. B moves first. After that B ≠ 0 and A learns too.
- `test_lora_toy.py` checks exactly this: "grad(A) == 0, grad(B) != 0 at step 0".

### Why the update is bit-identical at step 0

- Output = `base(x) + 0.0`. Adding exactly zero changes no bits.
- So `torch.equal(base_logits, lora_logits)` should be **True**, not just "close".
- This only holds in fp32, **no autocast**, same device. `verify.py` does it that way.

### Why α / r, and why α is fixed

- The update is scaled by `α / r`.
- I keep **α = 16 for every rank** (what the paper does).
- Consequence: at r = 1 the scale is 16; at r = 64 it is 0.25.
- This keeps the *size* of the update roughly comparable as r grows. Without it, a bigger r would effectively mean a bigger learning rate.
- ⚠️ This is a design choice that affects the sweep. A different α policy could move where the curve flattens.

### Why LoRA is needed on this GPU at all

Memory per trainable parameter with AdamW in fp32: weights 4 + gradient 4 + two Adam moments 8 = **16 bytes**.

| method | calculation | memory |
|---|---|---|
| full fine-tune | 360 M × 16 B | **~5.8 GB** ❌ (GPU has 4 GB) |
| LoRA r = 8 | 360 M × 4 B (frozen) + 1.6 M × 16 B | **~1.5 GB** + activations ✅ |

- Activations come on top. They scale with batch size and sequence length, not with LoRA rank.
- So even r = 64 costs little extra memory. The rank sweep is nearly "free" in memory terms; it is the **time** that adds up.

---

## 🛠️ Design decisions (and why)

- **SmolLM2-360M, not Qwen3-0.6B.** README allows either. The T1000 has 4 GB; 0.6B in fp32 is ~2.4 GB before activations. Too tight.
- **fp32 weights + fp16 autocast.** The T1000 is a Turing card. It has no bf16. fp16 autocast with a GradScaler is the usual workaround.
  - ⚠️ fp16 can overflow. If you see `nan` loss, that is the first suspect.
- **Micro-batch 2 × accumulation 8 = effective batch 16.** Small micro-batch to fit memory; accumulation to get a sensible batch. `run_sweep.py` auto-retries with gradient checkpointing if it hits OOM.
- **Alpaca-cleaned, 2,000 train / 200 val.** README says 1–2k examples. Fixed seed → every run sees the same split and the same batch order.
- **Loss only on the answer tokens.** Prompt tokens are masked with `-100`. An EOS token is appended to each answer so the model learns to **stop**.
- **All four attention projections (q, k, v, o).** README says "attention projections". QLoRA's finding supports adapting more than q and v.
  - Not tested: q+v only. Good extension if time allows.
- **2 epochs, lr 2e-4, cosine decay, 5% warm-up, grad-clip 1.0.** Standard LoRA settings. **Not tuned.** ⚠️ Same lr for every rank is a simplification.
- **Sweep uses my LoRA; peft is run once at r = 8.** That is enough for the "do the curves overlap" check.
- **Validation loss every 25 steps** (200 held-out examples) plus step 0.
  - Step-0 validation loss with LoRA = the base model's loss. A nice built-in sanity check.

---

## 🔮 Predictions (written BEFORE running)

Write down what you expect, then compare. Mine:

1. **Step-0 val loss is identical** for the base model and every LoRA run (zero-init).
2. **Most of the gain arrives by r = 4 or 8.** Large step from base → r = 1, then small steps.
3. **The curve flattens before r = 64.** r = 16 and r = 64 should be within noise of each other.
4. **My LoRA and peft overlap at r = 8**, but not perfectly. Different random A matrices, and different RNG use.
5. **The big before/after change is behaviour, not facts.** The base model rambles on or repeats; the tuned model answers and stops.

⚠️ How would I know "within noise"? I only run **one seed per rank**. To claim a plateau honestly I'd need the run-to-run spread — for example 3 seeds at r = 8. That is the week-2 lesson again: a single point has no error bar.

---

## ⏳ Results (fill in after running)

### Verification (`verify.py`)
- Trainable params, mine: ___ &nbsp;&nbsp; theory: ___ &nbsp;&nbsp; peft: ___
- Step-0 logits bit-identical: ___ (✅/❌)
- max |grad(A)| at step 0: ___ &nbsp;&nbsp; max |grad(B)|: ___

### Rank sweep (`results/rank_sweep_table.md`)

| rank | trainable params | final val loss | gain vs previous | minutes |
|---|---|---|---|---|
| base | 0 | ___ | | |
| 1 | ___ | ___ | | ___ |
| 2 | ___ | ___ | | ___ |
| 4 | ___ | ___ | | ___ |
| 8 | ___ | ___ | | ___ |
| 16 | ___ | ___ | | ___ |
| 64 | ___ | ___ | | ___ |

- Where do returns flatten? ___
- Did predictions 1–5 hold? ___

### Mine vs peft (rank 8)
- Final val loss: mine ___, peft ___, difference ___
- Do the curves overlap? ___

### Before/after (`results/before_after.md`)
- What changed in the answers? ___
- Any case where "after" is worse? ___

### Problems I hit
- ___

---

## 📋 Quantization — what I learned from the reading (not built this week)

Short version. Full detail in `PAPER_SUMMARY.md`.

- **QLoRA = 4-bit frozen base + 16-bit LoRA.** It saves *memory*, not time.
- **NF4** places its 16 levels at normal-distribution quantiles, because weights are roughly normal.
- **LLM.int8** found a handful of huge "outlier" dimensions that break naive int8 above ~6.7B parameters. Fix: do those few dimensions in fp16.
- **GPTQ** quantizes one column at a time and corrects the remaining columns for the rounding error. It is for **deployment**, not training.
- **Why quantization speeds up generation:** decoding is memory-bandwidth-bound. Fewer bytes per weight = less to read per token.

**Could I have used QLoRA here?** Yes, with `bitsandbytes` (not installed). It would shrink the 1.4 GB base to ~0.25 GB. Not needed for a 360M model. It matters for 7B+.

---

## ❓ Open questions

- Does the plateau move if α is scaled with r (e.g. α = 2r) instead of fixed?
- Does q+v only match q+k+v+o at the same parameter budget? (LoRA paper vs QLoRA paper disagree on emphasis.)
- Is validation loss on 200 examples precise enough to see differences of ~0.005?
