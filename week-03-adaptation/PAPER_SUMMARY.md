# Week 3 papers — summary

Covers the 3 core papers (LoRA, QLoRA, FLAN) and the 2 supporting ones (LLM.int8, GPTQ).
Written so you can scan it instead of reading all five in full.

**Written from my knowledge of the papers, not re-read line by line.**
Numbers are the headline ones from each abstract/main results.
Before you quote one in public, check it against the paper.

---

## 🗺️ The big picture (read this first)

**Week 3 is about one problem: big models are too big to touch.**

- Fine-tuning every weight needs ~16 bytes per parameter (weights + gradients + Adam state).
- Serving the model needs the weights in memory too.
- Each paper attacks a different part of that cost.

| paper | what it shrinks | how |
|---|---|---|
| LoRA | **trainable** parameters | learn a small low-rank update, freeze the rest |
| QLoRA | **stored** base weights | freeze the base in 4-bit, train LoRA on top |
| FLAN | *(nothing — it's the "why")* | what to fine-tune **on**: instructions |
| LLM.int8 | weights at **inference** | 8-bit, with a trick for outliers |
| GPTQ | weights at **inference** | 3–4 bit, one-shot, error-compensated |

---

## 1. Hu et al. (2021) — LoRA

**[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)**

### 🔑 Key point
**Freeze the pretrained weight W. Learn a low-rank update next to it: h = Wx + (α/r)·BAx.**
You train ~0.01%–1% of the parameters and lose almost no quality.

---

### 📋 The mechanism
- A is `r × in`, random init. B is `out × r`, **zero** init.
- Zero init means the update is 0 at step 0. The model starts as the base model.
- The update is scaled by **α / r**.
  - α is set once (to the first r tried) and **not tuned** when r changes.
  - So bigger r means a smaller per-direction scale. This keeps the learning rate roughly usable across ranks.
- After training you can **merge**: W' = W + (α/r)·BA. Inference cost is identical to the base model. No added latency.
- Adapters are tiny. On GPT-3 175B the checkpoint goes from ~350 GB to ~35 MB.

### 📋 The numbers (GPT-3 175B)
- Trainable parameters: **~10,000× fewer** than full fine-tuning.
- GPU memory during training: **~3× less**.
- Quality: on par with or better than full fine-tuning on the tasks tested (GLUE, WikiSQL, SAMSum, E2E).

### 📋 The rank evidence (this is the claim your build tests)
- The paper **asserts** that the weight update has low intrinsic rank.
- Then it **checks it empirically**. It does not prove it.
- Ablation: with q and v both adapted, **r = 1 or 2 already performs well** on WikiSQL and MNLI.
- Subspace analysis: the top directions found at r = 8 and r = 64 overlap strongly. Extra rank mostly adds noise directions.
- Another finding: ΔW does **not** copy the top directions of W. It amplifies directions that W only weakly uses (a factor of roughly 20× in their example).
- Spreading a fixed budget across **more weight types at lower rank** (q and v) beat putting it all in one type at higher rank.

---

### Figures and tables worth opening
- **The reparametrization diagram (Figure 1).** Two boxes A and B beside W. Clearest picture of the whole idea.
- **The rank table (Table 6, in Section 7.2).** Accuracy vs r. Look at how flat the rows are. That flatness is the claim.
- **Table 5 (Section 7.1).** Which weight matrices to adapt, at equal parameter budget.

---

### ⚠️ Caveats
- **"r = 1 is enough" is task-dependent.** The tasks tested are close to what the pretrained model already knows.
  - Later work (Biderman et al. 2024, *LoRA Learns Less and Forgets Less* — outside this week's reading) found LoRA lags full fine-tuning on harder, far-from-pretraining tasks like code and maths.
- The paper adapts **only q and v** in the main experiments. QLoRA (below) found that adapting **all linear layers** is needed to match full fine-tuning.
- Rank is "low" relative to d, but the real question is "low relative to the task". You are testing exactly this in the sweep.

---

## 2. Dettmers et al. (2023) — QLoRA

**[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)**

### 🔑 Key point
**Store the frozen base model in 4 bits. Train 16-bit LoRA adapters on top.**
This fine-tunes a **65B model on one 48 GB GPU** with no measured loss vs 16-bit fine-tuning.

---

### 📋 Three tricks
1. **NF4 (4-bit NormalFloat).**
   - Pretrained weights are roughly normally distributed.
   - NF4 places its 16 levels at **quantiles of a normal distribution**, so each level is used about equally often.
   - It includes an exact zero. Plain Int4 / FP4 are worse at the same bit width.
   - "Information-theoretically optimal" holds **for normal data**. Real weights are only approximately normal.
2. **Double quantization.**
   - Block-wise quantization stores one fp32 scale per block of 64 weights = 0.5 extra bits per weight.
   - Quantize those scales too (8-bit, blocks of 256) → ~0.127 bits per weight.
   - Saves **~0.37 bits per parameter**. About 3 GB on a 65B model.
3. **Paged optimizers.**
   - Optimizer state is paged to CPU RAM when the GPU is about to run out (NVIDIA unified memory).
   - This stops OOM crashes on long sequences.

### 📋 How the forward pass works
- Weights live in NF4.
- For each matmul, a block is **de-quantized on the fly to bf16**, used, and discarded.
- Gradients flow **through** the frozen 4-bit weights into the LoRA matrices. The 4-bit weights are never updated.
- LoRA A and B stay in 16-bit.

### 📋 Findings that matter
- **Adapt every linear layer.** Adapting only q and v (as in LoRA) did not match full fine-tuning. This is why our build patches q, k, v and o.
- **Data quality beats data quantity.** A small, clean set (OASST1, ~9k examples) beat a much larger one (FLAN v2 subset) *for chatbot quality*.
- Their model **Guanaco** reached ~99% of ChatGPT's score on the Vicuna benchmark after ~24 h on one GPU.

---

### Figures and tables worth opening
- **Figure 1.** Memory layout of full fine-tuning vs LoRA vs QLoRA. Shows where the bytes go.
- **The NF4 vs FP4 vs Int4 table.** Shows the data type matters at 4 bits.

---

### ⚠️ Caveats
- The Vicuna-benchmark result uses **GPT-4 as the judge**. The authors themselves flag that this is a noisy, biased measure.
- 4-bit makes each step slower (de-quantization cost). QLoRA is a **memory** win, not a speed win.
- "No loss vs 16-bit" is measured on the benchmarks in the paper, not on every task.

---

## 3. Wei et al. (2021) — FLAN

**[arXiv:2109.01652](https://arxiv.org/abs/2109.01652)**

### 🔑 Key point
**Fine-tune a language model on many tasks written as natural-language instructions, and it follows instructions for tasks it has never seen.**
That is instruction tuning. It is the reason a plain next-word predictor turns into an assistant.

---

### 📋 The setup
- Base model: a 137B-parameter pretrained LM (LaMDA-PT).
- Data: **60+ NLP datasets**, grouped into task clusters (sentiment, translation, QA, NLI…).
- Each dataset gets ~10 hand-written instruction templates, so the same task is phrased in several ways.
- **Held-out evaluation is by cluster.** To test NLI, *every* NLI dataset is removed from training. This is what makes "zero-shot on unseen tasks" a fair claim.

### 📋 The numbers
- FLAN zero-shot beats **zero-shot GPT-3 (175B) on 20 of 25 datasets**.
- It also beats **few-shot** GPT-3 on several (e.g. ANLI, RTE, BoolQ, OpenbookQA).

### 📋 Ablations (the part worth the most attention)
- **More task clusters → better** on held-out tasks.
- **Scale matters.** At **8B and below**, instruction tuning *hurt* held-out performance. It only helped at ~68B and 137B.
  - The paper's explanation: small models use their whole capacity to learn the training tasks and have none left to generalise.
- **Instructions matter.** Fine-tuning on the same data with only a dataset name, or no template, did much worse.
- Little or no gain on tasks that are already phrased as plain language modelling (e.g. commonsense completion, coreference).

---

### Figures worth opening
- **Figure 1** — the clusters and the held-out split. Shows the experimental design in one picture.
- **The scale ablation figure.** The "small models get worse" line is the surprising bit.

---

### ⚠️ Caveats and how it connects to your build
- The "small models are hurt" result is **specific to this paper's data and size range**.
  - Later work (Chung et al. 2022, Flan-T5 / Flan-PaLM — outside this week's reading) used far more tasks and showed small models *do* benefit.
- **Your build is a mini FLAN.** Alpaca-cleaned is instruction data, and SmolLM2-360M is small.
  - The base model already writes fluent text. What you will mostly see is it **learning the format**: answer the question, then stop.
  - Expect the biggest before/after difference to be *behaviour* (stops at the right place, follows the instruction), not new knowledge.

---

## 4. Dettmers et al. (2022) — LLM.int8()

**[arXiv:2208.07339](https://arxiv.org/abs/2208.07339)**

### 🔑 Key point
**8-bit inference with no quality loss, even at 175B — once you handle a few huge outlier features separately.**

---

### 📋 The problem
- Naive int8 quantization works fine up to ~1–2B parameters.
- Above that, quality collapses.

### 📋 The cause: emergent outlier features
- A few hidden dimensions (~6 out of thousands) have values **up to ~20× larger** than the rest.
- They show up in a growing share of layers as the model grows, and become systematic at **~6.7B parameters**.
- One huge value forces the whole quantization scale up, so every normal value gets squashed into a few int8 levels.
- Zeroing the outlier dimensions destroys performance. They carry real information.

### 📋 The fix: two ideas
1. **Vector-wise quantization.** Separate scale per row of X and per column of W, not one per tensor.
2. **Mixed-precision decomposition.**
   - Find the outlier dimensions (magnitude ≥ 6).
   - Multiply those in **fp16**.
   - Multiply everything else in **int8**.
   - The outliers are ~0.1% of the dimensions, so ~99.9% of the maths is still int8.

### 📋 The result
- Memory for inference roughly **halves** vs fp16.
- OPT-175B / BLOOM-176B can run on a single multi-GPU node of consumer-grade GPUs.

---

### ⚠️ Caveats
- It is a **memory** win. The mixed-precision split makes it **slower** than fp16 for smaller models.
- The "phase transition at 6.7B" is an empirical observation on the model families tested.

---

## 5. Frantar et al. (2022) — GPTQ

**[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)**

### 🔑 Key point
**Quantize a trained model to 3–4 bits in a few GPU-hours, with no retraining, by fixing the rounding error as you go.**

---

### 📋 How it works
- **Post-training, one-shot, layer by layer.** For each layer it minimises the error in that layer's *output* on a small calibration set (~128 text samples).
- It quantizes one weight column at a time. After rounding a column, it **adjusts the not-yet-quantized columns** to cancel the damage. The adjustment uses an approximate **second-order (Hessian)** estimate.
- This descends from Optimal Brain Quantization. GPTQ made it fast enough for 100B+ models:
  - Same column order for every row.
  - Lazy batched updates (better GPU use).
  - Cholesky decomposition for numerical stability.

### 📋 The numbers
- Quantizes a **175B model in ~4 GPU-hours** on one A100.
- **3–4 bit** with small perplexity increase. 2-bit and ternary work only with noticeable degradation.
- Generation speedups of **~3–4.5×** on A100 / A6000 using custom kernels.
  - The reason: token-by-token generation is **memory-bandwidth bound**. Fewer bytes to read = faster.

---

### ⚠️ Caveats
- Needs a **calibration set**. A mismatch with real use can hurt.
- The speedup comes from the **custom kernels**, not from the quantization alone.
- Weight-only: activations stay in fp16.

---

## 🧩 How the five fit together

- **LoRA** = *train* cheaply. **QLoRA** = train cheaply *and* store cheaply.
- **FLAN** = what data to train on.
- **LLM.int8 / GPTQ** = *serve* cheaply. Both are post-training.
- QLoRA uses NF4 (a data type), not GPTQ. Both are 4-bit, but different tools for different jobs:
  - NF4: simple, fast to apply, good for **training**.
  - GPTQ: slower to apply, error-compensating, good for **deployment**.

---

## ❓ Questions to carry into the build

1. **LoRA says r = 1 can be enough. Does your sweep agree?** Where does the curve flatten?
2. **QLoRA says adapt all layers. Would q and v alone have been worse?** (Not tested here. Possible extension.)
3. **FLAN says small models may not gain from instruction tuning.** Does 360M gain? Is the gain "format" or "knowledge"?
4. **Is val loss the right yardstick?** Loss can improve while answers don't. That is why the before/after table exists.
