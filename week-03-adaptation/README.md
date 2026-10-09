# Week 3 — Adaptation: Fine-Tuning, PEFT, Quantization

How a pretrained model becomes a useful one. This is the most directly employable week — most applied ML jobs involve adapting an existing model, not training one.

**Core**
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs* — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Wei et al. (2021), *Finetuned Language Models Are Zero-Shot Learners* (FLAN) — [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)

**Supporting**
- Dettmers et al. (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar et al. (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

LoRA rests on the claim that the weight *update* during fine-tuning has low intrinsic rank. Note that this is asserted and then validated empirically rather than proven, and that the paper's rank ablation is the evidence. The NF4 data type in QLoRA is a genuinely elegant piece of work — a quantile-based quantization scheme that is information-theoretically optimal for normally distributed weights.

**Build: implement LoRA, then use the real thing**

1. Write LoRA yourself: an `nn.Module` wrapper holding $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, with $B$ zero-initialised, that patches the attention projections of a HuggingFace model. About 50 lines.
2. Verify: freeze the base model, confirm that trainable parameter count matches theory, and that at step 0 the model's outputs are bit-identical to the base model (this is what zero-init of $B$ buys you).
3. Fine-tune `Qwen3-0.6B` or `SmolLM2-360M` on an instruction dataset (`databricks-dolly-15k` or a slice of `no_robots`) with your implementation, then with HF `peft`, and confirm the loss curves overlap.
" I'd instead use Alpaca-cleaned or TinyOrca with 1–2k examples."
4. Sweep rank $r \in \{1, 2, 4, 8, 16, 64\}$. Plot trainable parameters against validation loss. Find where the returns flatten.
5. Generate ten before/after completions and put them in the README.

The rank sweep plot is the deliverable. It answers a question people actually ask in production.

---

## How to run

See `notes.md` for the full step-by-step. Short version:

```bash
/home/jacob/venv/bin/pip install "transformers>=4.56" peft datasets   # one-off
/home/jacob/venv/bin/python src/verify.py        # step 2: checks on the real model
/home/jacob/venv/bin/python src/run_sweep.py     # steps 3-4: all training runs
/home/jacob/venv/bin/python src/plot_results.py  # the rank-sweep plot (the deliverable)
/home/jacob/venv/bin/python src/generate.py      # step 5: ten before/after completions
```

Reading summary: `PAPER_SUMMARY.md`. Notes and predictions: `notes.md`.

## Results

_Not run yet. After running, paste `results/rank_sweep_table.md`, embed `results/rank_sweep.png`,
and paste `results/before_after.md` here._
