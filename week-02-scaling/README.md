# Week 2 — Scaling, Data, and Architecture Beyond the Transformer

Why models are the sizes they are, and what is replacing the pure attention stack. Strongest week for your background — scaling laws are a curve-fitting exercise and most people in ML treat them as folklore rather than as fitted models with confidence intervals.

**Core**
- Kaplan et al. (2020), *Scaling Laws for Neural Language Models* — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2) — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

**Supporting**
- Fedus et al. (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- NVIDIA (2026), *Nemotron 3 Super* — [arXiv:2604.12374](https://arxiv.org/abs/2604.12374) — a current production model. Skim it for what a 2026 architecture actually looks like: mixture-of-experts, alternating attention and Mamba-2 layers, and a long section on training and quantization recipes.

Kaplan and Chinchilla disagree about the optimal ratio of parameters to tokens. Work out where the disagreement comes from — it is largely about whether the learning rate schedule is tuned per-run. This is a good example of how much of "scaling law" work is sensitive to experimental hygiene.

The hybrid attention/state-space design in Nemotron and Qwen3.6 is one of the clearest architectural trends of the last eighteen months, driven by the cost of long contexts in agent systems.

**Build: measure your own scaling law**

Reuse the Week 1 model.

1. Train 6–9 models across a grid of parameter counts (say 0.1M to 10M, varying width and depth) on a fixed dataset — TinyStories or wikitext-2.
2. Record final validation loss and total training FLOPs (estimate as $6ND$ — derive this, it is a good exercise).
3. Fit $L(N) = L_\infty + A N^{-\alpha}$ by least squares in log space. Report $\alpha$ with a confidence interval.
4. Plot loss against compute on log-log axes, draw the compute-optimal frontier, compare your exponent to Kaplan's ≈0.076.

### updated goal;
Train 4 models instead of 9.

For example

0.25M
0.5M
1M
2M

That's already enough to fit a power law and discuss why it differs from Kaplan.

The exponent will not match theirs, and the interesting part of the write-up is explaining why: different data distribution, different vocabulary, models far below the range where the power law was fitted. A careful negative result here is more impressive than a clean positive one.
