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

## Results

4 models trained on TinyStories (char-level tokenizer, ~4.5M training characters), varying only
`d_model` (`num_layers=2`, `num_heads=4`, `block_size=128` fixed - see `src/scaling_configs.py`).
Full run: `python src/run_all.py` then `python src/fit_scaling_law.py`; full details, deviations
from the original plan, and the statistical caveats in `notes.md`.

| run | target N | achieved N | final val loss | FLOPs at best | wall-clock (s) |
|---|---|---|---|---|---|
| scale_100k | 100,000 | 100,680 | 1.1555 | 1.060e+14 | 1029.9 |
| scale_250k | 250,000 | 244,032 | 1.1062 | 2.634e+14 | 1135.0 |
| scale_500k | 500,000 | 503,720 | 1.0363 | 5.437e+14 | 1698.7 |
| scale_1m | 1,000,000 | 1,007,600 | 0.9697 | 1.088e+15 | 2482.1 |

Total wall-clock across all 4 runs: 106.0 min.

**Fitted power law**, `L(N) = L_inf + A*N^-alpha`:

| fit | alpha | 95% CI | notes |
|---|---|---|---|
| full (curve_fit, 3 params) | **0.0756** | **[-5.67, 5.82]** (df=1) | `L_inf` fit to ~0 (hit its lower bound) |
| simple (OLS on log-log, `L_inf=0`) | **0.0768** | - | R^2 = 0.977 (df=2) |
| Kaplan et al. (2020) | ~0.076 | - | token-level, BPE-tokenized web text |

![Scaling law fit](results/scaling_law_fit.png)
![Compute-optimal frontier](results/compute_frontier.png)

**The alpha values above land almost exactly on Kaplan's 0.076 - and that's a coincidence, not a
confirmation.** With only 4 points and 3 fitted parameters, there's exactly 1 residual degree of
freedom, and the correctly-computed 95% CI (`t.ppf(0.975, df=1) ~= 12.7`, not the usual 1.96)
spans from strongly negative to strongly positive - wide enough to be consistent with almost any
exponent. The real result of this build isn't "we reproduced Kaplan's constant"; it's the CI
itself, plus a concrete list of reasons this setup structurally couldn't have cleanly reproduced
it even with a tighter fit: a width-only sweep (depth held fixed, unlike Kaplan's jointly-scaled
family), a character-level vocabulary instead of BPE, an N range spanning roughly one order of
magnitude (Kaplan's spans many), a small non-compute-optimally-trained dataset, and the
fundamental weakness of fitting 3 parameters to 4 points. As the framing above says: a careful
negative result here is more useful than a clean positive one would have been - see `notes.md`
for the full breakdown, including that none of the 4 runs actually triggered early stopping
(all were still improving, or nearly so, when the fixed epoch budget ran out), which means these
are "loss after equal training budget" numbers, not converged losses.
