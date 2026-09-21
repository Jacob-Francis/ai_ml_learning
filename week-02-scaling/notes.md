# Week 2 notes — measuring a scaling law

## Setup

- **Model**: week-1's `Transformer` (decoder-only, weight-tied embeddings), imported directly
  from `week-01-transformers/src/main.py` via a sibling-path `sys.path` hack (see `train.py`'s
  module docstring) rather than copy-pasted - so any week-1 bugfix carries over automatically.
- **Data**: TinyStories validation split (`TinyStories-valid.txt`, ~19.4MB, downloaded via
  `src/data_download.py`), **not** week-1's TinyShakespeare - TinyStories is a synthetic,
  restricted-vocabulary dataset built for training tiny language models, which matches this
  week's 0.1M-1M parameter regime much better.
- **Tokenizer**: character-level (`chars = sorted(set(text))`), same pattern as week-1, not BPE.
  Reasoning: with weight-tied embeddings, `vocab_size` is the *only* place vocabulary enters the
  parameter count (`N = D*(V+B+2) + L*(12D^2+9D)`, see `scaling_configs.py`). A BPE vocab of even
  ~1000 tokens would eat 30-60% of the smallest (0.1M) model's parameter budget, contaminating N
  with "vocab size" more than "transformer capacity" - the thing this sweep is actually trying to
  isolate. Char-level keeps vocab ~90, under 6% of even the smallest model.
  - **Consequence, stated up front**: char-level cross-entropy losses in this write-up are
    **not** comparable to published BPE-token TinyStories numbers. That's expected and doesn't
    undermine the exercise - it's the *exponent* alpha being measured, not the absolute loss.

## Environment (deviation #1)

The plan called for a fresh venv at `week-02-scaling/.venv`. Instead, a pre-existing shared venv
at `/home/jacob/venv` already had torch 2.9.1+cu128, numpy, matplotlib, and scipy installed
(confirmed before writing any code) - reusing it avoided a large redundant torch
download/install. Only `nbconvert`/`nbclient` (needed to execute the notebook headlessly) were
missing and were installed into that same venv. `requirements.txt` pins the versions actually
used, and documents this choice inline.

## GPU, not CPU (deviation #2)

The plan assumed "likely CPU - no GPU assumed". This machine actually has a GPU (Quadro T1000,
4GB, confirmed via `torch.cuda.is_available()` before training), and `train.py`'s
`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` picked it up
automatically (same auto-detection as week-1, no code change needed). This is *why* the full
4-run sweep finished in ~106 minutes rather than needing the multi-hour CPU budget planned for.

## Dataset subsampling (deviation #3, documented per the task's own instructions)

Benchmarked training speed before committing to a data size: at `d_model=200` (the largest
config), one training step took ~28-36 steps/sec on this GPU. At the *full* ~19.4MB file
(~17.5M training characters after the 90/10 split), one epoch would be ~4272 batches -
comfortably fast per-epoch, but with `train_epochs` up to 40 and 4 runs, a full run risked
several hours unattended for uncertain benefit (TinyStories' text is simple/repetitive; these
tiny 1-2 layer models don't need 17M+ characters to differentiate their capacities).

Chose to subsample to the **first 5,000,000 characters** of the file (`SUBSAMPLE_CHARS` in
`train.py`) - ~4.5M training characters after the split, ~1098 batches/epoch. This kept even the
largest model's epoch under a minute and the full sweep well within a few hours, while still
giving every model far more training characters than it has parameters. All 4 runs use the exact
same subsampled text (and therefore the exact same char-level vocab, V=90) for a clean,
consistent comparison.

## Run scope vs. the README's "updated goal" section

The README's "updated goal" section lists an example sweep of 0.25M/0.5M/1M/2M. Per the task
brief for this build (confirmed with the user ahead of time, not revisited here), the actual
sweep used **0.1M/0.25M/0.5M/1M** instead - still 4 points, same spirit ("that's already enough
to fit a power law"), just shifted down half an octave so the smallest model sits further from
the 250k-1M cluster. `d_model` for each target was hand-derived from the parameter-count formula
in `scaling_configs.py` (assuming V~=100) and came out within ~2% of every target once the real
vocab (V=90 for the subsampled text) was known - no config needed adjusting.

## Results

| run | target N | achieved N | final val loss | FLOPs at best | best step | wall-clock (s) |
|---|---|---|---|---|---|---|
| scale_100k | 100,000 | 100,680 | 1.1555 | 1.060e+14 | 42,822 | 1029.9 |
| scale_250k | 250,000 | 244,032 | 1.1062 | 2.634e+14 | 43,920 | 1135.0 |
| scale_500k | 500,000 | 503,720 | 1.0363 | 5.437e+14 | 43,920 | 1698.7 |
| scale_1m | 1,000,000 | 1,007,600 | 0.9697 | 1.088e+15 | 43,920 | 2482.1 |

**Total wall-clock across all 4 runs: 6361.0s (106.0 min).**

Every achieved N landed within ~2.5% of its target (see `results/scaling_table.md`, and the
notebook's parameter-count sanity table).

### An important caveat about "final" loss here

**None of the 4 runs actually triggered early stopping.** All hit the `train_epochs=40` cap;
`scale_250k`, `scale_500k`, and `scale_1m` all had their *best* epoch as epoch 40 itself (still
improving when training was cut off) - only `scale_100k` peaked one epoch early (epoch 39/40).
That means these are **losses after a fixed, equal epoch budget**, not losses at convergence -
and note that "equal epochs" is not "equal compute": `scale_1m` spent ~10x the FLOPs of
`scale_100k` to reach its (still-improving) final point, purely because it's a bigger model
doing the same number of passes over the same data, not because it was given a compute-optimal
allocation. Under a genuine Kaplan/Chinchilla-style comparison, bigger models are trained with
*more* compute, chosen to match how quickly they're improving - that wasn't done here (all 4 got
the same 40-epoch cap for practical wall-clock reasons). This is one of the mechanisms behind the
mismatch discussed below, in addition to the more obvious ones.

## The fit: alpha

Two fits, both against `(N, best val loss)` for the 4 runs above (see `src/fit_scaling_law.py`):

**Full fit**, `L(N) = L_inf + A*N^-alpha` (nonlinear least squares, `scipy.optimize.curve_fit`):
- **alpha = 0.0756**, SE = 0.4521
- **95% CI (df=1): [-5.669, 5.820]**
- `L_inf` fit to (numerically) 0 - it hit the `L_inf >= 0` boundary constraint, i.e. the data
  gave no evidence for a nonzero irreducible-loss floor in this N range.

**Simple fit**, `L(N) = A*N^-alpha` (L_inf pinned to 0, plain OLS on log-log - a sanity check that
doesn't depend on curve_fit's initial guess or bounds at all):
- **alpha = 0.0768**, R^2 = 0.9774 (df=2)

**Kaplan et al. (2020) report alpha ~= 0.076** for their token-level, BPE-tokenized web-text
sweep.

### The headline number is suspiciously close to Kaplan's. Here's why that's not the finding.

Both of our alphas (0.0756, 0.0768) sit almost exactly on Kaplan's 0.076. **This is very likely
coincidence, not agreement**, for one overriding reason: **the 95% CI on the full fit is
[-5.67, 5.82]**. With 4 data points and 3 free parameters (`L_inf`, `A`, `alpha`), there is
exactly **1 residual degree of freedom** - the correct CI multiplier is the t-distribution's
97.5th percentile at df=1 (`scipy.stats.t.ppf(0.975, df=1) ~= 12.71`), not the usual
large-sample 1.96. That's ~12.7x wider than a naive normal-approximation CI would report. An
interval spanning from strongly negative to strongly positive alpha is consistent with almost
any relationship between N and loss, including *no* relationship - it cannot distinguish "alpha
really is ~0.076" from "alpha is 0.3, or 0.02, or slightly negative, and we got unlucky/lucky
with 4 points." The point estimate landing near Kaplan's is a fun fact worth reporting, not
evidence of anything. (The simpler 2-parameter fit's df=2 CI would still be wide, though less
extreme - not computed here since the full fit's df=1 caveat already makes the point.)

**A careful negative result is more useful than a false positive read on this coincidence**, so
here are the concrete, structural reasons alpha *shouldn't* be expected to match Kaplan's even
setting the CI aside:

1. **Width-only sweep, not jointly-scaled.** `num_layers=2`, `num_heads=4`, `block_size=128` are
   fixed across all 4 runs; only `d_model` varies (see `scaling_configs.py`'s docstring for the
   reasoning). Kaplan et al. scale width *and* depth together as N grows. Our alpha describes
   "what happens as you widen this one exact 2-layer architecture," not the jointly-optimized
   family of architectures Kaplan's alpha describes - there's no a priori reason those two
   exponents should match.
2. **Character-level vocab vs. Kaplan's BPE tokenizer.** Our "tokens" (for the 6*N*tokens FLOPs
   estimate) are individual characters (~90 of them); Kaplan's are BPE subword tokens. A
   character carries far less information per token than a BPE token does, which changes the
   effective compute-per-unit-of-content ratio and thus how loss-vs-FLOPs should be expected to
   trade off, independent of anything about model capacity.
3. **N range far below Kaplan's fitted regime.** This sweep spans 100K-1M parameters - roughly
   one order of magnitude. Kaplan's fit is supported across many orders of magnitude (their
   paper's smallest models are a similar order of magnitude to ours, but the fit's stability
   comes from spanning up to ~1B+ parameters). A power law fit against one decade of N, with 4
   points, is a fundamentally weaker measurement than theirs regardless of the CI issue above.
4. **Small, repetitive dataset, not equal-compute-optimal training.** 4.5M training characters,
   equal epoch budgets rather than compute-optimal token counts per model size (see the caveat
   above) - none of these 4 runs are "converged" in the sense Kaplan's fitted points are.
5. **Only 4 points, 1 residual degree of freedom.** Restated because it's the dominant factor:
   even a "clean-looking" log-log plot (and ours does look clean - R^2=0.977 on the simple fit)
   with only 4 points cannot statistically pin down a 3-parameter nonlinear fit. The near-exact
   numeric agreement with 0.076 should be read as "this experiment cannot rule out Kaplan's
   value," not "this experiment confirms it."

## Compute-optimal frontier

Pooling every `(FLOPs, val_loss)` point logged across all 4 runs' `loss_history.json` (160 points
total, not just the 4 final ones) and taking the lower envelope gives 51 frontier points
(`results/compute_frontier.png`). At low compute budgets the frontier is dominated by
`scale_100k` (the smallest model gets to a given loss cheaper per-FLOP early on, since it takes
fewer FLOPs per epoch) before larger models take over at higher budgets - qualitatively the same
"smaller models are compute-efficient at small budgets" shape the scaling-laws literature
describes, though again not to be over-read quantitatively given points 4-5 above.

## Qualitative check: does bigger actually look better?

The notebook's generation playground (`Once upon a time...`, temperature 0.8) shows a genuine
qualitative trend alongside the loss numbers: `scale_100k`'s output is noticeably more garbled
("Once upon a time, there was a little boy move. The drepla were best unie smiled...") than
`scale_1m`'s ("Once upon a time, there was a little girl. She was very brave for was a held to go
home. She was so excited to exploring..." - still imperfect English, but far more coherent,
correctly-spelled, and grammatically structured). The loss numbers aren't just an abstract curve
- they track a real, visible difference in output quality.

## Summary

- Fitted alpha: **0.0756** (full 3-param fit, 95% CI [-5.67, 5.82], df=1) / **0.0768** (simple
  2-param OLS sanity check, R^2=0.977, df=2), vs. Kaplan's ~0.076.
- The near-exact numeric match is coincidental given the CI - not a meaningful confirmation.
- The real result is the CI itself, plus the five structural reasons above (architecture-family
  mismatch, tokenizer mismatch, narrow N range, non-compute-optimal training on a small dataset,
  and the fundamental 4-point/df=1 statistical limit) for why this experiment *couldn't* have
  cleanly reproduced Kaplan's number even with a "perfect" fit - which is the point of the
  exercise: scaling laws are fitted models with confidence intervals, not folklore constants.
