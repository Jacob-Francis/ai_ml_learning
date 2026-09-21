# Week 2 core papers — summary

Covers the 3 core papers only (Kaplan, Chinchilla, Mamba-2). Written so you
can scan it instead of reading all three in full. Figures are called out
where it's worth opening the paper just to look at the picture.

---

## 1. Kaplan et al. (2020) — Scaling Laws for Neural Language Models

**[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)**

### 🔑 Key point
Loss drops as a clean power law with model size (N), data (D), and compute
(C) — and **bigger models learn much faster per token**. So the "optimal"
strategy this paper argues for is: train huge models, feed them relatively
little data, and stop well before convergence.

---

### 📋 The numbers
- `L(N) = (N_c/N)^α_N`, fitted **α_N ≈ 0.076**, N_c ≈ 8.8×10¹³.
- Similar fits for data (α_D ≈ 0.095) and compute (α_C ≈ 0.05–0.057).
- Compute-optimal split: as compute grows, **model size should grow much
  faster than data** — roughly N ∝ C^0.73 vs only S (training steps) ∝ C^0.03.
- Model *shape* (wide-shallow vs narrow-deep) barely matters once total
  params are fixed — a 6-layer wide model and a 48-layer narrow model land
  within 3% of each other.

---

### Figures worth opening
- **Figure 1** — the three power-law curves (loss vs N, D, C) in one plot.
  This is the paper's whole thesis in one picture.
- **Figure 14** — compute-optimal allocation chart. Look at how steeply the
  "model size" line climbs compared to the almost-flat "training steps"
  line — that visual gap *is* the "go big, stop early" claim.

---

### ⚠️ Caveat — this got revised
This paper's "prioritize model size" guidance was substantially overturned
by Chinchilla (next section). **Chinchilla's** own critique of Kaplan's
methodology (not a self-admission in Kaplan's paper) argues the fixed
learning-rate schedule wasn't re-tuned for each run length, which likely
biased the fitted exponents toward favoring bigger models — see the caveat
under Chinchilla below for the sourced version of this claim. Treat
α_N ≈ 0.076 as "the number this specific setup produced," not a universal
constant — which is exactly the spirit of your build task.

---

## 2. Hoffmann et al. (2022) — Chinchilla

**[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)**

### 🔑 Key point
Directly contradicts Kaplan: compute-optimal training scales **model size
and data equally** (each roughly ∝ compute^0.5), not model-size-heavy. The
often-quoted "~20 tokens per parameter" isn't a rule stated in the paper —
it's just what falls out of Chinchilla's own specific model (70B params,
1.4T tokens) and later became popular shorthand for the equal-scaling
principle.

---

### 📋 The numbers
- Fits `L(N,D) = E + A/N^α + B/D^β` — note the extra **E term**, an
  irreducible loss floor Kaplan's simpler formula doesn't have.
  Fitted: E=1.69, A=406.4, B=410.7, α=0.34, β=0.28.
- Three independent methods (training-curve envelopes, IsoFLOP curves, and
  this parametric fit) all agree: **N ∝ C^0.5, D ∝ C^0.5** — a 50/50 split,
  not Kaplan's 0.73/0.27 (N-vs-D exponents; Kaplan's separate training-steps
  exponent of 0.03 is a different number, not the one being compared here).
- Proof: Chinchilla (70B params, 1.4T tokens) beats Gopher (280B params,
  300B tokens) at the *same* training compute — +7.6 points on MMLU
  (67.6% vs 60.0%).

---

### Figures worth opening
- **Figure 3** — IsoFLOP curves. Loss vs. model size at several fixed
  compute budgets, each tracing a parabola — the bottom of each parabola is
  that budget's optimal model size. This is also the figure that marks
  Gopher's compute budget and projects where it *should* have landed
  (~40B params) vs where it actually landed (280B, badly oversized for its
  token count).
- **Figure 4** — the parametric-fit view: loss contours from `L(N,D)` with
  the "efficient frontier" traced through them — the cleanest single picture
  of what compute-optimal looks like across a continuous range of budgets.

---

### ⚠️ Where the Kaplan disagreement actually comes from
Per the paper itself (Section 2 / Appendix B): **Kaplan used a training
schedule not matched to each run's length** — the cosine LR decay should
finish exactly when training stops, and Kaplan's shorter runs weren't given
a schedule tuned to their length, inflating their loss and skewing the
fitted exponents toward "bigger model" conclusions. Secondary factor:
Kaplan's models topped out smaller (~1.5B params vs Chinchilla's up to 16B).
This is directly the kind of "experimental hygiene" issue your README
points at.

---

## 3. Dao & Gu (2024) — Mamba-2 (Structured State Space Duality)

**[arXiv:2405.21060](https://arxiv.org/abs/2405.21060)**

### 🔑 Key point
Proves that **state-space models (SSMs) and a class of structured, linear
(non-softmax) attention variants are the same underlying computation**, just
computed two different ways — this does *not* cover standard softmax
attention, which the framework treats as a separate problem, not a member of
the equivalence class. That insight ("SSD") lets Mamba-2 reuse fast matmul
hardware tricks, making it 2–8× faster than the original Mamba.

---

### 📋 The numbers
- Both SSMs and this attention class reduce to multiplying by a "structured
  semiseparable matrix" — computed as a linear-time recurrence (the SSM
  way) or a quadratic attention-like form (the Transformer way).
- Mamba-2's architecture change: head dimension grows from 1 (Mamba-1) to
  64–128, much more Transformer-like — this is what unlocks the hardware
  speedup.
- 6× faster than FlashAttention-2 at 16K sequence length (crossover point
  ~2K tokens).
- At 2.7B params / 300B tokens, Mamba-2 beats Mamba-1, Pythia-2.8B, and even
  the larger Pythia-6.9B on quality.

---

### Figures worth opening
- **Figure 1** — the core SSD diagram, illustrating the relationship between
  state space models and attention through structured matrices.
- **Figure 3** — "structured masked attention," showing how linear-attention
  variants (e.g. RetNet) fall out as a special case of this framework.

---

### ⚠️ Caveat
This paper does **not** discuss hybrid attention+SSM architectures (like
Nemotron's alternating layers, mentioned in the week-02 README) — that's a
trend other papers/models built on top of this one, not something Mamba-2
itself proposes.

---

## How this connects to your build task

- ✅ The build's fitted α (0.076, full fit) landed almost exactly on
  Kaplan's 0.076 — but with only 4 data points fitting 3 parameters, the
  95% confidence interval is `[-5.67, 5.82]`, wide enough to be consistent
  with nearly any exponent. **That CI is the real finding here, not the
  point estimate** — it's a textbook illustration of the Kaplan/Chinchilla
  lesson above: a clean-looking number from an underpowered fit shouldn't
  be trusted at face value. Full breakdown in `week-02-scaling/notes.md`.
