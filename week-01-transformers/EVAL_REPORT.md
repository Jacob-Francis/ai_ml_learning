# Week 1 Transformer — Eval Report

Reviewed: `src/mytransformer.py`, `src/main.py`, `src/train.py`, `src/ablate.py`, `src/test_attention.py`.
Papers fetched and checked directly (arXiv PDFs, not memory): Vaswani et al. 2017, Su et al. 2021 (RoFormer), Zhang & Sennrich 2019 (RMSNorm), Shazeer 2020 (GLU Variants).

## Summary verdict

The math is right. Every formula I checked against its source paper — scaled dot-product attention, causal masking, RoPE's frequency/pairing/rotation, RMSNorm, SwiGLU — matches the paper exactly, including the less-obvious details (RoFormer's adjacent-pair convention vs. rotate-half, Shazeer's W/V naming, the 2/3 hidden-dim rescaling). There's no algorithmic bug anywhere in the four ablation swaps, and I traced the config flags end-to-end (`ablate.py` → `train.py` → `main.py` → `TransformerBlock`) to confirm all three swaps actually take effect.

Two real, concrete issues, both minor and neither a training-correctness risk:
1. **One docstring is factually backwards** (`mytransformer.py:285-286`) — it claims to be calling "PyTorch's built-in function" when it's actually the hand-rolled implementation the built-in is tested against. This is the kind of comment that would genuinely mislead a learner re-reading the code later.
2. **A one-token off-by-one in `get_batch`'s sampling range** (`train.py:55-56`) — never causes an out-of-bounds read, just silently excludes the single highest valid window from ever being sampled. Functionally negligible, but it's exactly the class of bug the task asked me to check for, so flagging precisely.

Everything else is nits: a couple of typos, a stale "Returns: output, attention_weights" doc that only returns one value, an unnecessarily double-unsqueezed test helper, and a ~3% parameter-count drift in SwiGLU from rounding to a multiple of 32.

## Correctness findings

### 1. Scaled dot-product attention & scaling constant — PASS

Vaswani et al., eq. 1: `Attention(Q,K,V) = softmax(QKᵀ/√dk)V`.

`mytransformer.py:298`:
```python
q_dot_k = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
```
Correct scaling constant, correct transpose axis (`-2,-1`, i.e. the seq_length/d_k axes, not batch/head). Matches.

### 2. Causal masking — PASS

`main.py:135`: `mask = torch.tril(torch.ones(seq_length, seq_length)).bool()` — lower-triangular including the diagonal, so position `i` attends to `j <= i`. `mytransformer.py:301`: `masked_fill(mask == 0, -inf)` before softmax. No off-by-one: the diagonal (self-attention) is correctly allowed, strictly-future positions are correctly `-inf`'d. Matches Vaswani's description ("prevent positions from attending to subsequent positions").

### 3. Weight tying — PASS

`main.py:83`: `self.head.weight = self.token_embedding.weight`. `nn.Embedding.weight` is `(vocab_size, d_model)`; `nn.Linear(d_model, vocab_size).weight` is `(out_features, in_features) = (vocab_size, d_model)` — shapes already line up, no transpose needed, as the comment correctly notes. Matches Vaswani §3.4 ("we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation").

One real deviation from the paper worth naming explicitly (not a bug): Vaswani §3.4 also says *"In the embedding layers, we multiply those weights by √d_model"* — this repo doesn't do that scaling anywhere. Instead it re-derives GPT-2-style `std=0.02` init for the tied embedding (`main.py:37-43`, with a good comment explaining why — unscaled `std=1.0` blew up first-step loss to ~159). This is a legitimate, self-consistent substitution (√d_model scaling is tied to the original paper's Xavier init + Noam LR schedule, which this repo doesn't use), same category as the other intentional post-norm→pre-norm-style deviations — just flagging it since it's not called out anywhere in comments the way the norm/MLP/positional swaps are.

### 4. RoPE — frequency, pairing, v not rotated — PASS

Su et al. eq. 15: `θᵢ = 10000^(-2(i-1)/d)` for `i ∈ [1, d/2]` (1-indexed), rotating adjacent pairs `(x₁,x₂), (x₃,x₄), ...`.

`mytransformer.py:135-136`:
```python
i = torch.arange(0, d_k, 2, ...)          # 0, 2, 4, ..., d_k-2
inv_freq = 1.0 / (10000 ** (i / d_k))
```
Re-indexing the paper's 1-indexed `θᵢ` to 0-indexed `j = i-1` gives `θⱼ = 10000^(-2j/d)`; the code's `i` variable steps by 2 so `i/d_k = 2j/d_k` for `j = 0..d_k/2-1` — identical formula. Confirmed correct, including matching the long-term-decay convention Su et al. §3.3 specifically calls out (`θᵢ = 10000^(-2i/d)`).

Pairing: `x1 = x[..., 0::2]`, `x2 = x[..., 1::2]` (`mytransformer.py:160-161`) — adjacent-dimension pairs, matching Su et al. eq. 15's block-diagonal rotation matrix `R^d_{Θ,m}` (pairs `(1,2),(3,4),...`) and eq. 34's efficient realization. This is deliberately *not* the LLaMA "rotate-half" convention (which pairs `x[i]` with `x[i+d/2]`) — correctly noted in the code comment, and correct given Su et al. is the cited paper.

Rotation: `rotated_x1 = x1*cos - x2*sin; rotated_x2 = x1*sin + x2*cos` (`mytransformer.py:167-168`) matches eq. 13's `[[cosθ,-sinθ],[sinθ,cosθ]]` exactly.

v is never rotated — `TransformerBlock.multi_head_attention` (`mytransformer.py:268-271`) only calls `apply_rope` on `q` and `k`. Correct: RoPE only needs to affect the `q·k` inner product (Su et al. eq. 2), and rotating v would just be lossy noise on content.

### 5. RoPE relative-position property — PASS (verified algebraically)

Su et al. eq. 16: `qₘᵀkₙ = xᵀWq R_{Θ,n-m} Wk x`, using `R_{Θ,m}ᵀR_{Θ,n} = R_{Θ,n-m}` (rotation matrices compose by angle subtraction). This holds because `R_{Θ,m}` is block-diagonal with orthogonal 2×2 rotation blocks per frequency, and each block's composed transpose-product depends only on `n-m`. The code's `apply_rope` implements exactly this per-block rotation, so the property carries over. Confirmed both on paper and empirically via the test (see Test coverage notes).

### 6. RMSNorm — PASS

Zhang & Sennrich, eq. 4: `āᵢ = aᵢ/RMS(a) · gᵢ`, `RMS(a) = √((1/n)Σaᵢ²)`. No mean subtraction, no bias — the paper is explicit that RMSNorm "totally remov[es] the mean statistic ... at the cost of sacrificing [re-centering] invariance."

`mytransformer.py:57-58`:
```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
return x / rms * self.weight
```
Exact match (the `+eps` is a standard numerical-stability addition not present in the paper's idealized formula, same as every real-world implementation including the paper's own released code). `self.weight` initialized to ones (`mytransformer.py:53`) matches the paper's `g` "set to 1 at the beginning." Correct, and correctly has no bias parameter anywhere in the class.

### 7. SwiGLU — PASS, including the notation-mapping check

Shazeer eq. 6: `FFN_SwiGLU(x,W,V,W2) = (Swish₁(xW) ⊗ xV) W2` — i.e. `W` is the *gated* (activated) matrix, `V` is the *un-activated* value matrix that gets multiplied through.

`mytransformer.py:90-96`:
```python
self.gate = nn.Linear(d_model, d_ff, bias=False)  # activated
self.up   = nn.Linear(d_model, d_ff, bias=False)  # unactivated
...
def forward(self, x):
    return self.down(self.silu(self.gate(x)) * self.up(x))
```
This maps `gate → W`, `up → V`, `down → W2`: `down(silu(gate(x)) * up(x))` = `W2·(Swish(xW) ⊗ xV)`, exactly eq. 6. The task flagged that Shazeer's naming could be a trap (which matrix is "the gate" isn't obvious from the paper's `W,V` alone) — the code gets it right and the comment (`mytransformer.py:75-79`) correctly explains *why* it renamed `V`→`up` rather than just copying the paper's letters. Bias-free on all three matrices also matches: "we omit the bias terms" (Shazeer §2).

Sizing rationale: Shazeer §2, verbatim — *"To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units dff ... by a factor of 2/3."* `swiglu_hidden_dim` computes `raw = int(8*d_model/3)` which is `(4*d_model)*(2/3)` simplified — matches. I checked the arithmetic holds exactly at the baseline config (`d_model=256`): GELU MLP param count = `2 * d_model * 4*d_model = 524,288`; SwiGLU with the *unrounded* `d_ff = 682.67` would give `3 * d_model * d_ff ≈ 524,288` — exact parity. After rounding up to a multiple of 32 (`d_ff = 704`), actual SwiGLU count is `3*256*704 = 540,672`, about **3.1% more** params than the GELU baseline. This is a real, if small, deviation from "matched parameter count" that the ablation table's story depends on somewhat — the code's own comment already flags the rounding as "a minor efficiency nicety, not required for correctness," but doesn't quantify the resulting drift. Worth knowing when interpreting the swiglu run's val-loss delta: a small fraction of any improvement could be attributable to the extra ~3% capacity rather than the architecture change itself. Minor nit, not a bug.

### 8. Baseline block: pre-norm, residuals — PASS

`mytransformer.py:220-225`:
```python
x_attained = self.multi_head_attention(self.norm1(x), mask=mask)
x = x + x_attained
x_ff = self.mlp(self.norm2(x))
x = x + x_ff
```
Correct Pre-LN: normalize before the sublayer, add the *un-normalized* `x` back as the residual. This is a known, intentional deviation from Vaswani's post-LN (`LayerNorm(x + Sublayer(x))`, §3.1) — correctly implemented for the pre-norm variant it's aiming for, and the code comment is accurate about what it's doing.

### 9. `train.py get_batch` — off-by-one in sampling range (minor, non-crashing)

`train.py:55-56`:
```python
max_start = len(data) - block_size - 1
start_idxs = torch.randint(0, max_start, (batch_size,))
```
Target offset is correct: `targets[t] = data[i+1+t]` is exactly `inputs[t]`'s next character (`train.py:58-59`) — no shift bug.

The bound, though, is off by one. The true valid range for a start index `i` is `0 <= i <= len(data) - block_size - 1` inclusive (so that `targets = data[i+1:i+1+block_size]` stays in bounds — the tightest constraint of the two slices). As an **exclusive** upper bound for `torch.randint(low, high)`, that range needs `high = len(data) - block_size` (not `- 1`). The code's `max_start = len(data) - block_size - 1` means `torch.randint(0, max_start)` only ever draws up to `max_start - 1 = len(data) - block_size - 2` — one short. So the single highest valid window (`i = len(data) - block_size - 1`) is never sampled.

This is not an out-of-bounds read (the task specifically asked me to check that) — I verified with a concrete example (`len(data)=100, block_size=10` → `max_start=89`, valid range should include `i=89`, but `randint(0,89)` tops out at 88) that no read ever exceeds `data`'s bounds. It's purely a lost window, functionally invisible at TinyShakespeare's scale (one position out of ~1M), but it is a genuine off-by-one worth a one-line fix (`max_start = len(data) - block_size` given the current exclusive-`randint` usage).

### 10. RoPE evenness of `d_k` — unguarded edge case (minor)

`rope_cos_sin`/`apply_rope` implicitly require `d_k` (= `d_model // num_heads`) to be even — `x[..., 0::2]` and `x[..., 1::2]` only have equal length when `d_k` is even. There's no assertion for this anywhere (the existing `assert d_model % num_heads == 0` only guards divisibility, not evenness of the quotient). Every config actually used in `ablate.py`/`train.py` happens to produce an even `d_k` (256/8=32), so this never triggers, but it's a latent crash (`RuntimeError` from a shape mismatch in `x1*cos - x2*sin`) waiting for a future hyperparameter change. Worth a one-line assert if the RoPE path is reused later.

### 11. Ablation swap wiring — PASS

Traced end-to-end: `ablate.py:42-47` (`RUNS` dict overrides exactly one `BASE_CONFIG` field per run) → `train.py:157-166` (`run_training` forwards `positional_embedding_type`/`norm_type`/`mlp_type` unchanged into `MyTransformer(...)`) → `main.py:52-60` (branches on `positional_embedding_type` to build or skip the learned position table) and `main.py:62-70` (passes the same three flags into every `TransformerBlock`) → `mytransformer.py:194-212` (`make_norm`/MLP branch) and `mytransformer.py:268-271` (RoPE branch inside attention). All three swaps are live and correctly isolated — no swap silently falls through to the baseline behavior.

## Readability findings

### Comment/docstring bugs (real, not style)

- **`mytransformer.py:285-286` — docstring is factually backwards.** The method:
  ```python
  def scaled_dot_product_attention(self, q, k, v, mask=None):
      """
      Compute the scaled dot-product attention using PyTorch's built-in function.
      This is used for testing against the custom implementation.
      ...
  ```
  is itself the hand-rolled, from-scratch implementation (manual `matmul`/`masked_fill`/`softmax`/`matmul`) — it calls no PyTorch built-in attention function at all. `test_attention.py:11-14` confirms this is exactly backwards: `myattention = myclass.scaled_dot_product_attention` (this method) is tested *against* `trueattention = nn.functional.scaled_dot_product_attention` (the actual built-in). A learner reading this method in isolation would come away thinking it's a thin wrapper around `F.scaled_dot_product_attention`, when it's actually the core deliverable of task 1 in the README. This is the single most confusing comment in the codebase and worth fixing — swap the docstring to describe the manual computation, and note *that* it's what gets tested against the built-in.

- **`mytransformer.py:244-246` — stale "Returns" doc.** `multi_head_attention`'s docstring claims:
  ```
  Returns:
      output: The result of the attention mechanism.
      attention_weights: The attention weights.
  ```
  but the function only `return output` (`mytransformer.py:282`) — a single value. Attention weights are stashed as a side-effect attribute (`self.last_attention_weights`, `mytransformer.py:308`) inside the *other* method, not returned here. Minor, but it's exactly the "comment overclaims" pattern the task asked me to look for.

- Typos in `multi_head_attention`'s docstring (`mytransformer.py:231-232, 238`): "with ocncatenation" (concatenation), "an the full multi head attention" (and the), "Intermidates" (Intermediates). Cosmetic, but in a from-scratch teaching repo these small errors stand out on a close read.

### Structure / clarity

- `SwiGLU`'s docstring (`mytransformer.py:70-86`) and the `gate`/`up` naming are genuinely good teaching comments — they preemptively resolve the exact notation ambiguity the task flagged (Shazeer's paper calls the activated matrix `W` and the passthrough matrix `V`, with no inherent "gate"/"value" labels), and explain *why* the code renames `V`→`up` rather than parroting the paper's letters. Worth calling out as a positive example, not just absence-of-bugs.

- `test_attention.py`'s `_rotate_at_position` helper (`test_attention.py:51-55`) double-unsqueezes `cos`/`sin` before passing them into `apply_rope`, which *also* unsqueezes internally (`mytransformer.py:164-165`). I confirmed empirically that the resulting rotated tensor ends up with shape `(1,1,1,1,1,4)` instead of the expected `(1,1,1,4)` — six dims instead of four, all the extra ones trivially size-1, so the final `.sum()`/`.norm()` calls downstream still produce correct numbers by accident of broadcasting. It's harmless today but genuinely confusing if a learner tries to trace tensor shapes through this helper by hand (as the README explicitly asks students to do for the RoPE derivation) — the extra unsqueezes in the helper are unnecessary and should be dropped so `apply_rope`'s own broadcasting logic is the only one in play.

- `x_attained` (`mytransformer.py:220`) and the comment `# attain; dot product over d_k...` (`mytransformer.py:296`) are a bit cryptic as names/wording — presumably a play on "attention," but it reads oddly on first pass compared to the otherwise very clear comments elsewhere in the file (e.g. the RMSNorm/SwiGLU/RoPE docstrings, which are excellent). Not wrong, just a small dip in an otherwise consistently good comment style.

- Per the task's framing, the per-head `setattr(self, f'q_linear_{i}', ...)` pattern (`mytransformer.py:188-191`) and the `torch.zeros(...)` + per-head loop assembly in `multi_head_attention` (`mytransformer.py:251-257`) are intentional, known-inefficient-but-correct choices, not flagged as a mistake. Worth noting only that it does cost some readability relative to a single batched `nn.Linear(d_model, num_heads*d_k)` + `.view()`/`.reshape()` — a learner comparing this against nanoGPT (which the README explicitly invites) will notice the difference immediately. That tradeoff is presumably exactly the point of doing it by hand first.

## Test coverage notes

- **Core attention math**: tested against `torch.nn.functional.scaled_dot_product_attention` to `atol=1e-5` (`test_attention.py:26-29`), matching the README's task-1 requirement exactly (same tolerance, same reference function, `attn_mask=None` on both sides for a clean comparison).
- **RoPE norm-preservation** (`test_attention.py:46-48`): a real check, but a weak one — any orthogonal 2×2 block transform (not just a true rotation; e.g. a sign-flipped reflection like `[[cosθ,-sinθ],[-sinθ,-cosθ]]`) would also pass this test, since norm-preservation only requires orthogonality, not the specific rotation direction. It's a fine cheap regression check but wouldn't catch every plausible sign-flip bug on its own.
- **RoPE relative-position invariance** (`test_attention.py:58-66`): this is the test that actually does the work. It uses two *distinct* random content vectors for q and k (avoiding the trivial "self dot-product is always the same length" case), checks three different `(i,j)` pairs that all share offset `i-j=3` at different absolute positions (5,2), (10,7), (30,27), and asserts the resulting dot products agree to `1e-3`. This would catch a broken frequency table, a q/k pairing mismatch, or an inconsistent rotation, though — as noted above in finding 5 — it wouldn't by itself distinguish "adjacent-pair" from "rotate-half" pairing, since both conventions independently satisfy the same relative-position algebra as long as q and k use the same convention. That's fine here since the paper-fidelity choice (adjacent pairs) is a design decision, not something the test needs to enforce.
- **Gaps**: no dedicated unit test exercises the full `multi_head_attention` wrapper end-to-end (the per-head projection loop + RoPE branch + concat), `RMSNorm`, `SwiGLU`, the causal-mask off-by-one, or weight tying — these are only exercised indirectly via full training runs in `train.py`/`ablate.py`. Given the README's explicit ask was specifically "unit-test attention... assert outputs match," this is in scope and satisfied; the other components just don't have the same direct-comparison treatment, which is worth knowing if a bug ever creeps into RMSNorm or SwiGLU — there's currently nothing that would catch it before a full training run.
