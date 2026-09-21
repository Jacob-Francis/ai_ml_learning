"""
Week 2, task 2: the four model configs for the scaling-law sweep.

Design choice: fix num_layers=2, num_heads=4, block_size=128 across every run, and vary *only*
d_model. This is deliberate, not laziness - Kaplan et al. scale width and depth together (and
also vary a handful of other things), which means their N doesn't isolate any single architecture
knob. Fixing everything except d_model here gives a clean single-axis sweep: the only thing
that changes between these 4 runs is total parameter count, so a power-law fit against N is
actually measuring "what happens as you widen this one exact architecture", not conflating that
with "what happens as you also add depth". The trade-off (spelled out again in notes.md) is that
this is one specific reason our fitted alpha won't match Kaplan's - his alpha describes a
jointly-scaled family of architectures, ours describes a width-only family.

--- Deriving the parameter-count formula ---

The model is `week-01-transformers/src/main.py`'s `Transformer`, with weight tying (the output
head reuses the token embedding matrix instead of learning its own, so it contributes 0 *extra*
parameters - see main.py's `self.head.weight = self.token_embedding.weight`). Let:
    V = vocab_size, B = block_size, D = d_model, L = num_layers

Embeddings (each counted once - the tied head adds nothing new):
    token_embedding:    V * D
    position_embedding: B * D   (nn.Embedding(block_size, d_model), 'learned' mode)

Per transformer block (mytransformer.TransformerBlock), with attention bias=False throughout:
    attention Q/K/V/O:  4 linears, each (D, D)                        -> 4 * D^2
    MLP (mlp_type='gelu', bias=True, d_ff = 4*D):
        Linear(D, 4D):  weight 4D*D + bias 4D                         -> 4D^2 + 4D
        Linear(4D, D):  weight D*4D + bias D                          -> 4D^2 + D
        MLP total:                                                    -> 8D^2 + 5D
    norm1, norm2 (LayerNorm(D)): weight D + bias D, two of them        -> 4D
    block total:  4D^2 + 8D^2 + 5D + 4D = 12D^2 + 9D

L blocks:            L * (12D^2 + 9D)
Final norm_f (LayerNorm(D)):  2D   (weight D + bias D)
head:                 0   (tied to token_embedding, bias=False)

Total:
    N = V*D + B*D + 2D + L*(12D^2 + 9D)
      = D*(V + B + 2) + L*(12*D^2 + 9*D)

This is the `estimate_params` function below. It assumes V ~= 100 as a planning estimate; the
real TinyStories char vocab turned out to be V=98 (see data_download.py output), close enough
that no config needed adjusting - achieved param counts (from `estimate_params` with the real
V=98) land within ~2% of every target, well under the 15%-off threshold that would have called
for tweaking d_model. The *actual* live-counted parameter count (via
`sum(p.numel() for p in model.parameters())` on the real, constructed model - which correctly
dedupes the tied embedding/head weight) is what gets recorded in each run's config.json, not this
hand-derived estimate - this formula is a planning tool, not the number the paper-quality results
should cite.
"""


def estimate_params(d_model, vocab_size=98, block_size=128, num_layers=2):
    """Hand-derived parameter count estimate (see module docstring for the derivation)."""
    D, V, B, L = d_model, vocab_size, block_size, num_layers
    return D * (V + B + 2) + L * (12 * D**2 + 9 * D)


# Shared hyperparameters across all 4 runs - only d_model varies (see module docstring).
SHARED = dict(
    num_layers=2,
    num_heads=4,
    block_size=128,
    positional_embedding_type='learned',
    norm_type='layernorm',
    mlp_type='gelu',
)

# The 4-point sweep. `d_model` values chosen so that, at the real TinyStories char vocab
# (V=98, confirmed by data_download.py / train.py's tokenizer), estimate_params(d_model) lands
# close to each target N - see the table in week-02-scaling/README.md for the original planning
# numbers (computed at the planning-time assumption V~=100) and notes.md for the as-run numbers.
SCALING_RUNS = [
    {"run_name": "scale_100k", "target_n": 100_000, "d_model": 60, **SHARED},
    {"run_name": "scale_250k", "target_n": 250_000, "d_model": 96, **SHARED},
    {"run_name": "scale_500k", "target_n": 500_000, "d_model": 140, **SHARED},
    {"run_name": "scale_1m", "target_n": 1_000_000, "d_model": 200, **SHARED},
]


if __name__ == "__main__":
    # Quick sanity check: print the estimate vs target for each config before committing to a
    # full training run (this is what caught that no d_model needed adjusting for V=98 vs V~=100).
    print(f"{'run_name':<12} {'target_n':>10} {'d_model':>8} {'estimate_n':>12} {'%% off':>8}")
    for cfg in SCALING_RUNS:
        est = estimate_params(cfg["d_model"], block_size=cfg["block_size"], num_layers=cfg["num_layers"])
        pct_off = 100 * (est - cfg["target_n"]) / cfg["target_n"]
        print(f"{cfg['run_name']:<12} {cfg['target_n']:>10,} {cfg['d_model']:>8} {est:>12,} {pct_off:>7.1f}%")
