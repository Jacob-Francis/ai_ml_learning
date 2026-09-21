# Week 1 ablation results

One swap at a time from a shared baseline (LayerNorm + learned positional embeddings + GELU MLP). See `ablate.py` for exact hyperparameters.

| run | swap | final val loss | best epoch | wall-clock (s) |
|---|---|---|---|---|
| baseline | (none) | 1.5579 | 20 | 1644.6 |
| rmsnorm | LayerNorm -> RMSNorm | 1.5706 | 34 | 2672.1 |
| rope | learned pos. emb. -> RoPE | 1.5419 | 10 | 1027.6 |
| swiglu | GELU MLP -> SwiGLU | 1.5994 | 49 | 3503.2 |

Total wall-clock across all 4 runs: 8918.8s
