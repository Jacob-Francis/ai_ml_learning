# Week 1 — The Transformer

The architecture everything else sits on. Read it once as a paper, then once as a spec you are going to implement.

**Core**
- Vaswani et al. (2017), *Attention Is All You Need* — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**Supporting**
- Zhang & Sennrich (2019), *Root Mean Square Layer Normalization* — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Shazeer (2020), *GLU Variants Improve Transformer* — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)

The last two are two-page papers describing changes that are in essentially every model shipped since 2022. Worth reading precisely because they show how much of the field is small empirical deltas rather than theory.

**Things to work out on paper**
- Why the $1/\sqrt{d_k}$ scaling, in terms of the variance of the dot product of two random vectors
- The arithmetic and memory complexity of attention in sequence length, and where FlashAttention's speedup actually comes from (it is not fewer FLOPs)
- RoPE as a rotation acting on 2D subspaces of the query/key vectors, and why relative position falls out of the inner product

**Build: a decoder-only transformer from scratch**

PyTorch, no `nn.Transformer`, no HuggingFace. Character-level, trained on TinyShakespeare (~1MB, trains in minutes on a T4).

1. Implement multi-head attention yourself. Unit-test it against `torch.nn.functional.scaled_dot_product_attention` — assert outputs match to within `1e-5`.
2. Build the full block: pre-norm, residuals, MLP, causal mask, weight tying.
3. Train to a sensible loss, sample some text.
4. **Ablate.** Swap LayerNorm → RMSNorm, learned positional embeddings → RoPE, GELU MLP → SwiGLU. Four training runs, one table of final validation loss and wall-clock time.

Ship: `model.py`, `train.py`, `test_attention.py`, a loss curve, an ablation table, and some generated text.

Karpathy's nanoGPT is the reference if you get stuck, but write yours first and diff afterwards.
