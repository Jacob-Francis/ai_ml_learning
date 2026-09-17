# Week 10 — Inference and Serving

**Papers**
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

**Build**

Implement speculative decoding from scratch with a 0.5B draft model and a 3B target, verify output distributions are identical, measure the speedup and the acceptance rate against draft model quality.
