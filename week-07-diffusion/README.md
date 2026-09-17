# Week 7 — Diffusion and Flow Matching

The most mathematically substantial topic here, and the one where your background is a genuine advantage over a typical bootcamp-trained candidate.

**Papers**
- Ho et al., *Denoising Diffusion Probabilistic Models* — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Song et al., *Score-Based Generative Modeling through SDEs* — [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- Song et al., *DDIM* — [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- Lipman et al., *Flow Matching for Generative Modeling* — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Rombach et al., *Latent Diffusion* — [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

**Build**

Train a DDPM on MNIST from scratch, derive and implement the ELBO, then implement flow matching on the same data and compare sample quality at 10, 50 and 1000 sampling steps. Derive the connection between the score function and the noise prediction target.
