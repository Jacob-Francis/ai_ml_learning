# Week 5 — Retrieval and RAG

The most commonly deployed LLM system pattern, and the one where the gap between the naive implementation and a good one is largest.

**Core**
- Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering* — [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Khattab & Zaharia (2020), *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* — [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

**Supporting**
- Gao et al. (2022), *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE) — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts* — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- Asai et al. (2023), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

Notice that DPR's contribution is in large part the *training procedure* — in-batch negatives and hard negative mining — rather than the architecture. ColBERT's late interaction is a nice piece of engineering: it keeps token-level granularity while remaining precomputable, sitting between a bi-encoder and a cross-encoder in both cost and quality.

**Build: RAG over your own paper library**

Index every PDF you have read so far in this programme. Pleasingly recursive, and it means you can evaluate it properly because you know the ground truth.

Write it without LangChain or LlamaIndex. The frameworks hide exactly the parts you are trying to learn.

1. Parse PDFs (`pymupdf`), chunk with overlap, keep metadata (paper title, section, page).
2. Embed with `sentence-transformers` (`bge-small-en-v1.5` or similar), index with FAISS.
3. Implement a BM25 baseline (`rank_bm25`).
4. Implement hybrid retrieval — reciprocal rank fusion over dense and sparse results.
5. Add a cross-encoder reranker over the top 50 (`bge-reranker-base`).
6. **Evaluate.** This is the point of the exercise. Hand-write 30 questions with known source chunks. Report recall@1, recall@5 and MRR for: BM25, dense, hybrid, hybrid+rerank. Also vary chunk size (256/512/1024 tokens).
7. Bolt a generation step on the end with any small instruct model.

Ship the results table. "I built a RAG pipeline" is a sentence on every junior CV in existence; "I measured recall@5 across four retrieval strategies and three chunk sizes on a hand-labelled evaluation set, and here is the table" is not.
