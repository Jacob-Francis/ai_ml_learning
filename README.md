# Modern AI/ML — A Paper-a-Day Curriculum

A self-directed programme working through the foundations of modern machine learning by reading primary literature and reimplementing the ideas. Six core weeks, 3–5 papers each, one build per week.

**Approach:** read the papers, implement the core idea from scratch where feasible, benchmark against the reference implementation.

---

## How to use this

Each week has:
- **Core papers** — read properly, work through the equations
- **Supporting papers** — skim for the result and the ablation tables
- **Build** — one day maximum, Python, committed as a self-contained subdirectory

Read ML papers differently to maths papers. There are no proofs to verify; the load-bearing content is the *ablation table* and the *experimental setup*. A useful order: abstract → figures → results tables → method → related work. Ask of every paper: what is the baseline, what single thing changed, and how much of the gain survives the ablation?

Keep a `notes.md` per week. Three or four paragraphs: what the paper claims, what the experiment actually shows, what you'd have wanted to see. 

- understanding experimental setup
- understanding datasets
- reading implementation details hidden in appendices
- checking GitHub code
- understanding ablations

### Compute

Everything here fits on a free Colab T4 or Kaggle's 2×T4 (30 hrs/week) if you stay under ~1B parameters. Good model choices for the fine-tuning weeks: `Qwen3-0.6B`, `SmolLM2-360M`, `Llama-3.2-1B`. Don't reach for a 7B model — you will spend the day fighting OOM errors instead of learning.

### Repo structure

```
├── README.md              # this file, plus a progress table
├── week-01-transformers/
│   ├── notes.md           # paper notes
│   ├── src/
│   ├── results/           # plots, tables, logs
│   └── README.md          # what you built, what you found
├── week-02-scaling/
...
```

Put a results table or plot in every week's README. 

---

## Week 1 — The Transformer

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

---

## Week 2 — Scaling, Data, and Architecture Beyond the Transformer

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

---

## Week 3 — Adaptation: Fine-Tuning, PEFT, Quantization

How a pretrained model becomes a useful one. This is the most directly employable week — most applied ML jobs involve adapting an existing model, not training one.

**Core**
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs* — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Wei et al. (2021), *Finetuned Language Models Are Zero-Shot Learners* (FLAN) — [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)

**Supporting**
- Dettmers et al. (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Frantar et al. (2022), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

LoRA rests on the claim that the weight *update* during fine-tuning has low intrinsic rank. Note that this is asserted and then validated empirically rather than proven, and that the paper's rank ablation is the evidence. The NF4 data type in QLoRA is a genuinely elegant piece of work — a quantile-based quantization scheme that is information-theoretically optimal for normally distributed weights.

**Build: implement LoRA, then use the real thing**

1. Write LoRA yourself: an `nn.Module` wrapper holding $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, with $B$ zero-initialised, that patches the attention projections of a HuggingFace model. About 50 lines.
2. Verify: freeze the base model, confirm that trainable parameter count matches theory, and that at step 0 the model's outputs are bit-identical to the base model (this is what zero-init of $B$ buys you).
3. Fine-tune `Qwen3-0.6B` or `SmolLM2-360M` on an instruction dataset (`databricks-dolly-15k` or a slice of `no_robots`) with your implementation, then with HF `peft`, and confirm the loss curves overlap.
" I'd instead use Alpaca-cleaned or TinyOrca with 1–2k examples."
4. Sweep rank $r \in \{1, 2, 4, 8, 16, 64\}$. Plot trainable parameters against validation loss. Find where the returns flatten.
5. Generate ten before/after completions and put them in the README.

The rank sweep plot is the deliverable. It answers a question people actually ask in production.

---

## Week 4 — Alignment and RL Post-Training

How base models become assistants, and — since 2025 — how they learn to reason. This is where reinforcement learning enters, and where your mathematics background pays off, because the derivations are the substance.

**Core**
- Ouyang et al. (2022), *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

**Supporting**
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Shao et al. (2024), *DeepSeekMath* — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) — introduces GRPO, section 4 is the relevant part
- Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences* — [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) — the origin of the whole RLHF line

**Derive these yourself**
- The DPO objective from the KL-constrained reward maximisation problem. The closed-form optimal policy, the change of variables that eliminates the reward model, and the Bradley-Terry substitution. It is about a page and it is the single most worthwhile derivation on this list.
- GRPO as PPO with the learned value function replaced by the empirical mean of a group of sampled completions. Understand what this buys (no critic network) and what it costs (higher variance, needs a verifiable reward).

The shift from RLHF (learned reward model, human preferences) to RLVR (verifiable rewards — does the code compile, is the answer correct) is the defining change in post-training over the last two years.

**Build: DPO from scratch, then GRPO on a toy task**

Morning:
1. Implement the DPO loss in about ten lines: `-logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))`.
2. Sanity-check on a two-armed toy problem where you know the answer analytically: does the implicit reward converge to the true preference gap?
3. Run DPO on your Week 3 model using `trl` with `ultrafeedback-binarized` (subsample to a few thousand pairs). Plot the implicit reward margin between chosen and rejected over training.

Afternoon (the more interesting half):
4. GRPO on Countdown, a small arithmetic-puzzle task: given four numbers, reach a target using `+ - * /`. The reward is a verifier, not a model — 1 if the expression evaluates to the target, 0.1 for correct formatting, 0 otherwise.
5. Use `trl`'s `GRPOTrainer` with a 0.5B model. Plot reward and mean completion length against training step.

If completion length creeps up as reward improves, you have reproduced in miniature the central observation of DeepSeek-R1: models given room and a verifier learn to think for longer. That plot is a genuinely good repo artefact.

---

## Week 5 — Retrieval and RAG

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

---

## Week 6 — Agents, Tool Use, and Test-Time Compute

The current frontier of applied work: what happens when you let a model run in a loop with tools and a budget.

**Core**
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models* — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning* — [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

**Supporting**
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning* — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools* — [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Jimenez et al. (2023), *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)

Self-consistency is worth thinking about statistically: sampling $k$ chains and taking the majority answer is a variance reduction technique, and the accuracy-versus-$k$ curve has a shape you can reason about. It is also the conceptual ancestor of every test-time-compute scaling result since.

**Build: a ReAct agent, no framework**

1. Write the loop yourself: a prompt template, a parser for `Thought:` / `Action:` / `Observation:`, a tool dispatch table, a step limit. Perhaps 150 lines.
2. Three tools: a sandboxed Python executor, the Week 5 retriever, and a web search.
3. Add a Reflexion layer — on failure, ask the model to write a short post-mortem, append it to the context, retry once.
4. **Evaluate.** Take 20–30 questions from HotpotQA (multi-hop, so retrieval genuinely helps) or GSM8K. Compare four conditions: direct answering, chain-of-thought, ReAct, ReAct+Reflexion. Report accuracy, mean tool calls, mean tokens.
5. Log every trajectory to JSON and commit a few, including failures.

The failure trajectories are the interesting content. Categorise them — parser failures, tool errors, loops, wrong retrieval — and write a paragraph on what you would fix. That paragraph is what distinguishes someone who has built an agent from someone who has watched a tutorial.

Use whatever model API you have access to. If you want it fully local, a 4B instruct model will work, badly, which is itself instructive.

---

## Where to go next

Four extension weeks, in rough order of how much they'd differentiate you given a maths PhD.

**Week 7 — Diffusion and Flow Matching.** The most mathematically substantial topic here, and the one where your background is a genuine advantage over a typical bootcamp-trained candidate. Ho et al., *Denoising Diffusion Probabilistic Models* ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239)); Song et al., *Score-Based Generative Modeling through SDEs* ([arXiv:2011.13456](https://arxiv.org/abs/2011.13456)); Song et al., *DDIM* ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502)); Lipman et al., *Flow Matching for Generative Modeling* ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747)); Rombach et al., *Latent Diffusion* ([arXiv:2112.10752](https://arxiv.org/abs/2112.10752)). Build: train a DDPM on MNIST from scratch, derive and implement the ELBO, then implement flow matching on the same data and compare sample quality at 10, 50 and 1000 sampling steps. Derive the connection between the score function and the noise prediction target.

**Week 8 — Evaluation.** Underrated, and a common gap on CVs. Liang et al., *HELM* ([arXiv:2211.09110](https://arxiv.org/abs/2211.09110)); Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)); Chiang et al., *Chatbot Arena* ([arXiv:2403.04132](https://arxiv.org/abs/2403.04132)); Schaeffer et al., *Are Emergent Abilities of LLMs a Mirage?* ([arXiv:2304.15004](https://arxiv.org/abs/2304.15004)). Build: implement Bradley-Terry maximum likelihood ranking from pairwise judgements yourself, with bootstrap confidence intervals, and use it to rank four small models judged by a larger one. Then quantify judge bias — position bias, length bias — by permuting the presentation order.

**Week 9 — Interpretability.** Elhage et al., *A Mathematical Framework for Transformer Circuits* and *In-context Learning and Induction Heads* (transformer-circuits.pub); Elhage et al., *Toy Models of Superposition*; Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600)). Build: train a sparse autoencoder on the residual stream of a small model using `nnsight` or `TransformerLens`, and find and document ten interpretable features.

**Week 10 — Inference and Serving.** Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)); Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)). Build: implement speculative decoding from scratch with a 0.5B draft model and a 3B target, verify output distributions are identical, measure the speedup and the acceptance rate against draft model quality.

---

## Staying current after week 10

- **Sebastian Raschka's *Ahead of AI*** — categorised paper lists, roughly twice a year, plus deep architectural write-ups.
- **arXiv `cs.LG` and `cs.CL`** via a filtered feed, not the firehose.
- **Model technical reports** — Nemotron, Qwen, Llama, DeepSeek, OLMo. Often more informative about actual practice than academic papers, because they document what was tried and discarded.
- **AI2's OLMo** — fully open pretraining data, code and checkpoints. The only way to see a complete pipeline end to end.

## Notes on making this legible to recruiters

- Commit incrementally throughout each project rather than in a single dump. The history is evidence of process.
- Every week's README opens with the result, not the setup.
- Pin dependencies. A `requirements.txt` that installs cleanly is a signal.
- Write one short blog-style post per project, or a top-level `RESULTS.md` collecting the plots. Most people never look past the front page.
- Say clearly what you tried that failed. Nobody with hiring authority believes six consecutive successful projects.
