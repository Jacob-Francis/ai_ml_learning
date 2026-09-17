# Week 6 — Agents, Tool Use, and Test-Time Compute

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
