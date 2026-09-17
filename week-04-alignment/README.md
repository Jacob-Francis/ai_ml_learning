# Week 4 — Alignment and RL Post-Training

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
