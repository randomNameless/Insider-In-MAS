"""WM-vs-prompt-judge evaluation (2026-10-27): is the distilled belief world model (debate-wm-v1) a
better judge of the honest Qwen3.5-9B agent's next belief than a zero-shot prompted judge?

Judge task (from debate_ppo/drollout.py + engines.honest_round): given the EXACT chat the honest
agent receives at a round (persona system prompt + round prompt with the previous round's messages,
including the attacker's), produce its option-letter distribution P(A..J) for this round. The PPO
reward is alpha * (Phi_{t+1} - Phi_t) with Phi = -mean_i P_i(gold) (untargeted) or +mean_i P_i(target)
(targeted), so the judge is called once per honest turn, three honest turns per attacker turn.

Modules: build_eval_set (held-out turns with 9B ground truth), judges (prompts + runners),
run_judge (CLI), trivial (no-model baselines), metrics (tables).
"""
