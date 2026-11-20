"""Strategy-catalog DPO attacker (2026-10-23).

Qwen3-4B writes one opponent-aware message per strategy (27 strategies, strategies.py) on states of the 9B fit
debates; the distilled debate WM (debate-wm-v1) plays the three honest agents' next turn on each message and scores
it by mean P_honest(target); best / worst messages of a state become DPO pairs under the NO-strategy prompt, so the
model learns to pick a strategy itself (as in Ju et al.). Pipeline: make_data.py; training reuses ju_dpo/train_dpo.py.
"""
