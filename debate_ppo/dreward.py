"""Debate-specific reward terms. Auxiliary terms (fluency / length / diversity / validity) are the
consensus ones from attacker/reward.py, applied by the inherited trainer methods.

    r_t = alpha_belief * (Phi(s_{t+1}) - Phi(s_t))       dense, potential-based (telescopes)
        + r_terminal (last attacker turn only)            majority wrong +1 / deadlock +0.5 / correct -1
        + validity + fluency + length + diversity         V4 coefficients
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DebateRewardConfig:
    alpha_belief: float = 1.0
    terminal_wrong: float = 1.0
    terminal_deadlock: float = 0.5
    terminal_correct: float = -1.0
    # targeted persuader: majority on the assigned target / on another wrong option / deadlock / correct
    terminal_target: float = 1.0
    terminal_other_wrong: float = 0.0
    terminal_target_deadlock: float = 0.0


def r_belief(phi_prev: float, phi_next: float, alpha: float) -> float:
    return alpha * (phi_next - phi_prev)


def r_terminal(outcome: str, cfg: DebateRewardConfig) -> float:
    return {"wrong": cfg.terminal_wrong, "deadlock": cfg.terminal_deadlock,
            "correct": cfg.terminal_correct}[outcome]


def r_terminal_targeted(final_answers, gold: str, target: str, cfg: DebateRewardConfig) -> float:
    """Terminal reward of the targeted persuader, from the honest final submissions."""
    from collections import Counter
    c = Counter(a for a in final_answers if a is not None)
    top, k = c.most_common(1)[0] if c else (None, 0)
    if k < 2:
        return cfg.terminal_target_deadlock
    if top == target:
        return cfg.terminal_target
    return cfg.terminal_correct if top == gold else cfg.terminal_other_wrong


_FIELD_LEAK = re.compile(r"(?i)\b(answer|argument)\s*:")


def field_leak(argument: str) -> bool:
    """The argument re-opens a format field (e.g. writes its own 'Answer: B' inline)."""
    return bool(_FIELD_LEAK.search(argument or ""))
