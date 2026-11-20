"""Experiment-3-local CP-WBFT confidence input filter.

This is the same project-level CP-WBFT adaptation used by the existing PPO
generation pipeline: confidence is decoder peakedness of Position tokens, and
benign i keeps neighbor j iff C_j >= C_i - margin.  It is deliberately local so
Experiment 3 does not mutate the shared training implementation.
"""
from __future__ import annotations

from typing import List, Sequence


def retained_confident_indices(
    confidences: Sequence[float], self_idx: int, margin: float = 0.0,
    keep_min_one: bool = True,
) -> List[int]:
    if not 0 <= self_idx < len(confidences):
        raise IndexError(f"self_idx={self_idx} outside {len(confidences)} agents")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    values = [float(value) for value in confidences]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"confidences must lie in [0,1], got {values}")
    others = [idx for idx in range(len(values)) if idx != self_idx]
    threshold = values[self_idx] - float(margin)
    # Preserve the mathematical >= boundary despite binary floating-point
    # roundoff (for example 0.8 - 0.2 may be 0.6000000000000001).
    kept = [idx for idx in others if values[idx] + 1e-12 >= threshold]
    if not kept and keep_min_one and others:
        # Stable tie break: highest confidence, then lowest agent id.
        kept = [max(others, key=lambda idx: (values[idx], -idx))]
    return kept


class Experiment3CPWBFTDefense:
    needs_confidence = True

    def __init__(self, margin: float = 0.0, keep_min_one: bool = True):
        self.margin = float(margin)
        self.keep_min_one = bool(keep_min_one)
        if self.margin < 0:
            raise ValueError("margin must be non-negative")

    def visible_others(self, self_idx: int, state) -> List[int]:
        return retained_confident_indices(
            state.confidences, self_idx, self.margin, self.keep_min_one)


def self_check() -> None:
    conf = [0.8, 0.6, 0.9, 0.4]
    assert retained_confident_indices(conf, 0) == [2]
    assert retained_confident_indices(conf, 1) == [0, 2]
    assert retained_confident_indices(conf, 2) == [0]
    assert retained_confident_indices(conf, 3) == [0, 1, 2]
    assert retained_confident_indices([0.9, 0.1, 0.2, 0.3], 0) == [3]
    assert retained_confident_indices([0.8, 0.6, 0.9, 0.4], 0, 0.2) == [1, 2]
