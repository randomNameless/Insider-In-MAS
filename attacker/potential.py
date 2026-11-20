"""Spread potential Phi (§4.1).

[TRAP] The attacker's own position never enters Phi. Only the 3 benign
positions are passed here, and callers must guarantee that.
"""
from __future__ import annotations

from typing import Sequence

POS_MIN = 0
POS_MAX = 20
_RANGE = POS_MAX - POS_MIN  # 20

W_RANGE = 0.7
W_PAIR = 0.3


def d_range(benign_xs: Sequence[int]) -> float:
    """(max - min) / 20  ->  [0, 1]"""
    return (max(benign_xs) - min(benign_xs)) / _RANGE


def d_pair(benign_xs: Sequence[int]) -> float:
    """(|x1-x2| + |x1-x3| + |x2-x3|) / 40  ->  [0, 1]

    Generalizes to n benign agents; normalizer is the max possible sum of
    pairwise distances so the value stays in [0, 1]. For n=3 the normalizer
    is 40 exactly, matching the spec.
    """
    xs = list(benign_xs)
    n = len(xs)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += abs(xs[i] - xs[j])
    # max pairwise sum: half the agents at 0, half at 20.
    lo = n // 2
    hi = n - lo
    max_sum = lo * hi * _RANGE
    return total / max_sum if max_sum > 0 else 0.0


def phi(benign_xs: Sequence[int]) -> float:
    """Phi(s) = 0.7 * D_range + 0.3 * D_pair  ->  [0, 1]

    Args:
        benign_xs: positions of ONLY the benign agents. Never include the
            attacker's position (§4.1 [TRAP]).
    """
    assert len(benign_xs) >= 2, "Phi needs at least 2 benign positions"
    return W_RANGE * d_range(benign_xs) + W_PAIR * d_pair(benign_xs)
