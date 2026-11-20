"""Paper-faithful global MSR input filter for Experiment 2.

For the complete four-agent graph and f=1, each honest agent receives the three
other agents' proposals, sorts them by scalar position, and removes exactly one
smallest and one largest entry. The retained position-message pairs are exposed
to the benign WM. Ties are resolved by agent index only to make multiset slicing
deterministic; equal scalar values remain mathematically interchangeable.
"""
from __future__ import annotations

from typing import List, Sequence


def retained_neighbor_indices(
    positions: Sequence[int], self_idx: int, f: int = 1
) -> List[int]:
    if f < 0:
        raise ValueError(f"f must be non-negative, got {f}")
    others = [j for j in range(len(positions)) if j != self_idx]
    if len(others) < 2 * f + 1:
        raise ValueError(
            f"global MSR requires at least 2f+1 received values: "
            f"received={len(others)} f={f}"
        )
    ordered = sorted(others, key=lambda j: (int(positions[j]), j))
    retained = ordered[f:len(ordered) - f] if f else ordered
    # Prompt order stays by agent index, matching Experiment 1; filtering is
    # determined by scalar order, presentation order is not.
    return sorted(retained)


class PaperGlobalMSRDefense:
    """Defense interface consumed by the unchanged shared BenignEngine."""

    def __init__(self, f: int = 1):
        self.f = int(f)
        if self.f < 0:
            raise ValueError("f must be non-negative")

    def visible_others(self, self_idx: int, state) -> List[int]:
        return retained_neighbor_indices(state.positions, self_idx, self.f)


def self_check() -> None:
    # Agent 0 receives positions {1, 10, 20}: retain the median (agent 2).
    assert retained_neighbor_indices([7, 1, 10, 20], 0, 1) == [2]
    # The rule is global, not relative to self: even self=0 still trims both
    # ends of the received multiset and retains 10.
    assert retained_neighbor_indices([0, 1, 10, 20], 0, 1) == [2]
    # Deterministic multiset tie handling: sort by (value, agent id), retain the
    # middle occurrence. This still removes f entries at each sorted end.
    assert retained_neighbor_indices([9, 5, 5, 5], 0, 1) == [2]
    assert retained_neighbor_indices([9, 5, 5, 5], 0, 0) == [1, 2, 3]
    try:
        retained_neighbor_indices([0, 1], 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("2f+1 guard did not fire")

