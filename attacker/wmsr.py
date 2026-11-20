"""W-MSR (Weighted Mean-Subsequence-Reduced) INPUT filter for the benign agents.

Defense for Experiment 2. Before each benign (victim) agent updates, the F most
extreme neighbor POSITIONS relative to that agent's own position are removed from
what it sees — both the trimmed neighbors' positions AND their messages. The frozen
world model then generates the benign agent's next position conditioned on the
RETAINED (non-extreme) neighbors only.

This is the classic MSR trim used as an input filter (not as the position-update
rule): the victim is still the WM, but a malicious agent whose broadcast position is
an outlier gets ignored by that victim this round, so the attacker must keep its
position inside the retained band to influence the victim.

Standard MSR trim for agent i with value x_i:
  - discard up to F neighbor values strictly GREATER than x_i (the largest ones),
  - discard up to F neighbor values strictly LESS than x_i (the smallest ones),
  - values equal to x_i are always kept.
With F=1 and 3 neighbors (2 benign + attacker) at least one neighbor always remains.
"""
from __future__ import annotations

from typing import List


class WMSRDefense:
    def __init__(self, F: int = 1):
        self.F = int(F)

    def visible_others(self, self_idx: int, state) -> List[int]:
        """Indices of the OTHER agents this benign agent still sees after the trim.
        `state` is a GameState (uses state.positions)."""
        pos = state.positions
        x = pos[self_idx]
        others = [j for j in range(len(pos)) if j != self_idx]
        above = sorted((j for j in others if pos[j] > x), key=lambda j: pos[j], reverse=True)
        below = sorted((j for j in others if pos[j] < x), key=lambda j: pos[j])
        trimmed = set(above[:self.F]) | set(below[:self.F])
        return [j for j in others if j not in trimmed]
