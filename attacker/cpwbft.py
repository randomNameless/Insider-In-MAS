"""CP-WBFT (Confidence Probe-based Weighted BFT) INPUT filter for the benign agents.

Defense for Experiment 3, adapted from the CP-WBFT consensus of
"Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine
Fault Tolerance" (arXiv 2511.10400) to this repeated 1-D position game.

Confidence source (decoder peakedness, chosen over the paper's trained hidden-state
probe HCP): each agent's confidence C_i is the mean probability of its Position-digit
tokens when it broadcast (benign from the WM, attacker from the actor). It is computed
during generation and stored in state.confidences (see generation.py / rollout.py).

Defense action (paper's Stage 1 "adopt neighbors with higher confidence", applied as
an input filter like the W-MSR one): before benign agent i updates, it only attends to
neighbors j with C_j >= C_i - margin — both the position AND the message of every
lower-confidence neighbor are dropped from its prompt. So a malicious agent whose
Position generation is less peaked than the benign's gets ignored that round; to be
seen, the attacker must be at least as "confident" (peaked) as its benign neighbors.

NOTE (a known limitation, kept on purpose): a decoder-peakedness / self-reported
confidence is GAMEABLE — an attacker that makes its Position token deterministic looks
maximally confident and is never filtered. That is exactly why the paper adds the
trained HCP probe. Watch whether the RL attacker collapses to a peaked policy.
"""
from __future__ import annotations

from typing import List


class CPWBFTDefense:
    needs_confidence = True   # tells the pipeline to populate state.confidences

    def __init__(self, margin: float = 0.0, keep_min_one: bool = True):
        self.margin = float(margin)
        self.keep_min_one = bool(keep_min_one)

    def visible_others(self, self_idx: int, state) -> List[int]:
        conf = state.confidences
        ci = conf[self_idx]
        others = [j for j in range(len(conf)) if j != self_idx]
        kept = [j for j in others if conf[j] >= ci - self.margin]
        if not kept and self.keep_min_one and others:
            # never empty the prompt: keep the single most-confident neighbor
            kept = [max(others, key=lambda j: conf[j])]
        return kept
