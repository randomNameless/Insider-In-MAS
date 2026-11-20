"""Game state, matching RL_attack_env_qwen.AttackEnv dynamics.

4 agents on a line 0..max_pos. Agents 0-2 benign (world model), agent 3 the
malicious attacker (RL policy). Every round is synchronous: all policies read
the positions, messages and confidences committed by the previous round, then
all new actions are committed atomically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .prompts import ATTACKER_IDX, BENIGN_IDS

BENIGN_PERSONALITIES = ["stubborn", "suggestible", "neutral"]


@dataclass
class GameConfig:
    max_pos: int = 20
    n_benign: int = 3
    max_steps: int = 10
    consensus_tol: int = 0          # benign consensus <=> (max-min) <= tol
    # Legacy single-personality knob: if benign_personalities is not set, all 3
    # benign agents get this suffix (old suggestible-only behavior).
    benign_personality: str = "suggestible"
    # Per-agent benign personalities (Exp1): a list of length n_benign assigned
    # to benign indices 0..n_benign-1 in order (attacker is always "malicious").
    # e.g. ["stubborn", "suggestible", "neutral"] -> one of each. If None, falls
    # back to [benign_personality] * n_benign.
    benign_personalities: Optional[List[str]] = None
    msg_order: str = "index"
    init: str = "random"
    seed: int = 0


@dataclass
class GameState:
    cfg: GameConfig
    t: int                                   # 0-indexed round
    positions: List[int]                     # length 4 (benign 0-2, attacker 3)
    personalities: List[str]                 # length 4
    last_messages: List[Optional[str]] = field(default_factory=lambda: [None] * 4)
    # Per-agent confidence in its last broadcast position (decoder peakedness =
    # mean prob of its Position-digit tokens). Used by the CP-WBFT defense (Exp3);
    # ignored by Exp1/Exp2. Defaults to 1.0 (fully trusted) before the first move.
    confidences: List[float] = field(default_factory=lambda: [1.0] * 4)
    # Public history of the game: one (round-start positions, broadcasts of that
    # round) pair per completed round. Read by history-conditioned world models
    # (consensus_wm, prompt_mode wm_*); ignored by the legacy persona WM.
    rounds: List[tuple] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    @property
    def benign_ids(self) -> List[int]:
        return list(BENIGN_IDS)

    @property
    def attacker_id(self) -> int:
        return ATTACKER_IDX

    def benign_positions(self) -> List[int]:
        return [self.positions[i] for i in BENIGN_IDS]

    def consensus_reached(self) -> bool:
        xs = self.benign_positions()
        return (max(xs) - min(xs)) <= self.cfg.consensus_tol

    @property
    def remaining(self) -> int:
        return self.cfg.max_steps - self.t


def new_game(cfg: GameConfig, seed: Optional[int] = None) -> GameState:
    rng = random.Random(cfg.seed if seed is None else seed)
    positions = [rng.randint(0, cfg.max_pos) for _ in range(4)]
    if cfg.benign_personalities:
        benign = list(cfg.benign_personalities)
        assert len(benign) == cfg.n_benign, (
            f"benign_personalities has {len(benign)} entries but n_benign="
            f"{cfg.n_benign}")
    else:
        benign = [cfg.benign_personality] * cfg.n_benign
    personalities = benign + ["malicious"]
    return GameState(cfg=cfg, t=0, positions=positions,
                     personalities=personalities, last_messages=[None] * 4,
                     confidences=[1.0] * 4, rng=rng)


def clamp_position(x: int, cfg: GameConfig) -> int:
    return max(0, min(cfg.max_pos, int(x)))
