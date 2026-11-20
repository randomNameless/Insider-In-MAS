"""Pure bookkeeping helpers for Experiment 5 agent-count scaling."""
from __future__ import annotations

import random
from collections import Counter


DEFAULT_BENIGN_COUNTS = (4, 5, 6, 7, 8)
DEFAULT_HORIZONS = (10, 15, 20)
PERSONALITIES = ("stubborn", "suggestible", "neutral")
ORIGINAL_PERSONALITIES = ("stubborn", "suggestible", "neutral")


def parse_condition(value: str) -> tuple[int, int]:
    try:
        benign_text, malicious_text = value.lower().split("x", 1)
        benign, malicious = int(benign_text), int(malicious_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid condition {value!r}; expected BENIGNxMALICIOUS") from exc
    if benign < 1 or malicious < 1:
        raise ValueError("benign and malicious counts must both be positive")
    return benign, malicious


def parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(item.strip()) for item in value.split(",")
                             if item.strip()}))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive comma-separated integers")
    return horizons


def paired_benign_personalities(seed: int, episode: int, n_benign: int,
                                max_benign: int = 8) -> list[str]:
    """Random, prefix-paired personalities with no type appearing four times.

    A full ``max_benign`` assignment is sampled first and smaller conditions use
    its prefix. Thus the same indexed benign agent keeps its personality across
    B=4..8 for a given seed/episode. Rejection sampling is deterministic and is
    uniform over valid full assignments.
    """
    n_benign, max_benign = int(n_benign), int(max_benign)
    if n_benign < 1 or n_benign > max_benign:
        raise ValueError("n_benign must be in [1, max_benign]")
    if max_benign > 3 * len(PERSONALITIES):
        raise ValueError("max_benign cannot satisfy the at-most-three constraint")
    rng = random.Random(int(seed) + int(episode) + 50_000_000)
    while True:
        assignment = [rng.choice(PERSONALITIES) for _ in range(max_benign)]
        if max(Counter(assignment).values()) <= 3:
            return assignment[:n_benign]


def paired_cooperative_added_personalities(seed: int, episode: int,
                                           n_benign: int,
                                           max_benign: int = 8) -> list[str]:
    """Original trio plus prefix-paired added neutral/suggestible agents."""
    n_benign, max_benign = int(n_benign), int(max_benign)
    if n_benign < 3 or n_benign > max_benign:
        raise ValueError("n_benign must be in [3, max_benign]")
    n_added = max_benign - 3
    rng = random.Random(int(seed) + int(episode) + 60_000_000)
    n_neutral = n_added // 2 + (rng.randint(0, 1) if n_added % 2 else 0)
    added = (["neutral"] * n_neutral
             + ["suggestible"] * (n_added - n_neutral))
    rng.shuffle(added)
    return list(ORIGINAL_PERSONALITIES) + added[:n_benign - 3]


def consensus_reached(positions, n_benign: int, tolerance: float = 0.0) -> bool:
    benign = [int(value) for value in positions[:n_benign]]
    if len(benign) != n_benign:
        raise ValueError("not enough positions for the benign consensus endpoint")
    return max(benign) - min(benign) <= float(tolerance)


def paired_initial_positions(seed: int, episode: int, max_steps: int,
                             max_pos: int, n_total: int) -> list[int]:
    """Prefix-paired states: smaller conditions share their first positions."""
    rng = random.Random(int(seed) + int(episode) + int(max_steps) * 1000)
    return [rng.randint(0, int(max_pos)) for _ in range(int(n_total))]
