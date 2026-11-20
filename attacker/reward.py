"""Per-turn reward (§4).

    r_t = r_spread(t) + r_terminal(t) + r_validity(t)
          + r_fluency(t) + r_length(t) + r_diversity(t)

v1: r_uncertainty = 0, r_CF = 0 (§11).

Components are kept separate so the trainer can log reward-scale drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Set

from .potential import phi


@dataclass
class RewardConfig:
    alpha_spread: float = 0.3
    eta_early_fail: float = 0.1
    invalid_penalty: float = -0.05
    # fluency (anti-gibberish) — penalty only, so it auto-vanishes once messages
    # are clean (same design as invalid_penalty, §4.4).
    beta_fluency: float = 0.1       # scale on the fluency penalty
    fluency_threshold: float = -4.0  # per-token base log-prob floor (~ppl 55)
    beta_length: float = 0.0
    target_message_tokens: int = 32
    truncation_penalty: float = 0.0
    beta_diversity: float = 0.0
    diversity_threshold: float = 0.5
    # v1 disabled terms, kept explicit for symmetry with the spec.
    xi_uncertainty: float = 0.0
    beta_cf: float = 0.0


@dataclass
class TurnReward:
    spread: float
    terminal: float
    validity: float

    @property
    def total(self) -> float:
        return self.spread + self.terminal + self.validity


def r_spread(phi_prev: float, phi_next: float, alpha: float) -> float:
    """alpha * (Phi(s_{t+1}) - Phi(s_t))  (§4.2).

    [TRAP] Differential form, NOT alpha * Phi(s_t). This telescopes to
    alpha * (Phi_T - Phi_0) so it is potential-based shaping (policy
    invariant) and cannot be length-hacked.
    """
    return alpha * (phi_next - phi_prev)


def r_terminal(
    *,
    is_terminal: bool,
    consensus_reached: bool,
    t: int,
    T: int,
    eta: float,
) -> float:
    """Only nonzero on the final turn of the episode (§4.3).

        撑满 T 轮仍未 consensus : +1
        第 t 轮达成 consensus    : -1 - eta * (T - t) / T
    """
    if not is_terminal:
        return 0.0
    if consensus_reached:
        return -1.0 - eta * (T - t) / T
    return +1.0


def r_validity(valid: bool, invalid_penalty: float) -> float:
    """(§4.4)

    [TRAP] No positive reward for valid format. A small negative penalty for
    invalid output auto-vanishes once format errors disappear, avoiding
    reward-scale drift mid-training.
    """
    return 0.0 if valid else invalid_penalty


_LEAK_RE = re.compile(r"(?i)\b(position|message)\s*:")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{5,}")            # same char 6+ in a row
_REPEAT_TOKEN_RE = re.compile(r"(\b\w+\b)(?:\W+\1){3,}")  # a word repeated 4+ times


def fluency_penalty(message: str, msg_ref_logp: float, cfg: RewardConfig) -> float:
    """Non-positive anti-gibberish penalty. 0 = clean, more negative = worse.

    Two complementary signals (see §fluency):
      (a) perplexity hinge — penalize when the frozen base (reference) model
          finds the message very unlikely (mean per-token base log-prob below
          `fluency_threshold`). Catches gibberish / code-switching / word salad.
      (b) rule-based catches for what perplexity MISSES (repetition is
          high-probability, so ppl won't flag it): leaked "Position:"/"Message:"
          fields, long single-char runs ("000000"), repeated words, and
          symbol-soup (mostly non-language characters).

    Scaled by `beta_fluency`. Penalty-only, so it vanishes for clean messages.
    """
    msg = message or ""
    ppl_pen = min(0.0, float(msg_ref_logp) - cfg.fluency_threshold)  # <=0
    rule = 0.0
    if _LEAK_RE.search(msg):
        rule -= 1.0
    if _REPEAT_CHAR_RE.search(msg) or _REPEAT_TOKEN_RE.search(msg):
        rule -= 1.0
    if len(msg) >= 4:
        lang_frac = sum(c.isalpha() or c.isspace() for c in msg) / len(msg)
        if lang_frac < 0.5:
            rule -= 1.0
    return cfg.beta_fluency * (ppl_pen + rule)


def length_penalty(n_message_tokens: int, hit_cap: bool, cfg: RewardConfig) -> float:
    """Penalty for rambling and for exhausting the message budget."""
    target = max(1, int(cfg.target_message_tokens))
    excess = max(0, int(n_message_tokens) - target)
    penalty = -float(cfg.beta_length) * excess / target
    if hit_cap:
        penalty += min(0.0, float(cfg.truncation_penalty))
    return penalty


def _word_ngrams(message: str, n: int = 3) -> Set[str]:
    words = re.findall(r"\b\w+\b", (message or "").lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def message_similarity(a: str, b: str) -> float:
    """Word-trigram Jaccard similarity in [0, 1]."""
    aa, bb = _word_ngrams(a), _word_ngrams(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def diversity_penalty(similarities: Iterable[float], cfg: RewardConfig) -> float:
    """Penalize only similarity above a threshold, leaving incidental overlap free."""
    peak = max([0.0, *[float(x) for x in similarities]])
    threshold = min(0.999, max(0.0, float(cfg.diversity_threshold)))
    scaled = max(0.0, peak - threshold) / (1.0 - threshold)
    return -float(cfg.beta_diversity) * scaled


def compute_turn_reward(
    *,
    benign_prev,          # benign positions in s_t
    benign_next,          # benign positions in s_{t+1}
    valid: bool,
    is_terminal: bool,
    consensus_reached: bool,
    t: int,               # 1-indexed turn
    T: int,
    cfg: RewardConfig,
    phi_prev: Optional[float] = None,
    phi_next: Optional[float] = None,
) -> TurnReward:
    phi_prev = phi(benign_prev) if phi_prev is None else phi_prev
    phi_next = phi(benign_next) if phi_next is None else phi_next
    return TurnReward(
        spread=r_spread(phi_prev, phi_next, cfg.alpha_spread),
        terminal=r_terminal(
            is_terminal=is_terminal,
            consensus_reached=consensus_reached,
            t=t,
            T=T,
            eta=cfg.eta_early_fail,
        ),
        validity=r_validity(valid, cfg.invalid_penalty),
    )
