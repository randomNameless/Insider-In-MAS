"""Turn-level GAE (§5).

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lam * A_{t+1}     # backward, A_{T+1}=0

Actor advantage uses lam = 0.90.
Critic regression target uses lam = 1.0 (decoupled GAE):

    R_hat_t = A_t^{lam=1.0} + V(s_t)

terminal turn: V(s_{T+1}) = 0.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    lam: float,
    bootstrap_value: float = 0.0,
) -> List[float]:
    """One episode. `values` = [V(s_1), ..., V(s_T)].

    bootstrap_value = V(s_{T+1}); 0.0 for a terminated episode (§5).
    """
    T = len(rewards)
    assert len(values) == T, "values must align with rewards (per-turn)"
    advantages = [0.0] * T
    next_value = bootstrap_value
    next_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        advantages[t] = next_adv
        next_value = values[t]
    return advantages


def gae_and_returns(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    lam_actor: float,
    lam_critic: float = 1.0,
    bootstrap_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Returns (actor_advantages, critic_targets) for one episode.

    critic_targets R_hat_t = A_t^{lam_critic} + V(s_t)  (decoupled, §5).
    """
    adv_actor = compute_gae(rewards, values, gamma, lam_actor, bootstrap_value)
    adv_critic = compute_gae(rewards, values, gamma, lam_critic, bootstrap_value)
    returns = [a + v for a, v in zip(adv_critic, values)]
    return adv_actor, returns
