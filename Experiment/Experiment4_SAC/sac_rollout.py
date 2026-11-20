"""Compatibility wrapper for the shared round-synchronous rollout."""
from __future__ import annotations

from attacker.rollout import rollout_batch


def rollout_batch_sac(*, actor, actor_tok, critic, critic_tok, benign_engine,
                      cfg, rcfg, seeds, sample=True):
    return rollout_batch(
        actor=actor, actor_tok=actor_tok, critic=critic,
        critic_tok=critic_tok, benign_engine=benign_engine,
        cfg=cfg, rcfg=rcfg, seeds=seeds, sample=sample)
