#!/usr/bin/env python
"""Qwen3 PPO v3: larger rollout, higher actor LR, restored Position entropy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attacker.trainer import Trainer  # noqa: E402
from exp1_nodefense import _banner, _check_invariants, run_preflight  # noqa: E402
from exp1_nodefense_qwen3_attack_v2 import load_variant  # noqa: E402


DEFAULT_VARIANT = HERE / "exp1_nodefense_qwen3_signal_v3.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 1 Qwen3 signal-strengthened PPO v3")
    ap.add_argument("--config", default=str(DEFAULT_VARIANT))
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    cfg = load_variant(args.config)
    _check_invariants(cfg)
    print("Variant: Qwen3-4B PPO signal v3")
    print(f"  actor_lr={cfg.lr.actor} rollout_N={cfg.ppo.batch_episodes_N} "
          f"beta_kl_ref={cfg.trust_region.beta_kl_ref} "
          f"ent_pos={cfg.trust_region.ent_coef_position} "
          f"ent_msg={cfg.trust_region.ent_coef_message}")
    _banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
        return
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()

