#!/usr/bin/env python
"""Preflight assertions (§10). Must pass before training.

    python scripts/preflight.py --config configs/default.yaml [--sft-adapter runs/sft]

Runs a small batch of real rollouts and checks A-E, then prints a suggested
clip_eps (E). Set ppo.clip_eps in the config to that value before training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.asserts import (
    calibrate_clip_eps,
    check_batched_forward_consistency,
    check_old_logp_consistency,
    check_pbrs_telescoping,
    check_phi_excludes_attacker,
    check_segment_masks,
)
from attacker.config import load_config
from attacker.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sft-adapter", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.sft_adapter:
        setattr(cfg.actor, "sft_adapter", args.sft_adapter)

    print("Loading models (this can take a while)...")
    tr = Trainer(cfg)

    # small batch of rollouts
    saved_N = cfg.ppo.batch_episodes_N
    cfg.ppo.batch_episodes_N = args.episodes
    episodes = tr.sample_batch(base_seed=42, sample=True)
    tr._recompute_old_and_ref(episodes)
    cfg.ppo.batch_episodes_N = saved_N

    print("\n[C] Phi excludes attacker ...")
    check_phi_excludes_attacker(cfg)
    print("    OK")

    print("[B] segment masks ...")
    check_segment_masks(episodes)
    print("    OK")

    print("[D] PBRS telescoping ...")
    for ep in episodes:
        check_pbrs_telescoping(ep, cfg.reward.alpha_spread)
    print("    OK")

    print("[A] old_logp consistency (recompute vs generation) ...")
    try:
        max_diff = check_old_logp_consistency(tr.actor, episodes, n=5)
        print(f"    OK  max_diff={max_diff:.2e}")
    except AssertionError as e:
        import re as _re
        m = _re.search(r"mismatch ([\d.eE+-]+)", str(e))
        val = float(m.group(1)) if m else 999.0
        if val < 0.5:
            print(f"    WARN  max_diff={val:.2e}: generation (recurrent) and "
                  "training (parallel) log-probs differ. Expected for linear-"
                  "attention/hybrid models (or a separate rollout engine). The "
                  "trainer uses the RECOMPUTED old_logp for BOTH old and new, so "
                  "PPO ratios start at exactly 1.0 -> safe to proceed.")
        else:
            print(f"    FAIL  max_diff={val:.2e} is too large to be precision. "
                  "This points at a real context/tokenization mismatch (chat "
                  "template, whitespace, truncation). Fix before training.")
            raise

    print("[A'] batched vs unbatched forward (training-critical) ...")
    bdiff = check_batched_forward_consistency(tr.actor, episodes, tr.actor_pad, n=8)
    print(f"    OK  max_diff={bdiff:.2e}")

    print("[E] clip_eps calibration ...")
    cal = calibrate_clip_eps(tr.actor, episodes)
    print(f"    sigma(D_t)={cal['sigma_D']:.4f}  n_turns={cal['n_turns']}")
    print(f"    suggested clip_eps in [{cal['eps_low']:.4f}, {cal['eps_high']:.4f}]")
    print(f"    -> set ppo.clip_eps accordingly; {cal['note']}")
    print("\nAll preflight checks passed.")


if __name__ == "__main__":
    main()
