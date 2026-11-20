#!/usr/bin/env python
"""Experiment 3 — CP-WBFT DEFENSE, MIXED benign personalities.

Same Turn-PPO attacker as Experiment 1 (Qwen3-4B-Instruct-2507, fresh LoRA, no SFT;
same critic / PPO / reward), trained against the SAME opd-mm benign world model — but
the benign (victim) agents now run a **CP-WBFT confidence input filter**:

  - each agent's confidence = decoder peakedness of its Position-digit tokens (benign
    from the WM, attacker from the actor), computed for free during generation;
  - before a benign updates, it only attends to neighbors with confidence >= its own
    (minus margin); every lower-confidence neighbor's position AND message are dropped.

Adapted from CP-WBFT (arXiv 2511.10400): the paper's trained hidden-state probe (HCP)
is skipped and its prompt probe (PCP) is replaced by decoder peakedness; the "adopt
higher-confidence neighbors" rule (Stage 1) is applied as an input filter.

    defense = CP-WBFT confidence filter (attacker/cpwbft.py), margin=0
    actor   = Qwen/Qwen3-4B-Instruct-2507   (LoRA, direct RL, no SFT — same as Exp1)
    critic  = Qwen/Qwen3-4B                  (full-finetune, bf16)
    WM      = /path/to/models/qwen35-wm  (frozen benign, mixed personalities)

Usage (on Lambda):
    python Experiment/Experiment1_Main/exp3_CP-WBFT.py --preflight --episodes 4
    python Experiment/Experiment1_Main/exp3_CP-WBFT.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attacker.config import load_config          # noqa: E402
from attacker.trainer import Trainer             # noqa: E402

DEFAULT_CONFIG = HERE / "exp3_cpwbft.yaml"


def _check_invariants(cfg) -> None:
    combo = sorted(getattr(cfg.game, "benign_personalities", []) or [])
    assert combo == ["neutral", "stubborn", "suggestible"], (
        f"Exp3 requires a mixed combo (1 stubborn + 1 suggestible + 1 neutral); "
        f"got benign_personalities={combo}")
    assert cfg.game.n_benign == 3, "Exp3 uses n_benign=3"
    assert getattr(cfg.actor, "sft_adapter", None) is None, (
        "Exp3 = direct RL on stock Instruct-2507 (no --sft-adapter), same init as Exp1")
    dcfg = getattr(cfg, "defense", None)
    assert dcfg is not None and getattr(dcfg, "type", None) == "cpwbft", (
        "Exp3 requires defense.type: cpwbft in the config")


def _banner(cfg) -> None:
    margin = getattr(cfg.defense, "margin", 0.0)
    print("=" * 68)
    print("Experiment 3 — CP-WBFT DEFENSE, MIXED benign personalities")
    print("-" * 68)
    print(f"  actor   : {cfg.actor.model_name}")
    print(f"  critic  : {cfg.critic.model_name}  (fp32={getattr(cfg.critic,'fp32',True)})")
    print(f"  WM      : {cfg.world_model.model_name}")
    print(f"  benign  : {list(cfg.game.benign_personalities)}  (indices 0,1,2)")
    print(f"  defense : CP-WBFT confidence filter (decoder peakedness), margin={margin}")
    print(f"  game    : max_pos={cfg.game.max_pos} steps={cfg.game.max_steps} "
          f"tol={cfg.game.consensus_tol}")
    print(f"  out     : {cfg.train.out_dir}")
    print("=" * 68)


def run_preflight(cfg, episodes: int) -> None:
    from attacker.asserts import (
        calibrate_clip_eps, check_batched_forward_consistency,
        check_old_logp_consistency, check_pbrs_telescoping,
        check_phi_excludes_attacker, check_segment_masks,
    )
    print("Loading models (this can take a while)...")
    tr = Trainer(cfg)
    saved_N = cfg.ppo.batch_episodes_N
    cfg.ppo.batch_episodes_N = episodes
    eps = tr.sample_batch(base_seed=42, sample=True)
    tr._recompute_old_and_ref(eps)
    cfg.ppo.batch_episodes_N = saved_N

    print("\n[C] Phi excludes attacker ...");     check_phi_excludes_attacker(cfg); print("    OK")
    print("[B] segment masks ...");               check_segment_masks(eps);        print("    OK")
    print("[D] PBRS telescoping ...")
    for ep in eps:
        check_pbrs_telescoping(ep, cfg.reward.alpha_spread)
    print("    OK")
    print("[A] old_logp consistency ...")
    try:
        d = check_old_logp_consistency(tr.actor, eps, n=5)
        print(f"    OK  max_diff={d:.2e}")
    except AssertionError as e:
        print(f"    WARN {e}\n    (trainer trains on RECOMPUTED old_logp -> ratios start at 1.0)")
    print("[A'] batched vs unbatched forward ...")
    b = check_batched_forward_consistency(tr.actor, eps, tr.actor_pad, n=8, tol=0.05)
    print(f"    OK  max_diff={b:.2e}  (fp32 real-sequence precision; tol=0.05)")
    print("[E] clip_eps calibration ...")
    cal = calibrate_clip_eps(tr.actor, eps)
    print(f"    sigma(D_t)={cal['sigma_D']:.4f}  n_turns={cal['n_turns']}")
    print(f"    suggested clip_eps in [{cal['eps_low']:.4f}, {cal['eps_high']:.4f}]")
    print(f"    -> set ppo.clip_eps in exp3_cpwbft.yaml; {cal['note']}")
    print("\nAll preflight checks passed.")


def main():
    ap = argparse.ArgumentParser(description="Experiment 3 (CP-WBFT defense, mixed personalities)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--preflight", action="store_true",
                    help="run preflight assertions + clip_eps calibration, then exit")
    ap.add_argument("--episodes", type=int, default=4, help="preflight rollout episodes")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _check_invariants(cfg)
    _banner(cfg)

    if args.preflight:
        run_preflight(cfg, args.episodes)
        return
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
