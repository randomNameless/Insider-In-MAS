#!/usr/bin/env python
"""Experiment 1 (Main) — NO-DEFENSE consensus attack, MIXED benign personalities.

Turn-PPO RL attacker (merged malicious Qwen3.5-4B + fresh RL LoRA) vs 3 benign world-model
agents with a MIXED personality combo (1 stubborn + 1 suggestible + 1 neutral),
in the plain consensus game with NO defense (benign read raw numeric positions).

    actor  = /path/to/models/malicious-merged (merged SFT base + fresh RL LoRA)
    critic = Qwen/Qwen3-4B                       (full-finetune, bf16)
    WM     = /path/to/qwen35_wm_pm_backup/opd-mm  (frozen; one model, the 3
             personalities are realized by the system-prompt suffix per agent)

Usage (run on Lambda after transferring code + WM; see RUN_INSTRUCT2507.md):

    # 1) preflight (REQUIRED) — prints a calibrated clip_eps; set ppo.clip_eps to it
    python Experiment/Experiment1_Main/exp1_nodefense.py --preflight

    # 2) train
    python Experiment/Experiment1_Main/exp1_nodefense.py

The config lives next to this file (exp1_nodefense.yaml). This script only wires
up sys.path, asserts the Exp1 invariants, and launches the verified Trainer — all
training logic stays in the reviewed `attacker/` package (unchanged).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]                      # .../consensus-attacker_Previous
sys.path.insert(0, str(REPO_ROOT))

from attacker.config import load_config          # noqa: E402
from attacker.trainer import Trainer             # noqa: E402

DEFAULT_CONFIG = HERE / "exp1_nodefense.yaml"


def _check_invariants(cfg) -> None:
    """Fail fast if the config drifted from the Experiment 1 spec."""
    combo = sorted(getattr(cfg.game, "benign_personalities", []) or [])
    assert combo == ["neutral", "stubborn", "suggestible"], (
        f"Exp1 requires a mixed combo (1 stubborn + 1 suggestible + 1 neutral); "
        f"got benign_personalities={combo}")
    assert cfg.game.n_benign == 3, "Exp1 uses n_benign=3"
    assert getattr(cfg.actor, "sft_adapter", None) is None, (
        "Exp1 uses a merged SFT model directly; do not pass --sft-adapter")
    assert float(cfg.ppo.w_next_position) == float(cfg.ppo.w_message), (
        "Exp1 requires equal Position and Message segment weights")
    assert int(cfg.ppo.logprob_microbatch_turns) == 1, (
        "Qwen3.5 Exp1 requires single-sequence actor logprob forwards")
    assert int(cfg.ppo.update_batch_turns) > int(cfg.ppo.logprob_microbatch_turns), (
        "optimizer/KL batch must be larger than the actor forward microbatch")
    assert int(cfg.eval.real_eval_episodes) >= 64, (
        "Exp1 checkpoint selection needs at least 64 fixed validation episodes")
    assert int(cfg.eval.validation_batch_episodes) <= int(cfg.eval.real_eval_episodes), (
        "validation batch cannot exceed the full validation pool")
    assert int(cfg.train.checkpoint_every) > 0, (
        "Exp1 must retain periodic actor checkpoints for post-training sweep")


def _banner(cfg) -> None:
    print("=" * 68)
    print("Experiment 1 (Main) — NO DEFENSE, MIXED benign personalities")
    print("-" * 68)
    print(f"  actor  : {cfg.actor.model_name}")
    print(f"  critic : {cfg.critic.model_name}  (fp32={getattr(cfg.critic,'fp32',True)})")
    print(f"  WM     : {cfg.world_model.model_name}")
    print(f"  benign : {list(cfg.game.benign_personalities)}  (indices 0,1,2)")
    print(f"  game   : max_pos={cfg.game.max_pos} steps={cfg.game.max_steps} "
          f"tol={cfg.game.consensus_tol}")
    print("  timing : synchronous snapshot -> generate all -> atomic commit")
    print(f"  PPO    : actor_microbatch={cfg.ppo.logprob_microbatch_turns} "
          f"update_batch={cfg.ppo.update_batch_turns}")
    print(f"  eval   : every={cfg.eval.real_eval_every} n={cfg.eval.real_eval_episodes} "
          f"batch={cfg.eval.validation_batch_episodes} fixed_seed={cfg.eval.validation_seed}")
    print(f"  saves  : every={cfg.train.checkpoint_every} + best/last/interrupted")
    print(f"  out    : {cfg.train.out_dir}")
    print("=" * 68)


def run_preflight(cfg, episodes: int) -> None:
    """Same checks as scripts/preflight.py, on the Exp1 config."""
    from attacker.asserts import (
        calibrate_clip_eps, check_batched_forward_consistency,
        check_actor_backward_consistency, check_old_logp_consistency, check_pbrs_telescoping,
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
    generation_logp_consistent = True
    try:
        d = check_old_logp_consistency(tr.actor, eps, n=5)
        print(f"    OK  max_diff={d:.2e}")
    except AssertionError as e:
        generation_logp_consistent = False
        print(f"    WARN {e}\n    (trainer trains on RECOMPUTED old_logp -> ratios start at 1.0)")
    print("[A'] batched vs unbatched forward ...")
    # Test the batch size the update actually uses. Qwen3.5 GDN on the torch
    # fallback is batch-composition-sensitive at B>1, so this Exp1 uses B=1
    # for actor old/new/ref logprob forwards.
    logprob_mb = int(getattr(cfg.ppo, "logprob_microbatch_turns",
                            getattr(cfg.ppo, "minibatch_turns", 1)))
    b = check_batched_forward_consistency(
        tr.actor, eps, tr.actor_pad, n=logprob_mb, tol=0.05)
    print(f"    OK  max_diff={b:.2e}  "
          f"(training logprob batch={logprob_mb}; tol=0.05)")
    print("[F] actor backward kernel ...")
    # Forward-only preflight misses CUDA/Triton backward incompatibilities and
    # silent wrong gradients. Test checkpointing OFF and ON separately.
    tn = eps[0].turns[0]
    tr.opt_actor.zero_grad()
    f_results = {}
    try:
        for label, checkpointing in (("ckpt_off", False), ("ckpt_on", True)):
            tr._set_checkpointing(checkpointing)
            f = check_actor_backward_consistency(
                tr.actor, tn.gen.input_ids, tn.gen.value_idx, tr.actor_pad)
            f_results[label] = f
            status = "WARN(scale)" if f["magnitude_warning"] else "OK"
            print(f"    {label}: {status} coverage={f['coverage']:.1%} "
                  f"grad_norm={f['grad_norm']:.3e} "
                  f"repeat_rel={f['repeat_rel_diff']:.3e}")
            for probe in f["probes"]:
                b = probe["best"]
                print(f"        {probe['name']}: auto={probe['autodiff']:.3e} "
                      f"fd={b['numerical']:.3e} ratio={b['numerical_over_autograd']:.3f} "
                      f"rel={b['rel_error']:.3f} eps={b['epsilon']:.0e} "
                      f"descent={probe['best_descent']:.3e}")
        off, on = f_results["ckpt_off"], f_results["ckpt_on"]
        assert off["probe"] == on["probe"], (
            f"checkpointing changed strongest gradient tensor: "
            f"off={off['probe']} on={on['probe']}")
        ckpt_rel = abs(off["autodiff"] - on["autodiff"]) / max(
            abs(off["autodiff"]), abs(on["autodiff"]), 1e-8)
        assert ckpt_rel <= 1e-3, (
            f"gradient checkpointing changes actor gradient: rel={ckpt_rel:.3e}")
        print(f"    checkpointing gradient agreement: OK rel={ckpt_rel:.3e}")
        if off["magnitude_warning"] or on["magnitude_warning"]:
            print("    WARN: gradient direction/descent are valid, but finite-difference "
                  "magnitude differs; inspect ratios above before training.")
    finally:
        tr.opt_actor.zero_grad()
        tr._set_checkpointing(False)
    print("[E] clip_eps calibration ...")
    if generation_logp_consistent:
        cal = calibrate_clip_eps(tr.actor, eps)
        print(f"    sigma(D_t)={cal['sigma_D']:.4f}  n_turns={cal['n_turns']}")
        print(f"    suggested clip_eps in [{cal['eps_low']:.4f}, {cal['eps_high']:.4f}]")
        print(f"    -> set ppo.clip_eps in exp1_nodefense.yaml; {cal['note']}")
    else:
        print(f"    SKIP: generation logprobs differ from recomputed old logprobs; "
              f"that would make calibration invalid. Keep configured "
              f"clip_eps={cfg.ppo.clip_eps} and tune from turn_clip_frac.")
    print("\nAll preflight checks passed.")


def main():
    ap = argparse.ArgumentParser(description="Experiment 1 (no defense, mixed personalities)")
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
