#!/usr/bin/env python
"""Experiment 9 entry point.

    python -m debate_ppo.run --config debate_ppo/configs/debate_v1.yaml --preflight --episodes 8
    python -m debate_ppo.run --config debate_ppo/configs/debate_v1.yaml     # train (run in tmux)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.config import load_config  # noqa: E402


def preflight(cfg, n_eps: int) -> None:
    import torch
    from attacker.asserts import (check_actor_backward_consistency, check_batched_forward_consistency,
                                  check_old_logp_consistency, check_pbrs_telescoping,
                                  check_segment_masks)
    from attacker.generation import logprobs_at_batched
    from attacker.ppo import turn_log_ratio
    from debate_ppo.dtrainer import DebateTrainer
    from debate_ppo.drollout import debate_metrics

    t0 = time.time()
    tr = DebateTrainer(cfg)
    print(f"[preflight] models loaded in {time.time() - t0:.0f}s; "
          f"GPU mem allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB (torch only)")
    saved = cfg.ppo.batch_episodes_N
    cfg.ppo.batch_episodes_N = n_eps
    t0 = time.time()
    eps = tr.sample_batch(base_seed=42, sample=True)
    cfg.ppo.batch_episodes_N = saved
    print(f"[preflight] {n_eps} debates rolled out + recomputed in {time.time() - t0:.0f}s "
          f"({tr._last_timing})")
    for ep in eps[:2]:
        print(json.dumps({"task": ep.task_id, "gold": ep.gold, "seat": ep.seat, "outcome": ep.outcome,
                          "phis": [round(x, 3) for x in ep.phis],
                          "transcript": ep.transcript}, indent=1)[:6000])
    print("[preflight] metrics:", json.dumps(debate_metrics(eps)))
    for ep in eps:
        for tn in ep.turns:
            print(f"   t={tn.t} valid={tn.valid} tokens={tn.message_tokens} cap={tn.message_hit_cap} "
                  f"r_belief={tn.reward_spread:+.3f} r_term={tn.reward_terminal:+.2f} "
                  f"r_flu={tn.reward_fluency:+.4f} r_len={tn.reward_length:+.4f} "
                  f"r_div={tn.reward_diversity:+.4f} adv={tn.advantage:+.3f}")

    print("[B] segment masks ...");  check_segment_masks(eps); print("    OK")
    print("[D] PBRS telescoping ...")
    for ep in eps:
        check_pbrs_telescoping(ep, cfg.reward.alpha_spread)
        assert len(ep.phis) == 4 and ep.length == 3
    print("    OK")
    print("[A] old_logp consistency (generation vs recompute) ...")
    try:
        d = check_old_logp_consistency(tr.actor, eps, n=6)
        print(f"    OK max_diff={d:.2e}")
    except AssertionError as e:
        print(f"    WARN {e}")
    mb = int(cfg.ppo.logprob_microbatch_turns)
    print(f"[A'] batched vs unbatched forward (training microbatch={mb}, and 4 for reference) ...")
    b = check_batched_forward_consistency(tr.actor, eps, tr.actor_pad, n=max(mb, 1), tol=1e-3)
    print(f"    OK max_diff={b:.2e}")
    try:
        b4 = check_batched_forward_consistency(tr.actor, eps, tr.actor_pad, n=4, tol=1e-3)
        print(f"    (B=4 max_diff={b4:.2e})")
    except AssertionError as e:
        print(f"    (B=4 would fail: {e})")
    print("[F] actor backward ...")
    tn = eps[0].turns[0]
    res = {}
    try:
        for label, ck in (("ckpt_off", False), ("ckpt_on", True)):
            tr._set_checkpointing(ck)
            f = check_actor_backward_consistency(tr.actor, tn.gen.input_ids, tn.gen.value_idx, tr.actor_pad)
            res[label] = f
            print(f"    {label}: coverage={f['coverage']:.1%} grad_norm={f['grad_norm']:.3e} "
                  f"fd_ratio={f['finite_diff_ratio']:.3f} descent={f['best_descent']:.2e}")
        rel = abs(res["ckpt_off"]["autodiff"] - res["ckpt_on"]["autodiff"]) / max(
            abs(res["ckpt_off"]["autodiff"]), 1e-8)
        assert rel <= 1e-3, f"checkpointing changes the gradient: rel={rel:.2e}"
        print(f"    OK ckpt agreement rel={rel:.2e}")
    finally:
        tr._set_checkpointing(False)
        tr.opt_actor.zero_grad()

    print("[E] clip_eps calibration: one real PPO update on this batch with clipping disabled, "
          "then sigma of the per-turn segment-weighted log-ratio D_t ...")
    weights = {"next_position": float(cfg.ppo.w_next_position), "message": float(cfg.ppo.w_message)}
    saved_clip = cfg.ppo.clip_eps
    cfg.ppo.clip_eps = 1e6
    tr.sample_batch = lambda base_seed, sample=True: eps
    cfg.ppo.batch_episodes_N = n_eps
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr.train_iteration(0)
    free, total = torch.cuda.mem_get_info()
    print(f"    update took {time.time() - t0:.0f}s; torch peak allocated "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB; "
          f"GPU free now {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    cfg.ppo.batch_episodes_N = saved
    cfg.ppo.clip_eps = saved_clip
    turns = [tn for ep in eps for tn in ep.turns]
    Ds, Dm, Da = [], [], []
    with torch.no_grad():
        for tn in turns:
            new = logprobs_at_batched(tr.actor, [tn.gen.input_ids], [tn.gen.value_idx], tr.actor_pad)[0].cpu()
            seg = tn.gen.seg_masks
            Ds.append(float(turn_log_ratio(new, tn.old_logp, seg, weights)))
            d = new - tn.old_logp
            if seg["message"].any():
                Dm.append(float(d[seg["message"]].mean()))
            if seg["next_position"].any():
                Da.append(float(d[seg["next_position"]].mean()))
    sd = statistics.pstdev(Ds)
    print(f"    after 1 iteration ({len(turns)} turns, {len(turns) // int(cfg.ppo.update_batch_turns)} "
          f"optimizer steps, lr={cfg.lr.actor}): sigma(D_t)={sd:.4f} max|D_t|={max(map(abs, Ds)):.4f}")
    print(f"    argument-segment sigma={statistics.pstdev(Dm):.4f}  answer-segment "
          f"sigma={statistics.pstdev(Da) if len(Da) > 1 else 0:.4f}")
    print(f"    suggested clip_eps in [{2 * sd:.4f}, {2.5 * sd:.4f}]  (configured {saved_clip})")
    if tr.monitor_on:
        print("[M] 9B monitor smoke test (on-demand engine while actor/critic/Adam are resident) ...")
        saved_n = cfg.monitor.n_questions
        cfg.monitor.n_questions = 4
        wm_eps = tr._eval_pool(tr._val_specs(4), tr.honest)
        mon = tr.monitor(wm_eps)
        cfg.monitor.n_questions = saved_n
        free, total = torch.cuda.mem_get_info()
        print(f"    OK {json.dumps({k: mon[k] for k in ('n', 'time_s')})}; GPU free after {free / 2**30:.1f} GiB")
        eps2 = tr._eval_pool(tr._val_specs(4), tr.honest)
        print(f"    WM engine usable after wake_up: {len(eps2)} debates")
    print("\nAll preflight checks passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--resume-adapter", default=None,
                    help="actor-only checkpoint dir (checkpoint-XXXX) to resume from")
    ap.add_argument("--start-it", type=int, default=0, help="iteration index to resume at")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.iterations is not None:
        cfg.train.iterations = a.iterations
    if a.out_dir:
        cfg.train.out_dir = a.out_dir
    if a.preflight:
        preflight(cfg, a.episodes)
        return
    from debate_ppo.dtrainer import DebateTrainer
    if a.resume_adapter:
        setattr(cfg.actor, "sft_adapter", a.resume_adapter)   # PeftModel.from_pretrained(..., is_trainable=True)
    DebateTrainer(cfg).fit(start_it=a.start_it)


if __name__ == "__main__":
    main()
