"""Preflight assertions that MUST pass before training (§10).

Run via scripts/preflight.py. These catch the failure modes that are silent —
they don't raise on their own during training, they just quietly corrupt the
gradient.
"""
from __future__ import annotations

import statistics
from typing import List

import torch

from .generation import logprobs_at, logprobs_at_batched
from .ppo import turn_log_ratio
from .potential import phi
from .rollout import Episode


# ---- A. old_logp consistency (§10.A) --------------------------------------
def check_old_logp_consistency(actor, episodes: List[Episode], n: int = 5,
                               tol: float = 1e-3) -> float:
    """Recompute log-probs with the training-engine forward and compare against
    the log-probs captured during generation. Max abs diff must be < tol.

    In a vLLM-rollout setup these differ systematically and get AMPLIFIED under
    the turn-level geometric mean — always train on the recomputed old_logp."""
    turns = [tn for ep in episodes for tn in ep.turns]
    turns = turns[:n]
    max_diff = 0.0
    for tn in turns:
        recomputed = logprobs_at(actor, tn.gen.input_ids.to(next(actor.parameters()).device),
                                 tn.gen.value_idx.to(next(actor.parameters()).device))
        diff = (recomputed.detach().cpu() - tn.gen.old_logp_gen.cpu()).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < tol, (
        f"old_logp mismatch {max_diff:.2e} >= {tol}. Rollout/training log-probs "
        f"disagree; train on recomputed old_logp and verify byte-identical context."
    )
    return max_diff


# ---- B. segment mask correctness (§10.B) ----------------------------------
def check_segment_masks(episodes: List[Episode]) -> None:
    for ep in episodes:
        for tn in ep.turns:
            m = tn.gen.seg_masks
            pos, msg = m["next_position"], m["message"]
            assert (pos & msg).sum() == 0, "segment masks overlap"
            assert tn.gen.value_idx.shape[0] >= 1, "no value tokens this turn"
            # value tokens must be fewer than all generated tokens (the forced
            # "Message: "/"\nPosition: " scaffolding is excluded)
            n_generated = tn.gen.input_ids.shape[0] - tn.gen.prompt_len
            assert tn.gen.value_idx.shape[0] < n_generated, (
                "value tokens include scaffolding"
            )
            assert int(tn.gen.value_idx.min()) >= tn.gen.prompt_len


# ---- A'. batched vs unbatched forward (training-critical) ------------------
def check_batched_forward_consistency(actor, episodes, pad_id, n: int = 8,
                                      tol: float = 1e-3) -> float:
    """logprobs_at_batched must match the single-sequence logprobs_at per token.
    old_logp / new_logp / ref_logp all flow through the batched forward, so any
    padding/gather bug here silently corrupts every PPO ratio. This is the
    guard for the batched-rollout throughput path."""
    device = next(actor.parameters()).device
    turns = [tn for ep in episodes for tn in ep.turns][:n]
    ids = [tn.gen.input_ids.to(device) for tn in turns]
    vidx = [tn.gen.value_idx.to(device) for tn in turns]
    with torch.no_grad():
        batched = logprobs_at_batched(actor, ids, vidx, pad_id)
        single = [logprobs_at(actor, i, v) for i, v in zip(ids, vidx)]
    max_diff = 0.0
    for b, s in zip(batched, single):
        max_diff = max(max_diff, float((b - s).abs().max()))
    assert max_diff < tol, (
        f"batched vs unbatched logprob mismatch {max_diff:.2e} >= {tol}. "
        f"Padding/position handling in logprobs_at_batched is wrong — fix before "
        f"training, this corrupts every ratio."
    )
    return max_diff


# ---- F. actor backward correctness ----------------------------------------
def check_actor_backward_consistency(actor, input_ids, value_idx, pad_id,
                                     repeat_rtol: float = 1e-4,
                                     finite_diff_rtol: float = 0.15) -> dict:
    """Check that the Qwen3.5->LoRA backward path is connected and plausible.

    Covers four failure classes that forward-only checks cannot detect:
      F1 trainable-gradient coverage;
      F2 finite/nonzero gradients;
      F3 deterministic repeated backward;
      F4 autograd direction vs central finite differences at several scales;
      F5 an actual small step along -gradient must reduce the diagnostic loss.

    No optimizer step is taken. The one temporarily perturbed LoRA tensor is
    restored exactly and all gradients are cleared before returning.
    """
    device = next(actor.parameters()).device
    ids = input_ids.to(device)
    vidx = value_idx.to(device)
    trainable = [(name, p) for name, p in actor.named_parameters()
                 if p.requires_grad]
    assert trainable, "actor has no trainable parameters"

    def diagnostic_loss():
        lps, ents = logprobs_at_batched(
            actor, [ids], [vidx], pad_id, return_entropy=True)
        # NLL exercises the sampled-token path; the small entropy term also
        # exercises the full-vocabulary gradient used by PPO's entropy bonus.
        return -lps[0].mean() - 1e-3 * ents[0].mean()

    def run_backward():
        actor.zero_grad(set_to_none=True)
        loss = diagnostic_loss()
        loss.backward()
        grads = {}
        missing = []
        nonfinite = []
        sq_norm = 0.0
        for name, p in trainable:
            if p.grad is None:
                missing.append(name)
                continue
            g = p.grad.detach()
            if not bool(torch.isfinite(g).all()):
                nonfinite.append(name)
            sq_norm += float(g.float().pow(2).sum())
            grads[name] = g.float().cpu().clone()
        return float(loss.detach()), grads, missing, nonfinite, sq_norm ** 0.5

    loss1, grads1, missing1, bad1, norm1 = run_backward()
    coverage = len(grads1) / len(trainable)
    assert coverage >= 0.99, (
        f"only {coverage:.1%} of trainable tensors received gradients; "
        f"examples missing={missing1[:5]}")
    assert not bad1, f"non-finite actor gradients: {bad1[:5]}"
    assert norm1 > 1e-8, f"actor gradient norm is effectively zero: {norm1:.3e}"

    loss2, grads2, missing2, bad2, norm2 = run_backward()
    coverage2 = len(grads2) / len(trainable)
    assert coverage2 >= 0.99 and not bad2, (
        f"second backward invalid: missing={missing2[:5]} nonfinite={bad2[:5]}")
    diff_sq = base_sq = 0.0
    max_abs_diff = 0.0
    for name, g1 in grads1.items():
        g2 = grads2[name]
        delta = g2 - g1
        diff_sq += float(delta.pow(2).sum())
        base_sq += float(g1.pow(2).sum())
        max_abs_diff = max(max_abs_diff, float(delta.abs().max()))
    repeat_rel = (diff_sq ** 0.5) / max(base_sq ** 0.5, 1e-12)
    assert repeat_rel <= repeat_rtol or max_abs_diff <= 1e-7, (
        f"actor backward is not repeatable: rel_diff={repeat_rel:.3e}, "
        f"max_abs_diff={max_abs_diff:.3e}")

    # Probe the two strongest LoRA tensors. A single finite-difference magnitude
    # can be misleading for hybrid recurrent kernels, but signs and the actual
    # descent test are robust. Multiple tensors tell us whether any discrepancy
    # is a harmless near-uniform scale or a directionally corrupted gradient.
    ranked = sorted(grads1, key=lambda name: float(grads1[name].pow(2).sum()),
                    reverse=True)[:2]
    trainable_by_name = dict(trainable)
    probe_originals = {
        name: trainable_by_name[name].detach().clone() for name in ranked
    }
    probes = []
    try:
        for probe_name in ranked:
            probe = trainable_by_name[probe_name]
            direction = grads1[probe_name].to(device=probe.device, dtype=probe.dtype)
            direction = direction / direction.float().norm().clamp_min(1e-12).to(
                direction.dtype)
            autodiff = float((grads1[probe_name].to(probe.device) *
                              direction.float()).sum())
            original = probe_originals[probe_name]
            base_loss = float(diagnostic_loss().detach())
            trials = []
            for epsilon in (3e-2, 1e-2, 3e-3, 1e-3):
                with torch.no_grad():
                    probe.copy_(original + epsilon * direction)
                    plus = float(diagnostic_loss())
                    probe.copy_(original - epsilon * direction)
                    minus = float(diagnostic_loss())
                    probe.copy_(original)
                numerical = (plus - minus) / (2.0 * epsilon)
                rel_error = abs(numerical - autodiff) / max(
                    abs(numerical), abs(autodiff), 1e-8)
                ratio = numerical / autodiff if abs(autodiff) > 1e-12 else float("nan")
                trials.append({"epsilon": epsilon, "plus": plus, "minus": minus,
                               "numerical": numerical, "rel_error": rel_error,
                               "numerical_over_autograd": ratio})
            best = min(trials, key=lambda item: item["rel_error"])
            same_direction = (best["numerical"] == 0.0 or autodiff == 0.0 or
                              best["numerical"] * autodiff > 0)
            # This is the optimizer-relevant test: a step opposite the computed
            # gradient must improve the same loss. Require a resolvable decrease,
            # trying all scales because too-small fp32 differences can quantize.
            descent_trials = [t for t in trials if t["minus"] < base_loss]
            assert same_direction, (
                f"actor gradient direction is reversed on {probe_name}: "
                f"autograd={autodiff:.3e}, numerical={best['numerical']:.3e}")
            assert descent_trials, (
                f"-gradient did not reduce actor diagnostic loss on {probe_name}: "
                f"base={base_loss:.8f}, trials={trials}")
            probes.append({"name": probe_name, "autodiff": autodiff,
                           "base_loss": base_loss, "best": best,
                           "trials": trials,
                           "best_descent": max(base_loss - t["minus"]
                                               for t in descent_trials)})
            with torch.no_grad():
                probe.copy_(original)
    finally:
        # Restore every probe from its own shape-matched snapshot, even if a
        # diagnostic forward/assertion fails partway through the second tensor.
        with torch.no_grad():
            for name, original_value in probe_originals.items():
                trainable_by_name[name].copy_(original_value)
        actor.zero_grad(set_to_none=True)

    primary = probes[0]
    best = primary["best"]
    magnitude_warning = any(
        p["best"]["rel_error"] > finite_diff_rtol for p in probes)
    return {
        "loss": loss1,
        "coverage": coverage,
        "grad_norm": norm1,
        "repeat_rel_diff": repeat_rel,
        "repeat_max_abs_diff": max_abs_diff,
        "probe": primary["name"],
        "autodiff": primary["autodiff"],
        "finite_diff": best["numerical"],
        "finite_diff_rel_error": best["rel_error"],
        "finite_diff_epsilon": best["epsilon"],
        "finite_diff_ratio": best["numerical_over_autograd"],
        "magnitude_warning": magnitude_warning,
        "best_descent": primary["best_descent"],
        "probes": probes,
    }


# ---- C. Phi excludes the attacker (§10.C) ---------------------------------
def check_phi_excludes_attacker(cfg) -> None:
    from .game import GameConfig, new_game
    gcfg = GameConfig(max_pos=cfg.game.max_pos, n_benign=cfg.game.n_benign,
                      max_steps=cfg.game.max_steps, consensus_tol=cfg.game.consensus_tol)
    st = new_game(gcfg, seed=0)
    bp = st.benign_positions()
    assert len(bp) == cfg.game.n_benign, "benign_positions leaks the attacker"
    before = phi(bp)
    st.positions[st.attacker_id] = 0
    after = phi(st.benign_positions())
    assert before == after, "Phi changed when only the attacker moved"


# ---- D. PBRS telescoping (§10.D) ------------------------------------------
def check_pbrs_telescoping(ep: Episode, alpha: float, tol: float = 1e-6) -> None:
    total = sum(tn.reward_spread for tn in ep.turns)
    expected = alpha * (ep.turns[-1].phi_next - ep.turns[0].phi_prev)
    assert abs(total - expected) < tol, (
        f"r_spread does not telescope: {total} vs {expected}"
    )


# ---- E. clip_eps calibration (§10.E) --------------------------------------
def calibrate_clip_eps(actor, episodes: List[Episode]) -> dict:
    """Measure sigma = std of D_t on the ACTUAL segment-weighted turn ratios
    (with eps effectively infinite, i.e. before any clipping), then suggest
    eps_low = 2*sigma, eps_high = 2.5*sigma (§10.E)."""
    device = next(actor.parameters()).device
    Ds: List[float] = []
    for ep in episodes:
        for tn in ep.turns:
            ids = tn.gen.input_ids.to(device)
            vidx = tn.gen.value_idx.to(device)
            with torch.no_grad():
                new_logp = logprobs_at(actor, ids, vidx)
            old_logp = tn.gen.old_logp_gen.to(device)
            seg = {k: v.to(device) for k, v in tn.gen.seg_masks.items()}
            D = turn_log_ratio(new_logp, old_logp, seg)
            Ds.append(float(D))
    sigma = statistics.pstdev(Ds) if len(Ds) > 1 else 0.0
    return {
        "sigma_D": sigma,
        "eps_low": 2.0 * sigma,
        "eps_high": 2.5 * sigma,
        "n_turns": len(Ds),
        "note": "target turn_clip_frac 5-15%; <2% eps too big, >25% lower LR / minibatch",
    }
