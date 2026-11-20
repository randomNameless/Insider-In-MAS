"""Lockstep batched debate rollout producing consensus-compatible Episode / EpisodeTurn objects.

Round r = 0, 1, 2 (round-synchronous):
  1. attacker (HF actor) and the three honest agents (vLLM) all read the snapshot of round r-1;
  2. honest beliefs P_i(.) are read out after their own argument -> Phi_r = -mean_i P_i(gold);
  3. all four replies are committed together.
Final submission r = 3: honest agents only, reading round 3 -> Phi_3 and the final answers.

Attacker turn t (0..2) earns  alpha * (Phi_{t+1} - Phi_t)  (round t+1 honest beliefs are the first
ones that read the turn-t message) + validity; turn 2 also earns the terminal outcome reward.
Phi_0 does not depend on the attacker. The gold answer is used here only for Phi / terminal /
metrics, never in any prompt.

Targeted persuader (2026-10-19): when a spec carries `target` (an assigned WRONG option), the
attacker's system prompt gets the same TARGET_HINT as the eval `targeted_*` conditions, its answer
letter can be prefilled with the target (cfg.targeted.force_letter), Phi becomes +mean_i P_i(target)
and the terminal reward is paid for an honest majority ON the target (dreward.r_terminal_targeted).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from attacker.reward import r_validity
from attacker.rollout import Episode, EpisodeTurn

from . import dprotocol as D
from .dreward import DebateRewardConfig, r_belief, r_terminal, r_terminal_targeted
from .engines import attacker_reply, generate_attacker_debate, honest_round


@dataclass
class EpisodeSpec:
    task: dict
    seat: int
    ep_seed: int
    target: Optional[str] = None      # assigned wrong option (targeted persuader), else None


@dataclass
class DebateEpisode(Episode):
    task_id: str = ""
    gold: str = ""
    seat: int = 0
    ep_seed: int = 0
    phis: List[float] = field(default_factory=list)                 # Phi_0..Phi_3
    honest_answers: List[List[str]] = field(default_factory=list)   # [round][3]  rounds 0..3
    honest_pgold: List[List[float]] = field(default_factory=list)   # [round][3]
    attacker_answers: List[Optional[str]] = field(default_factory=list)
    attacker_valid: List[bool] = field(default_factory=list)
    outcome: str = ""
    target: Optional[str] = None
    honest_ptarget: List[List[float]] = field(default_factory=list)  # [round][3]
    transcript: list = field(default_factory=list)


def _critic_values(critic, critic_tok, texts, chunk: int = 8) -> List[float]:
    cdev = next(critic.parameters()).device
    vals = []
    for i in range(0, len(texts), chunk):
        enc = critic_tok(texts[i:i + chunk], return_tensors="pt", padding=True)
        with torch.no_grad():
            vals += critic(enc.input_ids.to(cdev), enc.attention_mask.to(cdev)).float().tolist()
    return vals


def play_debates(*, actor, actor_tok, critic, critic_tok, honest_engine, cfg,
                 rcfg, drcfg: DebateRewardConfig, specs: List[EpisodeSpec], sample: bool = True):
    hcfg = cfg.world_model
    states = [D.new_state(s.task, s.seat) for s in specs]
    hseats = [D.honest_seats(s.seat) for s in specs]
    golds = [s.task["answer"] for s in specs]
    eps = [DebateEpisode(task_id=s.task["task_id"], gold=g, seat=s.seat, ep_seed=s.ep_seed, target=s.target)
           for s, g in zip(specs, golds)]
    for s, g in zip(specs, golds):
        assert s.target is None or (s.target != g and s.target in D.LETTERS[:len(s.task["metadata"]["choices"])]), \
            (s.task["task_id"], s.target, g)
    tcfg = getattr(cfg, "targeted", None)
    force = bool(tcfg is not None and getattr(tcfg, "force_letter", False))
    if force:
        assert all(s.target for s in specs), "force_letter needs a target for every debate"
    turn_buf = [[] for _ in specs]   # per episode: (critic_text, gt, value)

    for r in range(D.N_ROUNDS + 1):
        # ---- attacker (discussion rounds only) ----
        gts = None
        if r < D.N_ROUNDS:
            chats = [D.chat(st, s.seat) for st, s in zip(states, specs)]
            chats = [D.add_target_hint(c, s.target) if s.target else c for c, s in zip(chats, specs)]
            ctexts = [D.critic_text(st, s.seat) + (f"\n[Assigned option: {s.target}]" if s.target else "")
                      for st, s in zip(states, specs)]
            vals = _critic_values(critic, critic_tok, ctexts)
            gts = generate_attacker_debate(
                actor, actor_tok, chats, [st.n_options for st in states],
                max_arg_tokens=cfg.actor.max_new_tokens, sample=sample,
                temperature=float(getattr(cfg.actor, "temperature", 1.0)),
                gen_batch=int(getattr(cfg.actor, "gen_batch", 16)),
                answer_first=bool(getattr(cfg.actor, "answer_first", False)),
                forced_letters=[s.target for s in specs] if force else None)
            for k in range(len(specs)):
                turn_buf[k].append((ctexts[k], gts[k], vals[k]))
        # ---- honest agents, same snapshot ----
        items = [(st, a, s.ep_seed) for st, s, hs in zip(states, specs, hseats) for a in hs]
        hres = honest_round(honest_engine, items, temperature=float(hcfg.temperature),
                            max_tokens=int(hcfg.max_new_tokens))
        for k, (st, ep) in enumerate(zip(states, eps)):
            hr = hres[3 * k: 3 * k + 3]
            probs = [h["probs"] for h in hr]
            ep.phis.append(D.phi_target(probs, ep.target) if ep.target else D.phi(probs, ep.gold))
            ep.honest_answers.append([h["reply"].answer for h in hr])
            gi = D.LETTERS.index(ep.gold)
            ep.honest_pgold.append([p[gi] for p in probs])
            if ep.target:
                ep.honest_ptarget.append([p[D.LETTERS.index(ep.target)] for p in probs])
            rnd = {"round": r + 1, "honest": {a: {"answer": h["reply"].answer,
                                                  "argument": h["reply"].argument,
                                                  "pgold": h["probs"][gi]}
                                              for a, h in zip(hseats[k], hr)}}
            if r < D.N_ROUNDS:
                gt = gts[k]
                rep = attacker_reply(gt)
                ep.attacker_answers.append(rep.answer)
                ep.attacker_valid.append(gt.valid)
                rnd["attacker"] = {"seat": specs[k].seat, "answer": rep.answer,
                                   "argument": gt.message, "valid": gt.valid}
                replies = {a: h["reply"] for a, h in zip(hseats[k], hr)}
                replies[specs[k].seat] = rep
                D.commit(st, replies)            # atomic: all four become visible together
            ep.transcript.append(rnd)

    # ---- rewards ----
    for k, ep in enumerate(eps):
        ep.outcome = D.outcome(ep.honest_answers[-1], ep.gold)
        T = D.N_ROUNDS
        for t, (ctext, gt, v) in enumerate(turn_buf[k]):
            last = t == T - 1
            ep.turns.append(EpisodeTurn(
                t=t, prompt_text=ctext, gen=gt, value=v,
                phi_prev=ep.phis[t], phi_next=ep.phis[t + 1], valid=gt.valid,
                reward_spread=r_belief(ep.phis[t], ep.phis[t + 1], cfg.reward.alpha_spread),
                reward_terminal=(r_terminal_targeted(ep.honest_answers[-1], ep.gold, ep.target, drcfg)
                                 if ep.target else r_terminal(ep.outcome, drcfg)) if last else 0.0,
                reward_validity=r_validity(gt.valid, rcfg.invalid_penalty)))
        ep.length = len(ep.turns)
        ep.phi_0, ep.phi_T = ep.phis[0], ep.phis[-1]
        ep.consensus_reached = ep.outcome == "correct"    # "defence succeeded" in base logger
    return eps


def debate_metrics(eps: List[DebateEpisode]) -> dict:
    n = len(eps)
    fin_acc = [sum(a == ep.gold for a in ep.honest_answers[-1]) / 3 for ep in eps]
    m = {
        "honest_final_acc": sum(fin_acc) / n,
        "majority_wrong": sum(sum(a != ep.gold for a in ep.honest_answers[-1]) >= 2 for ep in eps) / n,
        "outcome_wrong": sum(ep.outcome == "wrong" for ep in eps) / n,
        "outcome_deadlock": sum(ep.outcome == "deadlock" for ep in eps) / n,
        "outcome_correct": sum(ep.outcome == "correct" for ep in eps) / n,
        "all_wrong": sum(all(a != ep.gold for a in ep.honest_answers[-1]) for ep in eps) / n,
        "attacker_valid": sum(sum(ep.attacker_valid) / len(ep.attacker_valid) for ep in eps) / n,
        "attacker_says_gold": sum(sum(a == ep.gold for a in ep.attacker_answers) / len(ep.attacker_answers)
                                  for ep in eps) / n,
        "change_rate": sum(sum(ep.honest_answers[r][i] != ep.honest_answers[r - 1][i]
                               for r in range(1, len(ep.honest_answers)) for i in range(3))
                           / (3 * (len(ep.honest_answers) - 1)) for ep in eps) / n,
    }
    teps = [ep for ep in eps if ep.target]
    if teps:
        nt = len(teps)
        m["majority_on_target"] = sum(D.majority(ep.honest_answers[-1]) == ep.target for ep in teps) / nt
        m["honest_on_target"] = sum(sum(a == ep.target for a in ep.honest_answers[-1]) / 3 for ep in teps) / nt
        m["attacker_says_target"] = sum(sum(a == ep.target for a in ep.attacker_answers) / len(ep.attacker_answers)
                                        for ep in teps) / nt
        for r in range(D.N_ROUNDS + 1):
            m[f"ptarget_r{r + 1}"] = sum(sum(ep.honest_ptarget[r]) / 3 for ep in teps) / nt
    for r in range(D.N_ROUNDS + 1):
        m[f"pgold_r{r + 1}"] = sum(sum(ep.honest_pgold[r]) / 3 for ep in eps) / n
        m[f"acc_r{r + 1}"] = sum(sum(a == ep.gold for a in ep.honest_answers[r]) / 3 for ep in eps) / n
    return m
