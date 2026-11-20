#!/usr/bin/env python
"""Experiment 6: Exp1 PPO transfer across communication topologies."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP1_EVAL = HERE.parent / "Experiment1_Eval"
EXP5 = HERE.parent / "Experiment5_AgentScaling"
sys.path[:0] = [str(REPO), str(EXP1_EVAL), str(EXP5)]

from RL_attack_env_qwen import (  # noqa: E402
    QwenWorldModelVLLM, parse_pos_msg, ROUND0_PROMPT_TEMPLATE,
    MULTIROUND_PROMPT_TEMPLATE, build_system_prompt,
    build_attacker_system_prompt,
)
from eval_attacks_consensus import HFQwen3, _render_wm_prompt  # noqa: E402
from scaling_logic import (  # noqa: E402
    paired_initial_positions, consensus_reached,
    paired_cooperative_added_personalities,
)
from topology_logic import (  # noqa: E402
    TOPOLOGIES, adjacency, build_topology, graph_metrics,
)


def local_chat(agent_idx, personality, positions, last_messages, neighbors, t):
    """Construct the Exp1 chat using only direct-neighbor snapshot state."""
    visible_positions = [int(positions[idx]) for idx in neighbors]
    if int(t) == 0:
        user = ROUND0_PROMPT_TEMPLATE.format(
            self_pos=int(positions[agent_idx]),
            other_positions=visible_positions,
        )
    else:
        lines = "\n".join(
            f"- Agent {idx}: {last_messages[idx] or ''}" for idx in neighbors)
        user = MULTIROUND_PROMPT_TEMPLATE.format(
            self_pos=int(positions[agent_idx]),
            other_positions=visible_positions,
            message_lines=lines,
        )
    system = (build_attacker_system_prompt() if personality == "malicious"
              else build_system_prompt(personality))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run_batch(*, wm, actor, args, topology, edges, episode_indices, mode="ppo"):
    n_benign, attacker, n_total = 5, 5, 6
    if mode not in ("no_attack", "ppo"):
        raise ValueError(f"unknown intervention mode {mode}")
    adj = adjacency(n_total, edges)
    states = []
    for episode in episode_indices:
        initial = paired_initial_positions(
            args.seed, episode, args.max_steps, args.max_pos, n_total)
        personalities = paired_cooperative_added_personalities(
            args.seed, episode, n_benign) + ["neutral" if mode == "no_attack" else "malicious"]
        assert "malicious" not in personalities[:n_benign]
        assert personalities.count("malicious") == int(mode == "ppo")
        states.append({
            "episode": episode,
            "initial_positions": initial,
            "positions": list(initial),
            "last_messages": [None] * n_total,
            "personalities": personalities,
            "t": 0, "done": False, "consensus": False,
            "rounds": [] if episode < args.dump_n else None,
        })

    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        benign_chats = []
        for state in active:
            for idx in range(n_total if mode == "no_attack" else n_benign):
                benign_chats.append(local_chat(
                    idx, state["personalities"][idx], state["positions"],
                    state["last_messages"], adj[idx], state["t"]))
        benign_raw = wm._generate(
            [_render_wm_prompt(wm, chat) for chat in benign_chats])

        # Match Exp5's model-call order exactly. Neither generation commits
        # outputs: the actor still reads the same untouched round snapshot.
        attacker_raw = []
        if mode == "ppo":
            attacker_chats = [local_chat(
                attacker, "malicious", state["positions"], state["last_messages"],
                adj[attacker], state["t"]) for state in active]
            attacker_raw = actor.generate_two_phase_batch(
                attacker_chats, use_adapter=True, temperature=1.0,
                max_msg_tokens=args.actor_max_message_tokens)

        benign_offset = 0
        for batch_idx, state in enumerate(active):
            snapshot_positions = list(state["positions"])
            snapshot_messages = list(state["last_messages"])
            if mode == "ppo":
                message, position = attacker_raw[batch_idx]
                if position is None:
                    position = snapshot_positions[attacker]
                intervention_output = (
                    max(0, min(args.max_pos, int(position))), message)

            benign_outputs = []
            wm_count = n_total if mode == "no_attack" else n_benign
            for idx in range(wm_count):
                raw = benign_raw[benign_offset + idx]
                pos, msg = parse_pos_msg(raw, snapshot_positions[idx], args.max_pos)
                benign_outputs.append((int(pos), msg))
            benign_offset += wm_count
            if mode == "no_attack":
                intervention_output = benign_outputs.pop()

            # Atomic commit: every output above used the same round snapshot.
            state["positions"] = ([pos for pos, _ in benign_outputs]
                                  + [intervention_output[0]])
            state["last_messages"] = ([msg for _, msg in benign_outputs]
                                      + [intervention_output[1]])
            state["t"] += 1
            state["consensus"] = consensus_reached(
                state["positions"], n_benign, args.consensus_tol)
            state["done"] = state["consensus"] or state["t"] >= args.max_steps

            if state["rounds"] is not None:
                state["rounds"].append({
                    "t": state["t"] - 1,
                    "snapshot_positions": snapshot_positions,
                    "snapshot_messages": snapshot_messages,
                    "visible_neighbors": {str(i): adj[i] for i in range(n_total)},
                    "received_others_per_agent": [
                        {"agent": i, "self_position": snapshot_positions[i],
                         "received_others": [
                             {"agent": j, "position": snapshot_positions[j],
                              "message": snapshot_messages[j]} for j in adj[i]]}
                        for i in range(n_total)],
                    "benign_outputs": [
                        {"agent": idx,
                         "personality": state["personalities"][idx],
                         "position": pos, "message": msg}
                        for idx, (pos, msg) in enumerate(benign_outputs)],
                    "intervention_output": {
                        "agent": attacker, "role": state["personalities"][attacker],
                        "position": intervention_output[0],
                        "message": intervention_output[1]},
                    "positions_after": list(state["positions"]),
                    "consensus_all_benign_after": bool(state["consensus"]),
                })
    return states


def main():
    p = argparse.ArgumentParser(description="Exp6 topology evaluation")
    p.add_argument("--wm_model_dir", default="~/models/qwen3.5-9b")
    p.add_argument("--llm_model_dir", required=True)
    p.add_argument("--rl_adapter", required=True)
    p.add_argument("--topologies", default="complete")
    p.add_argument("--modes", default="no_attack,ppo")
    p.add_argument("--horizons", default="10,15,20,30,40,50")
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--topology_seed", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--consensus_tol", type=float, default=0.0)
    p.add_argument("--wm_temperature", type=float, default=0.7)
    p.add_argument("--wm_max_new_tokens", type=int, default=200)
    p.add_argument("--wm_gpu_mem_util", type=float, default=0.60)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--actor_max_message_tokens", type=int, default=48)
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--dump_json", required=True)
    p.add_argument("--dump_n", type=int, default=200)
    args = p.parse_args()
    args.horizons = sorted(set(int(x) for x in args.horizons.split(",")))
    if not args.horizons or min(args.horizons) < 1:
        p.error("horizons must be positive")
    args.max_steps = max(args.horizons)
    modes = args.modes.split(",")
    if not modes or set(modes) - {"no_attack", "ppo"}:
        p.error("modes must be no_attack and/or ppo")
    if args.n_exp < 1 or args.eval_batch_size < 1:
        p.error("n_exp and eval_batch_size must be positive")
    topology_seed = args.seed if args.topology_seed is None else args.topology_seed
    names = [x.strip() for x in args.topologies.split(",") if x.strip()]
    unknown = sorted(set(names) - set(TOPOLOGIES))
    if unknown:
        p.error(f"unknown topologies: {unknown}")
    if not names or len(set(names)) != len(names) or len(set(modes)) != len(modes):
        p.error("topologies/modes must be nonempty and unique")
    print("[configuration] " + json.dumps(vars(args), ensure_ascii=False), flush=True)

    wm = QwenWorldModelVLLM(
        model_dir=os.path.expanduser(args.wm_model_dir), max_pos=args.max_pos,
        max_new_tokens=args.wm_max_new_tokens,
        gpu_memory_utilization=args.wm_gpu_mem_util,
        max_model_len=args.max_model_len, enforce_eager=args.enforce_eager,
        chat_template_kwargs={"enable_thinking": False},
        temperature=args.wm_temperature, seed=args.seed)
    actor = HFQwen3(
        args.llm_model_dir, adapter_dir=os.path.expanduser(args.rl_adapter),
        device="cuda", max_new_tokens=args.actor_max_message_tokens,
        temperature=args.llm_temperature, seed=args.seed) if "ppo" in modes else None

    summaries, records, graphs = {}, [], {}
    for name, mode in ((name, mode) for name in names for mode in modes):
        label = f"{name}/{mode}"
        edges = build_topology(name, topology_seed)
        metrics = graph_metrics(6, edges, 5)
        graphs[name] = {"edges": [list(edge) for edge in edges], **metrics}

        policy_seed = args.seed + 100_000
        random.seed(policy_seed)
        np.random.seed(policy_seed % (2 ** 32))
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        if hasattr(wm, "sampling"):
            wm.sampling.seed = args.seed + 200_000

        finished = []
        for start in range(0, args.n_exp, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, args.n_exp)
            finished.extend(run_batch(
                wm=wm, actor=actor, args=args, topology=name, edges=edges,
                episode_indices=range(start, stop), mode=mode))
            if args.log_every > 0 and (stop == args.n_exp or stop % args.log_every == 0):
                print(f"[{label}] completed {stop}/{args.n_exp}", flush=True)
        count = sum(int(state["consensus"]) for state in finished)
        summaries[label] = {
            "topology": name, "mode": mode, "max_steps": args.max_steps,
            "episodes": args.n_exp, "consensus": count,
            "rate": count / args.n_exp,
            "mean_ep_len": sum(state["t"] for state in finished) / args.n_exp,
            "by_horizon": {
                str(h): {
                    "consensus": sum(int(s["consensus"] and s["t"] <= h) for s in finished),
                    "rate": sum(int(s["consensus"] and s["t"] <= h) for s in finished) / args.n_exp,
                    "mean_ep_len": sum(min(s["t"], h) for s in finished) / args.n_exp,
                } for h in args.horizons},
            **metrics,
        }
        for state in finished:
            records.append({
                "topology": name, "mode": mode, "episode": state["episode"],
                "initial_positions": state["initial_positions"],
                "benign_personalities": state["personalities"][:5],
                "personalities": state["personalities"],
                "consensus": bool(state["consensus"]),
                "ep_len": state["t"], "rounds": state["rounds"],
            })
        row = summaries[label]
        print(f"condition={label:34s} episodes={args.n_exp:4d} "
              f"consensus={count:4d} rate={row['rate']:.3f} "
              f"mean_ep_len={row['mean_ep_len']:.2f}", flush=True)

    output = Path(args.dump_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": 6,
        "design_version": "paired_topology_6nodes_exp5_aligned_v6",
        "config": vars(args),
        "horizons": args.horizons, "max_steps": args.max_steps,
        "attacker": "Experiment1_no_defense_PPO_V4",
        "defense": "none", "n_scored_benign": 5, "n_total": 6,
        "intervention_node": 5, "modes": modes,
        "malicious_count_by_mode": {mode: int(mode == "ppo") for mode in modes},
        "round_timing": "synchronous_snapshot_atomic_commit",
        "visibility": "self_plus_direct_neighbors_only",
        "message_visibility": "direct_neighbors_previous_round_messages",
        "position_visibility": "direct_neighbors_round_start_positions",
        "consensus_scope": "all_5_target_benign_agents",
        "seed": args.seed, "topology_seed": topology_seed,
        "n_exp_per_topology": args.n_exp,
        "personality_sampling": (
            "agents_0_2_fixed_stubborn_suggestible_neutral;"
            "added_agents_random_balanced_neutral_or_suggestible"),
        "generation_order": "wm_then_actor_then_atomic_commit",
        "initial_state_seed_horizon": args.max_steps,
        "graphs": graphs, "summary": summaries, "episodes": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dump] wrote {len(records)} trajectories -> {output}", flush=True)


if __name__ == "__main__":
    main()
