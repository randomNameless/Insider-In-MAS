#!/usr/bin/env python
"""Experiment 3 no-attacker control: three CP-WBFT agents + neutral agent 3.

All four agents use the selected victim model and update synchronously. Agents
0--2 apply the Experiment-3 confidence filter; neutral agent 3 is undefended and
sees all three neighbors. Consensus is scored on agents 0--2, matching the main
attacker experiments. Confidence is Position-digit decoder peakedness from the
preceding synchronous round (all confidences start at 1.0).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = HERE.parent / "Experiment1_Eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from RL_attack_env_qwen import (  # noqa: E402
    build_system_prompt, ROUND0_PROMPT_TEMPLATE, MULTIROUND_PROMPT_TEMPLATE,
)
from attacker.generation import generate_benign_batched  # noqa: E402
from cpwbft_defense import retained_confident_indices, self_check  # noqa: E402
from eval_exp3_cpwbft import load_causal_model  # noqa: E402


PERSONALITIES = ["stubborn", "suggestible", "neutral", "neutral"]
DEFENDED_IDS = (0, 1, 2)


def consensus_reached(positions, tolerance: float = 0.0) -> bool:
    scored = list(positions[:3])
    if len(scored) != 3:
        raise ValueError("consensus requires agents 0, 1 and 2")
    return max(scored) - min(scored) <= tolerance


def visible_others(agent_idx: int, confidences, margin: float):
    if agent_idx in DEFENDED_IDS:
        return retained_confident_indices(
            confidences, agent_idx, margin=margin, keep_min_one=True)
    return [idx for idx in range(4) if idx != agent_idx]


def build_agent_chat(agent_idx, positions, messages, confidences, t, margin):
    visible = visible_others(agent_idx, confidences, margin)
    other_positions = [int(positions[idx]) for idx in visible]
    if t == 0:
        user = ROUND0_PROMPT_TEMPLATE.format(
            self_pos=int(positions[agent_idx]),
            other_positions=other_positions)
    else:
        lines = "\n".join(
            f"- Agent {idx}: {messages[idx] or ''}" for idx in visible)
        user = MULTIROUND_PROMPT_TEMPLATE.format(
            self_pos=int(positions[agent_idx]),
            other_positions=other_positions,
            message_lines=lines)
    return [
        {"role": "system", "content": build_system_prompt(PERSONALITIES[agent_idx])},
        {"role": "user", "content": user},
    ], visible


def initial_positions(seed: int, episode: int, max_steps: int, max_pos: int):
    rng = random.Random(seed + episode + max_steps * 1000)
    return [rng.randint(0, max_pos) for _ in range(4)]


def run_batch(model, tokenizer, states, args):
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        chats, fallbacks, index, visibility = [], [], [], {}
        for state_idx, state in enumerate(active):
            for agent_idx in range(4):
                chat, visible = build_agent_chat(
                    agent_idx, state["positions"], state["messages"],
                    state["confidences"], state["t"], args.margin)
                chats.append(chat)
                fallbacks.append(int(state["positions"][agent_idx]))
                index.append((state_idx, agent_idx))
                visibility[(state_idx, agent_idx)] = visible

        actions, confidences = generate_benign_batched(
            model, tokenizer, chats, fallbacks,
            max_new_tokens=args.max_new_tokens, sample=True,
            temperature=args.temperature, max_pos=args.max_pos)
        proposals = [[None] * 4 for _ in active]
        next_confidences = [[None] * 4 for _ in active]
        for (state_idx, agent_idx), action, confidence in zip(
                index, actions, confidences):
            proposals[state_idx][agent_idx] = (
                int(action.next_position), action.message, bool(action.valid))
            next_confidences[state_idx][agent_idx] = float(confidence)

        for state_idx, state in enumerate(active):
            before = list(state["positions"])
            confidence_before = list(state["confidences"])
            state["positions"] = [item[0] for item in proposals[state_idx]]
            state["messages"] = [item[1] for item in proposals[state_idx]]
            state["confidences"] = list(next_confidences[state_idx])
            state["t"] += 1
            reached = consensus_reached(state["positions"], args.consensus_tol)
            state["consensus"] = reached
            state["done"] = reached or state["t"] >= args.max_steps
            if state["rounds"] is not None:
                state["rounds"].append({
                    "t": state["t"] - 1,
                    "positions_before": before,
                    "positions_after": list(state["positions"]),
                    "confidences_used_for_filter": confidence_before,
                    "confidences_generated_for_next_round": list(state["confidences"]),
                    "visible_others_per_agent": [
                        list(visibility[(state_idx, idx)]) for idx in range(4)],
                    "neutral_agent3_filtered_by": [
                        idx for idx in DEFENDED_IDS
                        if 3 not in visibility[(state_idx, idx)]],
                    "outputs": [
                        {"agent": idx, "personality": PERSONALITIES[idx],
                         "position": proposals[state_idx][idx][0],
                         "message": proposals[state_idx][idx][1],
                         "valid": proposals[state_idx][idx][2],
                         "confidence": next_confidences[state_idx][idx]}
                        for idx in range(4)
                    ],
                    "consensus_after": reached,
                })


def main() -> None:
    self_check()
    parser = argparse.ArgumentParser(
        description="Experiment 3 no-attacker CP-WBFT control")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--condition", required=True,
                        choices=["wm_cpwbft_no_attacker",
                                 "qwen35_9b_cpwbft_no_attacker"])
    parser.add_argument("--n_exp", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_pos", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--consensus_tol", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump_json", required=True)
    parser.add_argument("--dump_n", type=int, default=200)
    args = parser.parse_args()
    if args.n_exp < 1 or args.eval_batch_size < 1:
        parser.error("n_exp and eval_batch_size must be positive")

    print(f"[model] loading {args.model_dir} (HF bf16, confidence enabled)")
    model, tokenizer = load_causal_model(args.model_dir, args.device)
    seed = args.seed + 100_000
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    finished = []
    for start in range(0, args.n_exp, args.eval_batch_size):
        stop = min(start + args.eval_batch_size, args.n_exp)
        states = []
        for episode in range(start, stop):
            init = initial_positions(
                args.seed, episode, args.max_steps, args.max_pos)
            states.append({
                "episode": episode,
                "initial_positions": list(init),
                "positions": list(init),
                "messages": [None] * 4,
                "confidences": [1.0] * 4,
                "t": 0, "done": False, "consensus": False,
                "rounds": [] if episode < args.dump_n else None,
            })
        run_batch(model, tokenizer, states, args)
        finished.extend(states)

    consensus = sum(int(state["consensus"]) for state in finished)
    mean_len = sum(state["t"] for state in finished) / len(finished)
    summary = {
        "episodes": len(finished), "consensus": consensus,
        "rate": consensus / len(finished), "mean_ep_len": mean_len,
    }
    print("\n" + "#" * 74)
    print(f"# EXPERIMENT 3 NO ATTACKER: {args.condition}")
    print("# agents 0-2 CP-WBFT; agent 3 neutral/undefended; consensus agents 0-2")
    print("#" * 74)
    print(f"condition={args.condition:34s} episodes={len(finished):4d} "
          f"consensus={consensus:4d} rate={summary['rate']:.3f} "
          f"mean_ep_len={mean_len:.2f}")

    output = Path(args.dump_json).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "3-no-attacker",
        "condition": args.condition,
        "model_dir": args.model_dir,
        "personalities": PERSONALITIES,
        "defended_agents": list(DEFENDED_IDS),
        "neutral_agent": 3,
        "consensus_scope": "agents_0_1_2",
        "confidence_timing": "previous synchronous round; initial confidence=1",
        "margin": args.margin, "seed": args.seed,
        "summary": summary,
        "episodes": [{
            "episode": state["episode"],
            "initial_positions": state["initial_positions"],
            "final_positions": state["positions"],
            "consensus": state["consensus"], "ep_len": state["t"],
            "rounds": state["rounds"],
        } for state in finished if state["rounds"] is not None],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[dump] wrote {len(payload['episodes'])} trajectories -> {output}")


if __name__ == "__main__":
    main()
