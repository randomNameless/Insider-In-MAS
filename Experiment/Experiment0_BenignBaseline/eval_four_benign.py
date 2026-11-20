#!/usr/bin/env python
"""Experiment 0: four benign LLM agents, no attacker and no defense.

All four agents act synchronously from the same pre-round state. As in
Experiment 1, consensus is computed over agents 0, 1 and 2 only; agent 3 still
participates in every round but is outside the endpoint metric. Personalities are fixed by index to
stubborn/suggestible/neutral/neutral.
"""
from __future__ import annotations

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# RL_attack_env_qwen.py is the external prototype dependency kept beside the
# existing Experiment-1 eval scripts on the run machine. A sibling experiment's
# working directory is not on Python's import path, so add both locations
# explicitly. This also makes `attacker.prompts` resolve without relying on an
# editable installation or the caller's current directory.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP1_EVAL_DIR = HERE.parent / "Experiment1_Eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP1_EVAL_DIR))

from RL_attack_env_qwen import (
    QwenWorldModelHF,
    QwenWorldModelVLLM,
    parse_pos_msg,
)
from attacker.prompts import build_chat


PERSONALITIES = ["stubborn", "suggestible", "neutral", "neutral"]


def consensus_reached(positions, tolerance: float = 0.0) -> bool:
    evaluated = positions[:3]
    if len(evaluated) != 3:
        raise ValueError("consensus requires positions for agents 0, 1 and 2")
    return max(evaluated) - min(evaluated) <= tolerance


def paired_initial_positions(seed: int, episode: int, max_steps: int, max_pos: int):
    # Same initial-state construction as Experiment 1 evaluation.
    rng = random.Random(seed + episode + max_steps * 1000)
    return [rng.randint(0, max_pos) for _ in range(4)]


def prompt_text(wm, messages):
    return wm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **wm.chat_template_kwargs,
    )


def run_active_batch(wm, states, max_pos: int, max_steps: int, tolerance: float):
    """Advance a batch until agents 0--2 reach consensus or time runs out."""
    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        chats = []
        index = []
        for state_idx, state in enumerate(active):
            # Build every agent query before applying any proposal: synchronous.
            for agent_idx, personality in enumerate(PERSONALITIES):
                chats.append(build_chat(
                    agent_idx,
                    personality,
                    state["positions"],
                    state["last_messages"],
                    state["t"],
                    msg_order="index",
                ))
                index.append((state_idx, agent_idx))

        raws = wm._generate([prompt_text(wm, chat) for chat in chats])
        proposals = [[None] * 4 for _ in active]
        for raw, (state_idx, agent_idx) in zip(raws, index):
            fallback = int(active[state_idx]["positions"][agent_idx])
            pos, msg = parse_pos_msg(raw, fallback, max_pos)
            proposals[state_idx][agent_idx] = (int(pos), msg)

        for state, proposal_group in zip(active, proposals):
            before = list(state["positions"])
            new_positions = [item[0] for item in proposal_group]
            new_messages = [item[1] for item in proposal_group]
            state["positions"] = new_positions
            state["last_messages"] = new_messages
            state["t"] += 1
            reached = consensus_reached(new_positions, tolerance)
            state["consensus"] = reached
            state["done"] = reached or state["t"] >= max_steps
            if state["rounds"] is not None:
                state["rounds"].append({
                    "t": state["t"] - 1,
                    "positions_before": before,
                    "positions_after": list(new_positions),
                    "outputs": [
                        {
                            "agent": i,
                            "personality": PERSONALITIES[i],
                            "position": int(new_positions[i]),
                            "message": new_messages[i],
                        }
                        for i in range(4)
                    ],
                    "consensus_after": reached,
                })


def main() -> None:
    p = argparse.ArgumentParser(
        description="Experiment 0: four benign agents, consensus over agents 0--2")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--condition", required=True,
                   choices=["wm_four_benign", "qwen35_9b_four_benign"])
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--consensus_tol", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gpu_mem_util", type=float, default=0.65)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--dump_json", default=None)
    p.add_argument("--dump_n", type=int, default=200)
    args = p.parse_args()

    if args.n_exp < 1 or args.eval_batch_size < 1:
        p.error("--n_exp and --eval_batch_size must be positive")
    if len(PERSONALITIES) != 4:
        raise AssertionError("Experiment 0 requires exactly four personalities")

    model_dir = os.path.expanduser(args.model_dir)
    chat_kwargs = {"enable_thinking": False}
    if args.backend == "vllm":
        wm = QwenWorldModelVLLM(
            model_dir=model_dir,
            max_pos=args.max_pos,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            chat_template_kwargs=chat_kwargs,
            temperature=args.temperature,
            seed=args.seed,
        )
    else:
        wm = QwenWorldModelHF(
            model_dir=model_dir,
            max_pos=args.max_pos,
            max_new_tokens=args.max_new_tokens,
            chat_template_kwargs=chat_kwargs,
        )

    rng_seed = args.seed + 100_000
    random.seed(rng_seed)
    np.random.seed(rng_seed % (2 ** 32))
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    if hasattr(wm, "sampling"):
        wm.sampling.seed = args.seed + 200_000

    finished = []
    for start in range(0, args.n_exp, args.eval_batch_size):
        stop = min(start + args.eval_batch_size, args.n_exp)
        states = []
        for episode in range(start, stop):
            init = paired_initial_positions(
                args.seed, episode, args.max_steps, args.max_pos)
            states.append({
                "episode": episode,
                "initial_positions": list(init),
                "positions": list(init),
                "last_messages": [None] * 4,
                "t": 0,
                "done": False,
                "consensus": False,
                "rounds": [] if args.dump_json and episode < args.dump_n else None,
            })
        run_active_batch(
            wm, states, args.max_pos, args.max_steps, args.consensus_tol)
        finished.extend(states)
        if args.log_every > 0:
            for state in states:
                ep = state["episode"]
                if ep % args.log_every == 0 or ep == args.n_exp - 1:
                    status = "CONSENSUS" if state["consensus"] else "NO CONSENSUS"
                    print(f"[{args.condition}] ep={ep:4d} {status} "
                          f"ep_len={state['t']}", flush=True)

    consensus_count = sum(int(state["consensus"]) for state in finished)
    rate = consensus_count / args.n_exp
    mean_len = sum(state["t"] for state in finished) / args.n_exp
    summary = {
        "episodes": args.n_exp,
        "consensus": consensus_count,
        "rate": rate,
        "mean_ep_len": mean_len,
    }

    print("\n" + "#" * 74)
    print(f"# EXPERIMENT 0: {args.condition}")
    print(f"# model={model_dir}")
    print(f"# personalities={PERSONALITIES}")
    print("# consensus scope=agents 0,1,2; agent 3 participates but is not scored")
    print("#" * 74)
    print(f"condition={args.condition:26s} episodes={args.n_exp:4d} "
          f"consensus={consensus_count:4d} rate={rate:.3f} "
          f"mean_ep_len={mean_len:.2f}", flush=True)

    if args.dump_json:
        out = Path(args.dump_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        records = [{
            "episode": state["episode"],
            "initial_positions": state["initial_positions"],
            "personalities": list(PERSONALITIES),
            "consensus": state["consensus"],
            "ep_len": state["t"],
            "final_positions": state["positions"],
            "rounds": state["rounds"],
        } for state in finished if state["rounds"] is not None]
        payload = {
            "experiment": 0,
            "condition": args.condition,
            "model_dir": model_dir,
            "consensus_scope": "agents_0_1_2",
            "personalities": list(PERSONALITIES),
            "seed": args.seed,
            "n_exp": args.n_exp,
            "summary": summary,
            "episodes": records,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[dump] wrote {len(records)} episodes -> {out}", flush=True)


if __name__ == "__main__":
    main()
