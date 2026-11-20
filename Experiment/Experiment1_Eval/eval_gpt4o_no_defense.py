#!/usr/bin/env python
"""Synchronous Experiment-1 transfer evaluation with GPT-4o benign agents.

Conditions: no attack, Random, Extreme, Prompt-based, DQN+Rephrase,
SFT Model Base, and Model-based RL PPO.  All four actions in a round are
generated from one immutable round-start snapshot and committed atomically.
Consensus is scored on benign agents 0, 1, and 2 exactly as in Experiment 1.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from RL_attack_env_qwen import AttackEnv, parse_pos_msg
from attacker.prompts import build_chat
from eval_attacks_consensus import (
    HFQwen3,
    MIXED_PERSONALITIES,
    load_dqn,
    make_attacker_fn,
)


CONDITIONS = [
    "no_attack", "random", "extreme", "prompt", "dqn", "model_base", "model_rl"
]
CONTROL_PERSONALITIES = ["stubborn", "suggestible", "neutral", "neutral"]


class OpenAIWorldModel:
    """Small concurrent Responses-API adapter used only for benign actions."""

    def __init__(self, model: str, max_pos: int, max_output_tokens: int,
                 temperature: float, workers: int, retries: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the official SDK: pip install -U openai") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI()
        self.model = model
        self.max_pos = int(max_pos)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.workers = int(workers)
        self.retries = int(retries)

    def _one(self, messages):
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    store=False,
                )
                return response.output_text or ""
            except Exception as exc:  # SDK exception classes vary by release.
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(30.0, 2.0 ** attempt + random.random()))
        raise RuntimeError(f"OpenAI request failed after {self.retries + 1} attempts") from last_error

    def generate_chats(self, chats):
        if not chats:
            return []
        outputs = [None] * len(chats)
        with ThreadPoolExecutor(max_workers=min(self.workers, len(chats))) as pool:
            futures = {pool.submit(self._one, chat): idx for idx, chat in enumerate(chats)}
            for future in as_completed(futures):
                outputs[futures[future]] = future.result()
        return outputs


def paired_initial_positions(seed, episode, max_steps, max_pos):
    rng = random.Random(seed + episode + max_steps * 1000)
    return [rng.randint(0, max_pos) for _ in range(4)]


def make_env(wm, args):
    return AttackEnv(
        world_model=wm,
        max_pos=args.max_pos,
        max_steps=args.max_steps,
        consensus_tol=args.consensus_tol,
        consensus_penalty=args.consensus_penalty,
        reward_mode="spread",
        attacker_msg_mode="template",
    )


def run_batch(condition, envs, wm, attacker_fn, initial_positions, record_flags):
    states = []
    personalities = CONTROL_PERSONALITIES if condition == "no_attack" else MIXED_PERSONALITIES
    for env, init, record in zip(envs, initial_positions, record_flags):
        env.reset(options={"positions": list(init), "personalities": list(personalities)})
        states.append({"env": env, "done": False, "consensus": False, "ep_len": 0,
                       "rounds": [] if record else None})

    while any(not state["done"] for state in states):
        active = [state for state in states if not state["done"]]
        active_envs = [state["env"] for state in active]

        if condition == "no_attack":
            agent3_chats = [build_chat(
                3, "neutral", [int(x) for x in env.pos], list(env.last_messages),
                int(env.t), msg_order="index") for env in active_envs]
            raw3 = wm.generate_chats(agent3_chats)
            attacker_outputs = [parse_pos_msg(raw, int(env.pos[3]), env.max_pos)
                                for raw, env in zip(raw3, active_envs)]
            # parse_pos_msg returns (position, message), same as attacker policies.
        elif hasattr(attacker_fn, "batch"):
            attacker_outputs = attacker_fn.batch(active_envs)
        else:
            attacker_outputs = [attacker_fn(env) for env in active_envs]

        benign_chats = []
        pending = []
        for state, (agent3_pos, agent3_msg) in zip(active, attacker_outputs):
            env = state["env"]
            snapshot_pos = [int(x) for x in env.pos]
            snapshot_msg = list(env.last_messages)
            chats = [env._build_messages(i) for i in range(3)]
            benign_chats.extend(chats)
            pending.append((snapshot_pos, snapshot_msg, int(agent3_pos), agent3_msg))

        benign_raws = wm.generate_chats(benign_chats)
        pred_groups = []
        offset = 0
        for state in active:
            env = state["env"]
            group = [parse_pos_msg(raw, int(env.pos[i]), env.max_pos)
                     for i, raw in enumerate(benign_raws[offset:offset + 3])]
            pred_groups.append(group)
            offset += 3

        for state, preds, saved in zip(active, pred_groups, pending):
            env = state["env"]
            snapshot_pos, snapshot_msg, agent3_pos, agent3_msg = saved
            # Atomic commit starts only after all four outputs were generated.
            env.prepare(agent3_pos)
            env._pending_atk_msg = agent3_msg
            _, _, terminated, truncated, _ = env.apply(preds)
            state["ep_len"] += 1
            state["consensus"] = bool(terminated)
            state["done"] = bool(terminated or truncated)
            if state["rounds"] is not None:
                state["rounds"].append({
                    "t": int(env.t) - 1,
                    "snapshot_positions": snapshot_pos,
                    "snapshot_messages": snapshot_msg,
                    "agent3_output": {"position": agent3_pos, "message": agent3_msg},
                    "benign_outputs": [{"agent": i, "position": int(pos), "message": msg}
                                       for i, (pos, msg) in enumerate(preds)],
                    "positions_after": [int(x) for x in env.pos],
                    "consensus_after": bool(terminated),
                })
    return states


def main():
    p = argparse.ArgumentParser(description="Experiment 1 no-defense transfer to GPT-4o")
    p.add_argument("--openai_model", default="gpt-4o-2024-08-06")
    p.add_argument("--llm_model_dir", required=True)
    p.add_argument("--rl_adapter", required=True)
    p.add_argument("--dqn_ckpt", required=True)
    p.add_argument("--dqn_device", default="cpu")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--consensus_tol", type=float, default=0.0)
    p.add_argument("--consensus_penalty", type=float, default=-10.0)
    p.add_argument("--openai_temperature", type=float, default=0.7)
    p.add_argument("--openai_max_output_tokens", type=int, default=200)
    p.add_argument("--openai_workers", type=int, default=16)
    p.add_argument("--openai_retries", type=int, default=6)
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--log_every", type=int, default=0)
    p.add_argument("--dump_json", required=True)
    p.add_argument("--dump_n", type=int, default=200)
    args = p.parse_args()

    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        p.error(f"unknown conditions: {unknown}")
    if args.n_exp < 1 or args.eval_batch_size < 1 or args.openai_workers < 1:
        p.error("episode, batch, and worker counts must be positive")

    wm = OpenAIWorldModel(args.openai_model, args.max_pos,
                          args.openai_max_output_tokens, args.openai_temperature,
                          args.openai_workers, args.openai_retries)
    attack_conditions = [x for x in conditions if x != "no_attack"]
    llm = None
    q_net = None
    attacker_fns = {}
    if attack_conditions:
        llm = HFQwen3(
            args.llm_model_dir,
            adapter_dir=(args.rl_adapter if "model_rl" in attack_conditions else None),
            device="cuda", max_new_tokens=args.max_new_tokens,
            temperature=args.llm_temperature, seed=args.seed)
        if "dqn" in attack_conditions:
            q_net = load_dqn(args.dqn_ckpt, torch.device(args.dqn_device))
        salts = {"random": 11, "extreme": 23, "prompt": 37,
                 "dqn": 53, "model_base": 67, "model_rl": 71}
        attacker_fns = {kind: make_attacker_fn(
            kind, llm, q_net, torch.device(args.dqn_device),
            random.Random(args.seed + salts[kind]), args.max_pos)
            for kind in attack_conditions}

    initial = [paired_initial_positions(
        args.seed, ep, args.max_steps, args.max_pos) for ep in range(args.n_exp)]
    summaries = {}
    records = []
    for condition in conditions:
        policy_seed = args.seed + 100_000
        random.seed(policy_seed)
        np.random.seed(policy_seed % (2 ** 32))
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        finished = []
        for start in range(0, args.n_exp, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, args.n_exp)
            idxs = list(range(start, stop))
            states = run_batch(
                condition, [make_env(wm, args) for _ in idxs], wm,
                attacker_fns.get(condition), [initial[i] for i in idxs],
                [i < args.dump_n for i in idxs])
            for idx, state in zip(idxs, states):
                state["episode"] = idx
                finished.append(state)
                if args.log_every and (idx % args.log_every == 0 or idx == args.n_exp - 1):
                    print(f"[{condition}] {idx + 1}/{args.n_exp} "
                          f"consensus={state['consensus']} ep_len={state['ep_len']}", flush=True)
        count = sum(int(s["consensus"]) for s in finished)
        summaries[condition] = {
            "episodes": args.n_exp, "consensus": count,
            "rate": count / args.n_exp,
            "mean_ep_len": sum(s["ep_len"] for s in finished) / args.n_exp,
        }
        for state in finished:
            if state["rounds"] is not None:
                records.append({
                    "condition": condition, "episode": state["episode"],
                    "initial_positions": initial[state["episode"]],
                    "consensus": state["consensus"], "ep_len": state["ep_len"],
                    "rounds": state["rounds"],
                })
        s = summaries[condition]
        print(f"condition={condition:12s} episodes={s['episodes']:4d} "
              f"consensus={s['consensus']:4d} rate={s['rate']:.3f} "
              f"mean_ep_len={s['mean_ep_len']:.2f}", flush=True)

    out = Path(args.dump_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": 1, "defense": "none", "victim_model": args.openai_model,
        "round_timing": "synchronous_snapshot_atomic_commit",
        "consensus_scope": "agents_0_1_2", "seed": args.seed,
        "n_exp_per_condition": args.n_exp, "summary": summaries, "episodes": records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dump] wrote {len(records)} trajectories -> {out}", flush=True)


if __name__ == "__main__":
    main()
