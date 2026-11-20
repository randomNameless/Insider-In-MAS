#!/usr/bin/env python
"""Collect Qwen3.5-9B consensus-game trajectories for persona-masked WM training.

Three benign agents (stubborn/suggestible/neutral) + attacker (Agent 3). All games
run in lockstep through one vLLM engine. Synchronous rounds: every agent reads the
round-start state, then all actions are committed (attacker/game.py semantics).

seat-mode fixed:   Agent 0/1/2 = stubborn/suggestible/neutral in every game.
seat-mode shuffle: a random permutation of the three personas per game.
Attacker per game: 9B with MALICIOUS suffix (prob --llm-attacker) or a random
jumper with templated messages, so the WM sees varied adversarial input.

One output record per benign agent-turn (the exact chat the 9B saw + its reply).

    .venv/bin/python -m consensus_wm.collect_9b --model ~/models/qwen3.5-9b \
        --out ~/data/cwm/fixed_train.jsonl --n-games 3000 --seat-mode fixed --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.action import parse_pos_msg  # noqa: E402
from attacker.game import BENIGN_PERSONALITIES  # noqa: E402
from attacker.prompts import ATTACKER_IDX, BENIGN_IDS, build_chat  # noqa: E402

MAX_POS = 20
RANDOM_MSGS = [
    "Let's all meet at {p}.",
    "I think {p} is the best spot for everyone.",
    "I'm heading to {p}; please come join me.",
    "Position {p} looks like the fairest middle ground.",
    "Everyone should move to {p}, it's closest overall.",
    "I'll wait at {p} so we can gather there.",
]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def new_game(gid, seat_mode, p_llm, seed):
    rng = random.Random(seed * 1_000_003 + gid)
    personas = list(BENIGN_PERSONALITIES)
    if seat_mode == "shuffle":
        rng.shuffle(personas)
    return {
        "gid": gid, "rng": rng, "t": 0, "done": False, "consensus_round": None,
        "positions": [rng.randint(0, MAX_POS) for _ in range(4)],
        "personas": personas + ["malicious"],
        "attacker": "llm" if rng.random() < p_llm else "random",
        "last_messages": [None] * 4,
        "traj": [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-games", type=int, required=True)
    ap.add_argument("--seat-mode", choices=["fixed", "shuffle"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-attacker", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=os.path.expanduser(a.model), gpu_memory_utilization=a.gpu_mem,
              max_model_len=4096, enable_prefix_caching=True, seed=a.seed)
    sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_new_tokens, seed=None)

    games = [new_game(g, a.seat_mode, a.llm_attacker, a.seed) for g in range(a.n_games)]
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rec = 0
    t0 = time.time()
    with open(out, "w") as fo:
        for t in range(a.max_steps):
            live = [g for g in games if not g["done"]]
            if not live:
                break
            jobs, chats = [], []
            for g in live:
                ids = list(BENIGN_IDS) + ([ATTACKER_IDX] if g["attacker"] == "llm" else [])
                for i in ids:
                    jobs.append((g, i))
                    chats.append(build_chat(i, g["personas"][i], g["positions"], g["last_messages"], t))
            outs = llm.chat(chats, sp, use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False})
            new_pos = {id(g): list(g["positions"]) for g in live}
            new_msg = {id(g): list(g["last_messages"]) for g in live}
            for (g, i), chat, o in zip(jobs, chats, outs):
                text = o.outputs[0].text
                act = parse_pos_msg(text, fallback_pos=g["positions"][i], max_pos=MAX_POS)
                new_pos[id(g)][i] = act.next_position
                new_msg[id(g)][i] = act.message
                if i in BENIGN_IDS:
                    rec = {"split_seed": a.seed, "seat_mode": a.seat_mode, "gid": g["gid"], "t": t,
                           "agent": i, "persona": g["personas"][i], "attacker": g["attacker"],
                           "seat_personas": g["personas"][:3],
                           "positions": list(g["positions"]),
                           "last_messages": list(g["last_messages"]),
                           "system": chat[0]["content"], "user": chat[1]["content"],
                           "output": text, "next_position": act.next_position,
                           "message": act.message, "valid": act.valid}
                    fo.write(json.dumps(rec) + "\n")
                    n_rec += 1
            for g in live:
                if g["attacker"] == "random":
                    p = g["rng"].randint(0, MAX_POS)
                    new_pos[id(g)][ATTACKER_IDX] = p
                    new_msg[id(g)][ATTACKER_IDX] = g["rng"].choice(RANDOM_MSGS).format(p=p)
                g["positions"], g["last_messages"] = new_pos[id(g)], new_msg[id(g)]
                g["t"] = t + 1
                xs = [g["positions"][i] for i in BENIGN_IDS]
                if max(xs) == min(xs):
                    g["done"], g["consensus_round"] = True, t + 1
            log(f"round {t}: {len(live)} live games, {len(chats)} calls, {n_rec} records, "
                f"{time.time() - t0:.0f}s")
    summ = out.with_suffix(".games.json")
    json.dump([{"gid": g["gid"], "personas": g["personas"], "attacker": g["attacker"],
                "consensus_round": g["consensus_round"]} for g in games], open(summ, "w"))
    cr = sum(g["consensus_round"] is not None for g in games) / len(games)
    log(f"done: {n_rec} records -> {out}; consensus rate {cr:.3f}")


if __name__ == "__main__":
    main()
