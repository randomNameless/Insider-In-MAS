#!/usr/bin/env python
"""Evaluate persona-masked consensus WMs against the Qwen3.5-9B agents.

openloop:   for every benign agent-turn in a held-out 9B test file, sample --n Position replies
            from the model given the true game prefix. --prompt persona|id builds the WM prompt
            (consensus_wm/build_sft.py); --prompt ref feeds the exact 9B prompt (use with the 9B
            itself to get its own sampling distribution). Writes one row per agent-turn.
closedloop: the model plays all three benign agents (WM prompts with public history) against
            the random-jumper attacker, same init positions/attacker RNG as collect_9b.py
            --llm-attacker 0 with the same seed, so curves are paired with the 9B run.

    python -m consensus_wm.eval_wm openloop --model ~/runs/cwm_sft_C/best --prompt id \
        --inp ~/data/cwm/shuffle_test.jsonl --out ~/runs/cwm_eval/C_on_shuffle.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.action import parse_pos_msg  # noqa: E402
from attacker.prompts import ATTACKER_IDX, BENIGN_IDS, build_user_prompt  # noqa: E402
from consensus_wm.build_sft import game_rounds, history_block, wm_system  # noqa: E402
from consensus_wm.collect_9b import MAX_POS, RANDOM_MSGS, log, new_game  # noqa: E402


def load_llm(path, gpu_mem, seed):
    from vllm import LLM
    return LLM(model=os.path.expanduser(path), gpu_memory_utilization=gpu_mem, max_model_len=4096,
               enable_prefix_caching=True, seed=seed)


def chat(llm, chats, sp):
    return llm.chat(chats, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})


def openloop(a):
    from vllm import SamplingParams
    by_gid = defaultdict(list)
    for line in open(os.path.expanduser(a.inp)):
        r = json.loads(line)
        by_gid[r["gid"]].append(r)
    jobs, chats = [], []
    for gid in sorted(by_gid):
        rounds = game_rounds(by_gid[gid])
        for r in by_gid[gid]:
            if a.prompt == "ref":
                c = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
            else:
                persona = r["persona"] if a.prompt == "persona" else None
                c = [{"role": "system", "content": wm_system(r["agent"], persona)},
                     {"role": "user", "content": history_block(rounds[:r["t"]]) + r["user"]}]
            jobs.append(r); chats.append(c)
    llm = load_llm(a.model, a.gpu_mem, a.seed)
    sp = SamplingParams(n=a.n, temperature=a.temperature, max_tokens=200)
    t0 = time.time()
    outs = chat(llm, chats, sp)
    log(f"{len(chats)} prompts x {a.n} samples in {time.time() - t0:.0f}s")
    out = Path(os.path.expanduser(a.out)); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fo:
        for r, o in zip(jobs, outs):
            cur = r["positions"][r["agent"]]
            acts = [parse_pos_msg(x.text, fallback_pos=cur, max_pos=MAX_POS) for x in o.outputs]
            fo.write(json.dumps({k: r[k] for k in ("gid", "t", "agent", "persona", "attacker", "seat_mode")}
                                | {"cur": cur, "true_next": r["next_position"],
                                   "samples": [x.next_position for x in acts],
                                   "true_message": r["message"],
                                   "messages": [x.message for x in acts],
                                   "valid": sum(x.valid for x in acts) / len(acts)}) + "\n")
    log(f"-> {out}")


def closedloop(a):
    from vllm import SamplingParams
    llm = load_llm(a.model, a.gpu_mem, a.seed)
    sp = SamplingParams(temperature=a.temperature, max_tokens=200)
    games = [new_game(g, a.seat_mode, 0.0, a.seed) for g in range(a.n_games)]
    for g in games:
        g["rounds"] = []
    for t in range(a.max_steps):
        live = [g for g in games if not g["done"]]
        if not live:
            break
        jobs, chats = [], []
        for g in live:
            for i in BENIGN_IDS:
                persona = g["personas"][i] if a.prompt == "persona" else None
                user = history_block(g["rounds"]) + build_user_prompt(i, g["positions"], g["last_messages"], t)
                jobs.append((g, i))
                chats.append([{"role": "system", "content": wm_system(i, persona)},
                              {"role": "user", "content": user}])
        outs = chat(llm, chats, sp)
        new_pos = {id(g): list(g["positions"]) for g in live}
        new_msg = {id(g): list(g["last_messages"]) for g in live}
        for (g, i), o in zip(jobs, outs):
            act = parse_pos_msg(o.outputs[0].text, fallback_pos=g["positions"][i], max_pos=MAX_POS)
            new_pos[id(g)][i], new_msg[id(g)][i] = act.next_position, act.message
        for g in live:  # same RNG call order as collect_9b.py for random attackers
            p = g["rng"].randint(0, MAX_POS)
            new_pos[id(g)][ATTACKER_IDX] = p
            new_msg[id(g)][ATTACKER_IDX] = g["rng"].choice(RANDOM_MSGS).format(p=p)
            g["rounds"].append((list(g["positions"]), new_msg[id(g)]))
            g["positions"], g["last_messages"] = new_pos[id(g)], new_msg[id(g)]
            xs = [g["positions"][i] for i in BENIGN_IDS]
            if max(xs) == min(xs):
                g["done"], g["consensus_round"] = True, t + 1
        log(f"round {t}: {len(live)} live")
    out = Path(os.path.expanduser(a.out)); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump([{"gid": g["gid"], "personas": g["personas"], "attacker": g["attacker"],
                "consensus_round": g["consensus_round"]} for g in games], open(out, "w"))
    log(f"consensus rate {sum(g['consensus_round'] is not None for g in games) / len(games):.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["openloop", "closedloop"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=["persona", "id", "ref"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inp")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--seat-mode", choices=["fixed", "shuffle"], default="fixed")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    openloop(a) if a.mode == "openloop" else closedloop(a)


if __name__ == "__main__":
    main()
