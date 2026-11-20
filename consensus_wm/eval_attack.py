#!/usr/bin/env python
"""Attacker vs the REAL Qwen3.5-9B honest agents (synchronous game, attacker/game.py semantics).

Honest agents: 9B via vLLM with their true persona prompts (exactly as in collect_9b.py).
Attacker (Agent 3): --attacker rl   = Qwen3-4B + PPO LoRA adapter (--adapter), MALICIOUS prompt,
                                      two-phase templated generation (attacker/generation.py);
                    --attacker base = same without the adapter (prompt-only attacker);
                    --attacker random = random jumper with templated messages (collect_9b.py).
Seats fixed (stubborn/suggestible/neutral) or shuffled per game. Paired init positions across
attackers via --seed. Writes one JSON per run: per-game consensus round + transcripts.

    python -m consensus_wm.eval_attack --attacker rl --adapter ~/runs/cwm_ppo_C/best \
        --out ~/runs/cwm_attack/C_fixed.json --n-games 300 --seed 31
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.action import parse_pos_msg  # noqa: E402
from attacker.generation import generate_attacker_batched  # noqa: E402
from attacker.prompts import ATTACKER_IDX, BENIGN_IDS, build_chat  # noqa: E402
from consensus_wm.collect_9b import MAX_POS, RANDOM_MSGS, log, new_game  # noqa: E402


def load_attacker(base, adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", choices=["rl", "base", "random"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=os.path.expanduser("~/models/qwen3-4b"))
    ap.add_argument("--victim", default=os.path.expanduser("~/models/qwen3.5-9b"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-games", type=int, default=300)
    ap.add_argument("--seat-mode", choices=["fixed", "shuffle"], default="fixed")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--atk-batch", type=int, default=32)
    ap.add_argument("--atk-max-tokens", type=int, default=48)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    a = ap.parse_args()
    if a.attacker == "rl" and not a.adapter:
        ap.error("--attacker rl needs --adapter")

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.victim, gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              enable_prefix_caching=True, seed=a.seed)
    sp = SamplingParams(temperature=0.7, max_tokens=200)
    atk = atk_tok = None
    if a.attacker in ("rl", "base"):
        atk, atk_tok = load_attacker(a.base, a.adapter if a.attacker == "rl" else None)
    torch.manual_seed(a.seed)

    games = [new_game(g, a.seat_mode, 0.0, a.seed) for g in range(a.n_games)]
    for g in games:
        g["log"] = []
    t0 = time.time()
    for t in range(a.max_steps):
        live = [g for g in games if not g["done"]]
        if not live:
            break
        # honest agents read the round-start snapshot
        chats = [build_chat(i, g["personas"][i], g["positions"], g["last_messages"], t)
                 for g in live for i in BENIGN_IDS]
        outs = llm.chat(chats, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        new_pos = {id(g): list(g["positions"]) for g in live}
        new_msg = {id(g): list(g["last_messages"]) for g in live}
        k = 0
        for g in live:
            for i in BENIGN_IDS:
                act = parse_pos_msg(outs[k].outputs[0].text, fallback_pos=g["positions"][i], max_pos=MAX_POS)
                new_pos[id(g)][i], new_msg[id(g)][i] = act.next_position, act.message
                k += 1
        # attacker reads the same snapshot
        if a.attacker == "random":
            for g in live:
                p = g["rng"].randint(0, MAX_POS)
                new_pos[id(g)][ATTACKER_IDX] = p
                new_msg[id(g)][ATTACKER_IDX] = g["rng"].choice(RANDOM_MSGS).format(p=p)
        else:
            for s in range(0, len(live), a.atk_batch):
                chunk = live[s:s + a.atk_batch]
                achats = [build_chat(ATTACKER_IDX, "malicious", g["positions"], g["last_messages"], t)
                          for g in chunk]
                gts = generate_attacker_batched(atk, atk_tok, achats, [g["positions"][ATTACKER_IDX] for g in chunk],
                                                max_msg_tokens=a.atk_max_tokens, sample=True,
                                                temperature=1.0, max_pos=MAX_POS)
                for g, gt in zip(chunk, gts):
                    new_pos[id(g)][ATTACKER_IDX] = max(0, min(MAX_POS, gt.next_position))
                    new_msg[id(g)][ATTACKER_IDX] = gt.message
        for g in live:
            g["log"].append({"t": t, "positions": list(g["positions"]), "broadcast": new_msg[id(g)]})
            g["positions"], g["last_messages"] = new_pos[id(g)], new_msg[id(g)]
            xs = [g["positions"][i] for i in BENIGN_IDS]
            if max(xs) == min(xs):
                g["done"], g["consensus_round"] = True, t + 1
        log(f"round {t}: {len(live)} live, {time.time() - t0:.0f}s")
    out = Path(os.path.expanduser(a.out)); out.parent.mkdir(parents=True, exist_ok=True)
    res = {"args": vars(a), "games": [{"gid": g["gid"], "personas": g["personas"],
                                       "consensus_round": g["consensus_round"], "log": g["log"]} for g in games]}
    json.dump(res, open(out, "w"))
    cr = sum(g["consensus_round"] is not None for g in games) / len(games)
    log(f"attacker={a.attacker} seats={a.seat_mode}: consensus rate {cr:.3f} over {len(games)} games -> {out}")


if __name__ == "__main__":
    main()
