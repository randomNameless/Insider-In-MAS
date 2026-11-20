#!/usr/bin/env python
"""Rejection sampling for adversarial training of the honest debate agents (2026-11-08).

For every attacked prompt (debate_wm/build_at_data.py) the victim model writes K replies with the evaluation decoder
(argument sampled at T=0.7, stopped at "Answer:", then the option letter sampled from the readout at T=0.7; the same
honest_round rule as eval_debate.py). A reply is accepted when its sampled letter equals the target answer:
    --target cons : the clean-run answer of the same agent / round (consistency; gold never used)
    --target gold : the gold answer (upper-bound variant)
One accepted reply per prompt is kept (the first accepted in seed order, so the kept text is an ordinary sample from the
model's own distribution, not a confidence-maximised one). Output = SFT records of debate_wm/sft_train.py:
    {messages, target "<argument>\\nAnswer: X", probs (readout after the kept argument), n_options, meta}
All K candidates and their letters are kept in --cands-out for the record.

    python -m debate_wm.at_sample --model ~/models/qwen3.5-9b --in ~/data/at/fit_attacked.jsonl \
        --out ~/data/at/fit_cons_sft.jsonl --cands-out ~/data/at/fit_cons_cands.jsonl --target cons --k 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402
from debate_wm.build_sft import target_text  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cands-out", required=True)
    ap.add_argument("--target", choices=["cons", "gold"], default="cons")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-tag", default="", help="extra seed salt (a second sampling pass over the same prompts)")
    a = ap.parse_args()
    from debate_ppo.engines import make_vllm_engine
    recs = [json.loads(l) for l in open(a.inp)]
    if a.limit:
        recs = recs[: a.limit]
    key = "clean_answer" if a.target == "cons" else "gold"
    recs = [r for r in recs if r[key]]
    log(f"{len(recs)} prompts, K={a.k}, target={a.target}")
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=6144)
    chats, nopt, seeds, owner = [], [], [], []
    for n, r in enumerate(recs):
        for k in range(a.k):
            chats.append(r["messages"]); nopt.append(r["n_options"])
            seeds.append(D.stable_seed("at_sample" + a.seed_tag, r["meta"]["task_id"], r["meta"]["seed"], r["meta"]["round"], r["meta"]["agent"], k))
            owner.append(n)
    outs = []
    B = 4096
    for s in range(0, len(chats), B):
        outs += eng.argue_then_read(chats[s:s + B], nopt[s:s + B], seeds=seeds[s:s + B], temperature=a.temperature,
                                    max_tokens=a.max_tokens)
        log(f"  {min(s + B, len(chats))}/{len(chats)} candidates")
    cands = [[] for _ in recs]
    for n, sd, o in zip(owner, seeds, outs):
        ans = D.sample_answer(o["probs"], a.temperature, sd)
        cands[n].append({"argument_text": o["argument_text"], "argument": o["argument"], "answer": ans, "probs": o["probs"],
                         "truncated": o["truncated"], "valid": bool(o["argument"])})
    st = Counter()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f, open(a.cands_out, "w") as g:
        for r, cs in zip(recs, cands):
            tgt = r[key]
            ok = [c for c in cs if c["valid"] and not c["truncated"] and c["answer"] == tgt]
            g.write(json.dumps({"meta": r["meta"], "target_answer": tgt, "cands": cs, "accepted": len(ok)}) + "\n")
            st["prompts"] += 1
            st[f"accepted_{min(len(ok), a.k)}"] += 1
            if not ok:
                continue
            c = ok[0]
            raw = c["argument_text"] + ("" if c["argument_text"].endswith("\n") else "\n") + "Answer: " + tgt
            f.write(json.dumps({"messages": r["messages"], "target": target_text(raw, tgt), "probs": c["probs"],
                                "n_options": r["n_options"],
                                "meta": {**r["meta"], "source": f"rs_{a.target}", "target_answer": tgt, "n_accepted": len(ok)}}) + "\n")
            st["kept"] += 1
    log(json.dumps(st, indent=1))
    json.dump({"args": vars(a), "stats": st}, open(a.out + ".stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
