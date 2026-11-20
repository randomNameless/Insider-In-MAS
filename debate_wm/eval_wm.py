#!/usr/bin/env python
"""Turn-level accuracy of a debate world model against the teacher (Qwen3.5-9B) on held-out turns.

For every teacher turn (debate_wm/build_sft.py records from TEST debates) the candidate model gets
the identical context and we compare option distributions with the teacher's recorded one:

  tf  : teacher-forced -- prompt + TEACHER's own argument + "Answer:"  -> student letter distribution
        (does the model read the same belief out of the same reasoning?)
  gen : free-running   -- prompt -> student writes its own argument -> "Answer:" readout
        (what a WM honest agent would actually do in a rollout; same sampler as training)

Metrics per mode: top-1 agreement (argmax), sampled-answer agreement rate, total-variation distance,
KL(teacher||model), |dP(gold)|, Pearson r of P(gold), and the answer-change behaviour vs the previous
round. Broken down by persona, round and 4th-agent condition.

    python -m debate_wm.eval_wm --model PATH --records data/test9b.jsonl --n 12000 --out results/wm_X.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402

LETTERS = D.LETTERS


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def kl(p, q):
    return sum(a * (math.log(max(a, 1e-8)) - math.log(max(b, 1e-8))) for a, b in zip(p, q))


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def summarize(rows):
    out = {"n": len(rows)}
    for mode in ("tf", "gen"):
        rs = [r for r in rows if mode in r]
        if not rs:
            continue
        out[mode] = {
            "top1_agree": sum(r[mode]["top1"] == r["t_top1"] for r in rs) / len(rs),
            "sample_agree": sum(r[mode]["top1"] == r["t_ans"] for r in rs) / len(rs),
            "tv": sum(r[mode]["tv"] for r in rs) / len(rs),
            "kl": sum(r[mode]["kl"] for r in rs) / len(rs),
            "abs_dpgold": sum(abs(r[mode]["pgold"] - r["t_pgold"]) for r in rs) / len(rs),
            "pgold_r": pearson([r[mode]["pgold"] for r in rs], [r["t_pgold"] for r in rs]),
            "mean_pgold_model": sum(r[mode]["pgold"] for r in rs) / len(rs),
            "mean_pgold_teacher": sum(r["t_pgold"] for r in rs) / len(rs),
            "acc_model_top1": sum(r[mode]["top1"] == r["gold"] for r in rs) / len(rs),
            "acc_teacher_top1": sum(r["t_top1"] == r["gold"] for r in rs) / len(rs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--n", type=int, default=12000, help="turns sampled (stratified by persona x round)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="tf,gen")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-text", default=None, help="jsonl: generated argument per turn (gen mode)")
    a = ap.parse_args()

    from debate_ppo.engines import make_vllm_engine
    from protocol import generation_prompt  # debate_pilot/protocol.py

    recs = [json.loads(l) for l in open(a.records)]
    strata = defaultdict(list)
    for r in recs:
        strata[(r["meta"]["persona"], r["meta"]["round"])].append(r)
    rng = random.Random(a.seed)
    per = max(1, a.n // len(strata))
    sample = [r for k in sorted(strata) for r in rng.sample(strata[k], min(per, len(strata[k])))]
    print(f"{len(sample)} turns from {len(recs)} records, {len(strata)} strata", flush=True)

    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=6144)
    modes = a.modes.split(",")
    rows = []
    for r in sample:
        t = r["probs"]
        t_top1 = LETTERS[max(range(len(t)), key=t.__getitem__)]
        rows.append({"meta": r["meta"], "gold": r["meta"]["gold"], "t_top1": t_top1, "t_ans": r["answer"],
                     "t_pgold": t[LETTERS.index(r["meta"]["gold"])]})
    gi = [LETTERS.index(r["meta"]["gold"]) for r in sample]
    nopt = [r["n_options"] for r in sample]

    def fill(mode, probs):
        for row, p, g in zip(rows, probs, gi):
            row[mode] = {"top1": LETTERS[max(range(len(p)), key=p.__getitem__)], "pgold": p[g],
                         "tv": tv(row_t[id(row)], p), "kl": kl(row_t[id(row)], p)}

    row_t = {id(row): r["probs"] for row, r in zip(rows, sample)}
    t0 = time.time()
    if "tf" in modes:
        prompts = [generation_prompt(eng.tok, r["messages"]) + r["target"][: r["target"].rfind("Answer:")] + "Answer:"
                   for r in sample]
        probs, _ = eng.readout(prompts, nopt)
        fill("tf", probs)
        print(f"tf done {time.time() - t0:.0f}s", flush=True)
    if "gen" in modes:
        seeds = [D.stable_seed("wm_eval", r["meta"]["task_id"], r["meta"]["seed"], r["meta"]["round"],
                               r["meta"]["agent"], r["meta"]["cond"]) for r in sample]
        outs = eng.argue_then_read([r["messages"] for r in sample], nopt, seeds=seeds, temperature=0.7,
                                   max_tokens=160)
        fill("gen", [o["probs"] for o in outs])
        for row, o in zip(rows, outs):
            row["gen"]["truncated"] = o["truncated"]
        if a.save_text:
            with open(a.save_text, "w") as f:
                for r, row, o in zip(sample, rows, outs):
                    f.write(json.dumps({"meta": r["meta"], "teacher_target": r["target"],
                                        "model_text": o["argument_text"], "model_top1": row["gen"]["top1"],
                                        "teacher_answer": r["answer"]}) + "\n")
        print(f"gen done {time.time() - t0:.0f}s", flush=True)

    res = {"model": a.model, "records": a.records, "n_turns": len(rows), "seed": a.seed,
           "overall": summarize(rows),
           "by_persona": {k: summarize([r for r in rows if r["meta"]["persona"] == k])
                          for k in sorted({r["meta"]["persona"] for r in rows})},
           "by_round": {k: summarize([r for r in rows if r["meta"]["round"] == k])
                        for k in sorted({r["meta"]["round"] for r in rows})},
           "by_cond": {k: summarize([r for r in rows if r["meta"]["cond"] == k])
                       for k in sorted({r["meta"]["cond"] for r in rows})}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["overall"], indent=1))


if __name__ == "__main__":
    main()
