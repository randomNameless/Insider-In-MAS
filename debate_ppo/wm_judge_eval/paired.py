#!/usr/bin/env python
"""Paired differences between two judges on the same turns (rounds 2-4): answer accuracy, flip AUC and
per-attacker-turn reward Pearson, with 95% CIs from a bootstrap over questions (task_id clusters).

    python -m debate_ppo.wm_judge_eval.paired --records R --preds DIR --pairs wm__roleplay_gen:q4b__observer_v1,...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo.wm_judge_eval import metrics as M  # noqa: E402


def stats(rows_a, rows_b, rr_a, rr_b):
    sa, sb = M.summarize(rows_a), M.summarize(rows_b)
    ra, rb = M.reward_summary(rr_a), M.reward_summary(rr_b)
    return {"acc": sa["acc_sample"] - sb["acc_sample"],
            "flip_auc": sa["flip"]["auc"] - sb["flip"]["auc"],
            "reward_r": ra["pearson"] - rb["pearson"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--t-min", type=int, default=0, help="reward transitions t >= this (1 = the two a DCM-style judge can score)")
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.records)]
    out = {}
    for pair in a.pairs.split(","):
        na, nb = pair.split(":")
        ra_rows = [r for r in M.load(records, Path(a.preds) / f"{na}.jsonl") if r["meta"]["round"] > 1]
        rb_rows = [r for r in M.load(records, Path(a.preds) / f"{nb}.jsonl") if r["meta"]["round"] > 1]
        assert [r["id"] for r in ra_rows] == [r["id"] for r in rb_rows]
        full_a = M.load(records, Path(a.preds) / f"{na}.jsonl"); full_b = M.load(records, Path(a.preds) / f"{nb}.jsonl")
        rr_a = [r for r in M.reward_rows(full_a) if r["t"] >= a.t_min]
        rr_b = [r for r in M.reward_rows(full_b) if r["t"] >= a.t_min]
        ka, kb = {(r["key"], r["t"]) for r in rr_a}, {(r["key"], r["t"]) for r in rr_b}
        common = ka & kb
        rr_a = sorted([r for r in rr_a if (r["key"], r["t"]) in common], key=lambda r: (r["key"], r["t"]))
        rr_b = sorted([r for r in rr_b if (r["key"], r["t"]) in common], key=lambda r: (r["key"], r["t"]))
        assert [(r["key"], r["t"]) for r in rr_a] == [(r["key"], r["t"]) for r in rr_b]
        point = stats(ra_rows, rb_rows, rr_a, rr_b)
        gq = defaultdict(list); gr = defaultdict(list)
        for i, r in enumerate(ra_rows):
            gq[r["meta"]["task_id"]].append(i)
        for i, r in enumerate(rr_a):
            gr[r["task_id"]].append(i)
        keys = sorted(gq)
        rng = random.Random(0)
        boots = defaultdict(list)
        for _ in range(a.n_boot):
            ks = rng.choices(keys, k=len(keys))
            ia = [i for k in ks for i in gq[k]]; ir = [i for k in ks for i in gr.get(k, [])]
            s = stats([ra_rows[i] for i in ia], [rb_rows[i] for i in ia], [rr_a[i] for i in ir], [rr_b[i] for i in ir])
            for m, v in s.items():
                boots[m].append(v)
        res = {}
        for m, v in point.items():
            b = sorted(boots[m])
            res[m] = {"diff": v, "ci95": [b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]]}
        out[pair] = res
        print(pair, json.dumps(res))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
