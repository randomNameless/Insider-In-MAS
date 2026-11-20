#!/usr/bin/env python
"""Subsample whole debates (all 12 honest-turn records of a game) from a build_sft.py fit file,
stratified over (cond, seed). 2026-11-11: cheap-WM test, 3,000 real debates.

    python debate_wm/subsample_games.py --inp ~/data/wm/v1_fit.jsonl --n 3000 --out ~/data/wm3k/fit3k.jsonl
"""
import argparse, collections, json, random

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
key = lambda m: (m["task_id"], m["cond"], m["seed"], m["seat"])
strata = collections.defaultdict(set)
for l in open(a.inp):
    m = json.loads(l)["meta"]; strata[(m["cond"], m["seed"])].add(key(m))
rng = random.Random(a.seed)
per = a.n // len(strata); extra = a.n - per * len(strata)
keep = set()
for i, s in enumerate(sorted(strata)):
    keep |= set(rng.sample(sorted(strata[s]), per + (1 if i < extra else 0)))
n = 0
with open(a.out, "w") as f:
    for l in open(a.inp):
        if key(json.loads(l)["meta"]) in keep:
            f.write(l); n += 1
print(f"{len(keep)} games, {n} records -> {a.out}")
