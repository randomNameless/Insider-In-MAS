#!/usr/bin/env python
"""Summarise eval_wm.py outputs against the 9B reference.

openloop: each WM file is paired row-by-row (gid, t, agent) with the 9B ref file on the same test set.
  mean_err  = |mean WM position - mean 9B position| per agent-turn (8 samples each)
  hit       = P(WM sample == the realised 9B move); the ref row's own hit is the noise ceiling
  move_tv   = total-variation distance between pooled WM and 9B distributions of (next - cur)
  stay      = P(next == cur)
Grouped by persona and by round bucket (1, 2, 3, 4+).

    python -m consensus_wm.analyze openloop --ref ref9b_shuffle.jsonl A=A_shuffle.jsonl C=C_shuffle.jsonl
    python -m consensus_wm.analyze closedloop ref=9b_cl_fixed.json A=A_cl_fixed.json ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict


def rows(path):
    return {(r["gid"], r["t"], r["agent"]): r for r in map(json.loads, open(path))}


def bucket(t):
    return str(t + 1) if t < 3 else "4+"


def tv(a, b):
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def stats(pairs):
    err = hit = stay_w = stay_r = 0.0
    dw, dr = [], []
    for w, r in pairs:
        mw = sum(w["samples"]) / len(w["samples"]); mr = sum(r["samples"]) / len(r["samples"])
        err += abs(mw - mr)
        hit += sum(s == r["true_next"] for s in w["samples"]) / len(w["samples"])
        dw += [s - w["cur"] for s in w["samples"]]; dr += [s - r["cur"] for s in r["samples"]]
    n = len(pairs)
    return {"n": n, "mean_err": err / n, "hit": hit / n, "move_tv": tv(dw, dr),
            "stay_wm": sum(d == 0 for d in dw) / len(dw), "stay_9b": sum(d == 0 for d in dr) / len(dr)}


def openloop(args):
    ref = rows(args[args.index("--ref") + 1])
    arms = [x.split("=", 1) for x in args if "=" in x]
    print("| arm | group | n | mean_err | hit | move_tv | stay WM | stay 9B |")
    print("|---|---|---|---|---|---|---|---|")
    for name, path in [("9B-self", None)] + arms:
        W = ref if path is None else rows(path)
        groups = defaultdict(list)
        for k, r in ref.items():
            w = W[k]
            groups["all"].append((w, r))
            groups[r["persona"]].append((w, r))
            groups[f"round {bucket(r['t'])}"].append((w, r))
            groups[f"{r['persona']} r{bucket(r['t'])}"].append((w, r))
        for g in ["all", "stubborn", "suggestible", "neutral", "round 1", "round 2", "round 3", "round 4+"] + \
                 sorted(x for x in groups if " r" in x):
            if g in groups:
                s = stats(groups[g])
                print(f"| {name} | {g} | {s['n']} | {s['mean_err']:.2f} | {s['hit']:.3f} | {s['move_tv']:.3f} "
                      f"| {s['stay_wm']:.2f} | {s['stay_9b']:.2f} |")


def closedloop(args):
    print("| model | consensus | " + " | ".join(f"by r{t}" for t in range(1, 11)) + " |")
    print("|---|---|" + "---|" * 10)
    for name, path in (x.split("=", 1) for x in args):
        G = json.load(open(path))
        cr = [g["consensus_round"] for g in G]
        cum = [sum(c is not None and c <= t for c in cr) / len(G) for t in range(1, 11)]
        print(f"| {name} | {cum[-1]:.3f} | " + " | ".join(f"{c:.2f}" for c in cum) + " |")


if __name__ == "__main__":
    {"openloop": openloop, "closedloop": closedloop}[sys.argv[1]](sys.argv[2:])
