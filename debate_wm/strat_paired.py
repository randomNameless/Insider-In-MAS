#!/usr/bin/env python
"""Paired comparisons for a strategy-catalog DPO eval run (2026-10-24), beyond the "vs none" table of summary.txt:
condition-vs-condition bootstrap diffs (same questions / seeds / seats), per-seed deltas vs none, and the rate of
messages that satisfy the catalog's message rules (strategies.check_message).

  python debate_wm/strat_paired.py RUN [--pairs stratdpo_q3_4b:strat_q3_4b,stratdpo_q3_4b:strat_q3_4b__D1]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo.eval_debate import _boot_diff, debate_metrics, load_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pairs", default="stratdpo_q3_4b:strat_q3_4b", help="a:b,... -> a - b")
    a = ap.parse_args()
    st = load_state(Path(a.run))
    per = defaultdict(lambda: defaultdict(list))
    seed = defaultdict(lambda: defaultdict(list))
    for d in st["debates"]:
        m = debate_metrics(d)
        per[d["cond"]][d["task_id"]].append(m)
        seed[(d["cond"], d["seed"])][d["task_id"]].append(m)
    out = {}
    for pr in a.pairs.split(","):
        x, y = pr.split(":")
        if x in per and y in per:
            for k in ("final_acc", "majority_on_target"):
                out[f"{x} - {y} {k}"] = _boot_diff(per[x], per[y], k)

    def acc(c, s):
        ms = [m for v in seed[(c, s)].values() for m in v]
        return sum(m["final_acc"] for m in ms) / len(ms)
    seeds = sorted({s for _, s in seed})
    for c in per:
        if c != "none":
            out[f"{c} per-seed vs none"] = {s: round(acc(c, s) - acc("none", s), 3) for s in seeds}
    chk = defaultdict(lambda: [0, 0])
    for d in st["debates"]:
        for e in d.get("strat", []):
            if "check" in e:
                chk[d["cond"]][0] += e["check"]["valid"]
                chk[d["cond"]][1] += 1
    out["catalog_rule_valid_rate"] = {c: round(v / n, 3) for c, (v, n) in chk.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
