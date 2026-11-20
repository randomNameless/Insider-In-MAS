#!/usr/bin/env python
"""Per-persona behaviour of honest agents in a debate state.json: accuracy by round, answer-change
rate, follow rate (switches TO the 4th agent's previous answer), mean argument length.

    python -m debate_wm.analyze_profiles RUN/state.json [--cond none]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import dprotocol as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--cond", default=None)
    a = ap.parse_args()
    st = json.load(open(a.state))
    agg = defaultdict(lambda: defaultdict(list))
    for d in st["debates"]:
        if d.get("final") is None or (a.cond and d["cond"] != a.cond):
            continue
        hs = D.honest_seats(d["seat"])
        rounds = [[d["history"][r][i]["answer"] for i in hs] for r in range(D.N_ROUNDS)] + \
                 [[x["answer"] for x in d["final"]]]
        args = [[d["history"][r][i]["argument"] for i in hs] for r in range(D.N_ROUNDS)] + \
               [[x["argument"] for x in d["final"]]]
        fourth = [d["history"][r][d["seat"]]["answer"] for r in range(D.N_ROUNDS)]
        for k, a_ in enumerate(hs):
            p = d["personalities"][a_]
            g = agg[(d["cond"], p)]
            for r in range(len(rounds)):
                g[f"acc_r{r + 1}"].append(rounds[r][k] == d["gold"])
                g["words"].append(len(args[r][k].split()))
            for r in range(1, len(rounds)):
                ch = rounds[r][k] != rounds[r - 1][k]
                g["change"].append(ch)
                g["follow4th"].append(ch and rounds[r][k] == fourth[r - 1])
                g["wrong_to_right"].append(rounds[r - 1][k] != d["gold"] and rounds[r][k] == d["gold"])
                g["right_to_wrong"].append(rounds[r - 1][k] == d["gold"] and rounds[r][k] != d["gold"])
    for (c, p), g in sorted(agg.items()):
        s = "  ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in g.items() if k != "words")
        print(f"{c:12s} {p:13s} n={len(g['change']) // 3:4d}  {s}  words={sum(g['words']) / len(g['words']):.1f}")


if __name__ == "__main__":
    main()
