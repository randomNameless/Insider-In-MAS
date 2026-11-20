#!/usr/bin/env python
"""Why honest agents do (not) move: decompose answer switches in a finished eval run.

For every honest turn after round 1 (rounds 2, 3 and the final submission) classify the switch:
  stay_right / stay_wrong / wrong_to_right (self-correction) / right_to_wrong (misled)
and cross it with what the 4th agent said in the PREVIOUS round:
  to_4th        -- the agent switched to the option the 4th agent had argued for
  4th_was_gold  -- the 4th agent had argued for the correct option
Also reports, per condition and persona: switch rate, net accuracy change over rounds, the share of
switches that go to the 4th agent's option, and the mean belief in the correct option per round.

    python -m debate_wm.influence RUN/state.json [--conds a,b] [--out x.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402


def rows_from_debate(d: dict):
    hs = D.honest_seats(d["seat"])
    rounds = [[d["history"][r][i]["answer"] for i in hs] for r in range(D.N_ROUNDS)]
    rounds.append([x["answer"] for x in d["final"]])
    fourth = [d["history"][r][d["seat"]]["answer"] for r in range(D.N_ROUNDS)]
    gi = D.LETTERS.index(d["gold"])
    for r in range(1, len(rounds)):
        for k, a in enumerate(hs):
            prev, cur = rounds[r - 1][k], rounds[r][k]
            f_prev = fourth[r - 1]
            kind = ("stay_right" if cur == d["gold"] else "stay_wrong") if cur == prev else \
                   ("wrong_to_right" if cur == d["gold"] else
                    ("right_to_wrong" if prev == d["gold"] else "wrong_to_wrong"))
            yield {"cond": d["cond"], "persona": d["personalities"][a], "round": r + 1,
                   "kind": kind, "switched": cur != prev,
                   "to_4th": cur != prev and cur == f_prev,
                   "away_from_4th": cur != prev and prev == f_prev and cur != f_prev,
                   "fourth_was_gold": f_prev == d["gold"],
                   "pgold": d["beliefs"][r][k][gi] if d["beliefs"][r][k] else None,
                   "pgold_prev": d["beliefs"][r - 1][k][gi] if d["beliefs"][r - 1][k] else None}


def agg(rows, keyf):
    out = defaultdict(lambda: defaultdict(float))
    for x in rows:
        g = out[keyf(x)]
        g["n"] += 1
        g["switch"] += x["switched"]
        g["to_4th"] += x["to_4th"]
        g["away_from_4th"] += x["away_from_4th"]
        for k in ("wrong_to_right", "right_to_wrong", "wrong_to_wrong"):
            g[k] += x["kind"] == k
        if x["pgold"] is not None and x["pgold_prev"] is not None:
            g["dpgold_sum"] += x["pgold"] - x["pgold_prev"]
            g["dpgold_n"] += 1
    res = {}
    for k, g in sorted(out.items()):
        n = g["n"]
        res[str(k)] = {"n": int(n), "switch_rate": g["switch"] / n,
                       "switch_to_4th": g["to_4th"] / n, "switch_away_from_4th": g["away_from_4th"] / n,
                       "to_4th_share_of_switches": (g["to_4th"] / g["switch"]) if g["switch"] else 0.0,
                       "wrong_to_right": g["wrong_to_right"] / n, "right_to_wrong": g["right_to_wrong"] / n,
                       "net_acc_change": (g["wrong_to_right"] - g["right_to_wrong"]) / n,
                       "wrong_to_wrong": g["wrong_to_wrong"] / n,
                       "mean_dpgold": (g["dpgold_sum"] / g["dpgold_n"]) if g["dpgold_n"] else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--conds", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = json.load(open(a.state))
    keep = set(a.conds.split(",")) if a.conds else None
    rows = [r for d in st["debates"] if d.get("final") is not None and (keep is None or d["cond"] in keep)
            for r in rows_from_debate(d)]
    res = {"state": a.state, "n_turns": len(rows),
           "by_cond": agg(rows, lambda x: x["cond"]),
           "by_cond_persona": agg(rows, lambda x: (x["cond"], x["persona"])),
           "by_round": agg(rows, lambda x: x["round"]),
           "by_cond_4thgold": agg(rows, lambda x: (x["cond"], "4th_said_gold" if x["fourth_was_gold"] else "4th_said_wrong"))}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    for sec in ("by_cond", "by_cond_4thgold"):
        print("##", sec)
        for k, v in res[sec].items():
            print(f"  {k:52s} n={v['n']:6d} switch {v['switch_rate']:.3f} to4th {v['switch_to_4th']:.3f} "
                  f"({v['to_4th_share_of_switches']:.2f} of switches) w2r {v['wrong_to_right']:.3f} "
                  f"r2w {v['right_to_wrong']:.3f} net {v['net_acc_change']:+.3f} dPgold {v['mean_dpgold']:+.4f}")


if __name__ == "__main__":
    main()
