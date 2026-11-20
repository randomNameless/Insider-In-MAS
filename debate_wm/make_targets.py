#!/usr/bin/env python
"""Pick one assigned WRONG option per question for the targeted diagnostic.

Target = the wrong option the honest agents themselves find most attractive in a clean (`none`)
debate: the wrong option most often answered by the honest agents in round 1, tie-broken by their
mean round-1 belief. This mirrors the "static run" target choice in latent_agent_transfer. Computed
once, offline, from an existing `none` state file; the gold answer is used only here, never in a prompt.

    python -m debate_wm.make_targets RUN/state.json --out targets.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = json.load(open(a.state))
    votes = collections.defaultdict(collections.Counter)
    belief = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for d in st["debates"]:
        if d["cond"] != "none" or not d["history"]:
            continue
        hs = D.honest_seats(d["seat"])
        meta[d["task_id"]] = (d["gold"], d["n_options"])
        for k, i in enumerate(hs):
            ans = d["history"][0][i]["answer"]
            if ans and ans != d["gold"]:
                votes[d["task_id"]][ans] += 1
            probs = d["beliefs"][0][k]
            if probs:
                for li, p in enumerate(probs):
                    belief[d["task_id"]][D.LETTERS[li]].append(p)
    targets, stats = {}, collections.Counter()
    for t, (gold, k) in meta.items():
        opts = [c for c in D.LETTERS[:k] if c != gold]
        mb = {c: (sum(belief[t][c]) / len(belief[t][c]) if belief[t][c] else 0.0) for c in opts}
        best = max(opts, key=lambda c: (votes[t][c], mb[c]))
        targets[t] = best
        stats["voted" if votes[t][best] else "belief_only"] += 1
    json.dump({"source": a.state, "rule": "most-voted wrong option in round 1 of `none`, tie-break mean belief",
               "n": len(targets), "targets": targets}, open(a.out, "w"), indent=1)
    print(f"{len(targets)} targets -> {a.out}  ({dict(stats)})")


if __name__ == "__main__":
    main()
