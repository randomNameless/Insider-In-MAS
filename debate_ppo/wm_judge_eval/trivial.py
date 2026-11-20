#!/usr/bin/env python
"""No-model baselines, written in the same prediction format (probs over the valid letters):

  no_change       one-hot on the agent's own previous-round answer (round 1: uniform)
  follow_majority one-hot on the most common previous-round answer among the OTHER three agents
                  (ties -> own previous answer; round 1: uniform)
  follow_fourth   one-hot on the 4th agent's (attacker's / neutral's) previous-round answer
  base_rate       own previous answer with probability 1 - f, the rest spread evenly, where f is the
                  empirical flip rate of rounds 2-4 in the eval set (the "majority class" flip
                  predictor with a constant flip probability; round 1: uniform)
  gt_probs        the 9B's own recorded distribution (NOT a judge: the ceiling for predicting a sample
                  drawn from that distribution; calibration-perfect by construction)

One-hot rows get 1e-6 smoothing so log-loss is finite.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo import dprotocol as D  # noqa: E402

LETTERS = D.LETTERS
EPS = 1e-6


def onehot(letter, k):
    if letter is None or letter not in LETTERS[:k]:
        return [1.0 / k] * k
    p = [EPS] * k
    p[LETTERS.index(letter)] = 1.0
    z = sum(p)
    return [x / z for x in p]


def spread(letter, k, f):
    """1-f on `letter`, f spread evenly on the other k-1 letters."""
    if letter is None or letter not in LETTERS[:k]:
        return [1.0 / k] * k
    p = [f / (k - 1)] * k
    p[LETTERS.index(letter)] = 1.0 - f
    return p


def predict(rec: dict, name: str, flip_rate: float):
    m, k = rec["meta"], rec["n_options"]
    if m["round"] == 1 and name != "gt_probs":
        return [1.0 / k] * k
    if name == "no_change":
        return onehot(m["prev_answer"], k)
    if name == "follow_majority":
        c = Counter(a for a in m["others_prev"].values() if a)
        if not c:
            return onehot(m["prev_answer"], k)
        top = c.most_common()
        best = [a for a, n in top if n == top[0][1]]
        return onehot(best[0] if len(best) == 1 else m["prev_answer"], k)
    if name == "follow_fourth":
        return onehot(m["fourth_prev"], k)
    if name == "base_rate":
        return spread(m["prev_answer"], k, flip_rate)
    if name == "gt_probs":
        return list(rec["gt"]["probs"])
    raise ValueError(name)


NAMES = ("no_change", "follow_majority", "follow_fourth", "base_rate", "gt_probs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.records)]
    later = [r for r in recs if r["meta"]["round"] > 1]
    flip_rate = sum(r["gt"]["flip"] for r in later) / len(later)
    print(f"flip rate rounds 2-4: {flip_rate:.4f}")
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        with open(f"{a.out_prefix}__{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({"id": r["id"], "probs": predict(r, name, flip_rate)}) + "\n")
        json.dump({"model": None, "judge": name, "n": len(recs), "flip_rate": flip_rate,
                   "timing": {"wall_s": 0.0, "n_turns": len(recs), "s_per_turn": 0.0, "gen_tokens": 0,
                              "gen_tokens_per_turn": 0.0, "prompt_tokens": 0}},
                  open(f"{a.out_prefix}__{name}.timing.json", "w"), indent=1)
    print("trivial baselines written")


if __name__ == "__main__":
    main()
