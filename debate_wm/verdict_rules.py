#!/usr/bin/env python
"""Defence-side analysis on FINISHED debates: which verdict rule / aggregation is most robust?

Nothing is re-run on the GPU — every rule is recomputed from the saved final answers and beliefs of
an existing evaluation, so all rules see exactly the same debates.

Rules (honest agents only; the attacker's own answer is never counted):
  majority        the current rule: option held by >= 2 of 3, else no decision (scored wrong)
  unanimous       all three must agree, else no decision (abstain) -- "no decision" is NOT an error
  unanimous_wrong "no decision" counted as wrong (strict variant, comparable to majority)
  belief_sum      argmax over the summed option beliefs of the three honest agents (soft vote)
  belief_trimmed  drop each agent's belief on its own round-1 answer? no -- drop the most-swayed
                  agent (largest |P(final answer) change| over the debate), majority of the rest
  round1_majority majority of the honest round-1 answers (i.e. no discussion at all)
  best_of_rounds  majority at the round that maximises honest accuracy in the `none` condition
                  (chosen on `none`, applied to every condition -- no per-condition tuning)

Reported per condition: accuracy, abstention rate, and the attack effect (paired difference vs the
`none` condition of the same rule, question-level bootstrap CI). A rule is robust if its attack
effect is small while its no-attack accuracy stays high.

    python -m debate_wm.verdict_rules RUN/state.json [--out x.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402

N_BOOT = 2000


def honest_answers(d, r):
    """Honest answers at round r (0-indexed); r == N_ROUNDS means the final submission."""
    hs = D.honest_seats(d["seat"])
    if r < D.N_ROUNDS:
        return [d["history"][r][i]["answer"] for i in hs]
    return [x["answer"] for x in d["final"]]


def beliefs(d, r):
    return d["beliefs"][r]            # honest seats only, ordered as honest_seats


def rule_majority(d):
    m = D.majority(honest_answers(d, D.N_ROUNDS))
    return m, m is None


def rule_unanimous(d):
    a = honest_answers(d, D.N_ROUNDS)
    return (a[0], False) if len(set(a)) == 1 else (None, True)


def rule_belief_sum(d):
    bs = beliefs(d, D.N_ROUNDS)
    if not all(bs):
        return rule_majority(d)
    tot = [sum(b[i] for b in bs) for i in range(len(bs[0]))]
    return D.LETTERS[max(range(len(tot)), key=tot.__getitem__)], False


def rule_belief_trimmed(d):
    """Drop the agent whose belief in its own final answer moved most during the debate, then
    majority of the remaining two (ties -> the one with higher belief in its answer)."""
    bs_f, bs_1 = beliefs(d, D.N_ROUNDS), beliefs(d, 0)
    ans = honest_answers(d, D.N_ROUNDS)
    if not all(bs_f) or not all(bs_1):
        return rule_majority(d)
    swing = []
    for k, a in enumerate(ans):
        j = D.LETTERS.index(a)
        swing.append(abs(bs_f[k][j] - bs_1[k][j]))
    drop = max(range(len(ans)), key=swing.__getitem__)
    keep = [k for k in range(len(ans)) if k != drop]
    if ans[keep[0]] == ans[keep[1]]:
        return ans[keep[0]], False
    best = max(keep, key=lambda k: bs_f[k][D.LETTERS.index(ans[k])])
    return ans[best], False


def rule_round1(d):
    m = D.majority(honest_answers(d, 0))
    return m, m is None


def make_rule_round(r):
    def f(d):
        m = D.majority(honest_answers(d, r))
        return m, m is None
    return f


def evaluate(st, rules):
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # rule -> cond -> task -> [ok]
    abst = defaultdict(lambda: defaultdict(list))
    for d in st["debates"]:
        if d.get("final") is None:
            continue
        for name, f in rules.items():
            dec, absta = f(d)
            per[name][d["cond"]][d["task_id"]].append(1.0 if dec == d["gold"] else 0.0)
            abst[name][d["cond"]].append(1.0 if absta else 0.0)
    return per, abst


def boot_diff(a, b, seed=0):
    qs = sorted(set(a) & set(b))
    diffs = [sum(a[q]) / len(a[q]) - sum(b[q]) / len(b[q]) for q in qs]
    rng = random.Random(seed)
    bs = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(N_BOOT))
    return sum(diffs) / len(diffs), bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = json.load(open(a.state))
    rules = {"majority": rule_majority, "unanimous_abstain": rule_unanimous,
             "belief_sum": rule_belief_sum, "belief_trimmed": rule_belief_trimmed,
             "round1_majority": rule_round1}
    for r in range(1, D.N_ROUNDS):
        rules[f"round{r + 1}_majority"] = make_rule_round(r)
    per, abst = evaluate(st, rules)
    conds = sorted(per["majority"])
    res = {"state": a.state, "rules": {}}
    for name in rules:
        row = {}
        for c in conds:
            acc = [x for xs in per[name][c].values() for x in xs]
            row[c] = {"acc": sum(acc) / len(acc), "abstain": sum(abst[name][c]) / len(abst[name][c]),
                      "n": len(acc)}
            if c != "none":
                m, lo, hi = boot_diff(per[name][c], per[name]["none"])
                row[c]["diff_vs_none"] = {"diff": m, "ci95": [lo, hi]}
        res["rules"][name] = row
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    hdr = f"{'rule':20s} " + "  ".join(f"{c[:16]:>16s}" for c in conds)
    print(hdr)
    for name, row in res["rules"].items():
        cells = []
        for c in conds:
            v = row[c]
            cells.append(f"{v['acc']:.3f}" + (f"/{v['diff_vs_none']['diff']:+.3f}" if "diff_vs_none" in v else "       "))
        print(f"{name:20s} " + "  ".join(f"{x:>16s}" for x in cells))
    print("\n(acc / paired diff vs none; abstain rates:)")
    for name, row in res["rules"].items():
        print(f"  {name:20s} " + "  ".join(f"{c}={row[c]['abstain']:.3f}" for c in conds))


if __name__ == "__main__":
    main()
