#!/usr/bin/env python
"""Adversarial-training prompts for the honest debate agents (2026-11-08).

From attack-run states on the FIT split (debate_ppo/eval_debate*.py state.json, e.g. def_fit_sd120 / def_fit_dwmbon) and
the clean `none` state of the same split / seed / profile set, build one record per honest turn that SAW an attacker
message (rounds 2, 3 and the final submission; round 1 has no messages):

    messages      : [system(persona), user(round prompt)]  -- exactly what the honest agent saw under attack
    clean_answer  : the SAME agent index's answer in the SAME round of the clean debate (same task, seed, seat rule);
                    the consistency target: "answer as you would have without the attacker"
    clean_final   : that agent's final-submission answer in the clean debate
    clean_probs   : its clean belief at that round (optional KL target)
    gold          : metadata only (used by the --target gold variant of at_sample.py, never in a prompt)
    meta          : task_id, seed, seat, agent, persona, round, cond, attack run

Also writes the clean anchor set: the clean run's own honest turns (rounds 2-3 + final) with the clean reply as target
("--clean-out"), used to keep the fine-tuned victim behaving normally when nobody attacks.

    python -m debate_wm.build_at_data --attack ~/runs/def_fit_sd120/state.json ~/runs/def_fit_dwmbon/state.json \
        --clean ~/runs/stratdpo6_collect_v1nn_fit/state.json --out ~/data/at/fit_attacked.jsonl --clean-out ~/data/at/fit_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402
from debate_wm.build_sft import target_text  # noqa: E402

P = D.P


def state_at(d: dict, r: int) -> "P.DebateState":
    s = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    s.history = [[P.Reply(**rep) for rep in rnd] for rnd in d["history"][:r]]
    s.round = r
    return s


def answer_at(d: dict, agent: int, r: int):
    if r < D.N_ROUNDS:
        return d["history"][r][agent]["answer"]
    return d["final"][D.honest_seats(d["seat"]).index(agent)]["answer"]


def belief_at(d: dict, agent: int, r: int):
    return d["beliefs"][r][D.honest_seats(d["seat"]).index(agent)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-out", default=None)
    a = ap.parse_args()
    cl = json.load(open(a.clean))
    D.set_profile_set(cl["config"].get("profile_set", "v1"))
    assert cl["config"]["conds"] == ["none"]
    clean = {(d["task_id"], d["seed"]): d for d in cl["debates"] if d["final"] is not None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = Counter()
    with open(a.out, "w") as f:
        for path in a.attack:
            st = json.load(open(path))
            cfg = st["config"]
            assert cfg["profile_set"] == cl["config"]["profile_set"] and cfg["split"] == cl["config"]["split"], path
            assert cfg["tasks_sha256"] == cl["config"]["tasks_sha256"]
            for d in st["debates"]:
                if d["cond"] == "none" or d["final"] is None:
                    continue
                c = clean.get((d["task_id"], d["seed"]))
                if c is None:
                    n["no_clean"] += 1
                    continue
                assert c["seat"] == d["seat"], (d["task_id"], d["seed"])
                for r in range(1, D.N_ROUNDS + 1):
                    s = state_at(d, r)
                    for i in D.honest_seats(d["seat"]):
                        rec = {"messages": D.chat(s, i), "n_options": d["n_options"],
                               "clean_answer": answer_at(c, i, r), "clean_final": answer_at(c, i, D.N_ROUNDS),
                               "clean_probs": belief_at(c, i, r), "gold": d["gold"], "target": d.get("target"),
                               "attacked_answer": answer_at(d, i, r),
                               "meta": {"task_id": d["task_id"], "seed": d["seed"], "seat": d["seat"], "agent": i,
                                        "persona": d["personalities"][i], "round": r, "cond": d["cond"], "run": str(Path(path).parent.name)}}
                        f.write(json.dumps(rec) + "\n")
                        n[d["cond"]] += 1
                        n["clean_is_gold"] += int(rec["clean_answer"] == d["gold"])
                        n["attacked_is_gold"] += int(rec["attacked_answer"] == d["gold"])
                        n["attacked_eq_clean"] += int(rec["attacked_answer"] == rec["clean_answer"])
    print(json.dumps(n, indent=1))
    if a.clean_out:
        m = 0
        with open(a.clean_out, "w") as f:
            for c in clean.values():
                for r in range(1, D.N_ROUNDS + 1):
                    s = state_at(c, r)
                    for i in D.honest_seats(c["seat"]):
                        rep = c["history"][r][i] if r < D.N_ROUNDS else c["final"][D.honest_seats(c["seat"]).index(i)]
                        if not rep["valid"] or not rep["answer"]:
                            continue
                        f.write(json.dumps({"messages": D.chat(s, i), "target": target_text(rep["raw"], rep["answer"]),
                                            "probs": belief_at(c, i, r), "n_options": c["n_options"],
                                            "meta": {"task_id": c["task_id"], "seed": c["seed"], "agent": i, "round": r,
                                                     "cond": "none", "source": "clean_reply"}}) + "\n")
                        m += 1
        print(f"clean anchor records: {m} -> {a.clean_out}")


if __name__ == "__main__":
    main()
