#!/usr/bin/env python
"""Turn debate trajectories (debate_ppo/eval_debate.py state.json, honest agents = teacher) into
SFT / distillation records for a debate world model.

One record per HONEST turn (rounds 1-3 + final submission, agents = the three honest seats):
    messages : [system(persona), user(round prompt)]  -- exactly what the teacher saw
    target   : teacher output, normalised to the readout format:
               "<argument text as generated>\\nAnswer: <sampled letter>"
    probs    : teacher option distribution read out after its own argument (for the KL loss)
    meta     : task_id, gold, persona, round, cond, seed, seat (gold is metadata only)

The 4th agent's turns (attacker, or neutral agent in `none`) are context only, never targets.

    python -m debate_wm.build_sft --states RUN1/state.json RUN2/state.json --out data/fit.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402

P = D.P


def target_text(raw: str, answer: str) -> str:
    """raw = generated argument text (stopped at 'Answer:') + 'Answer: X'. Rebuild it so the text
    before 'Answer:' is byte-identical to the readout prompt (newline-terminated argument)."""
    arg = raw[: raw.rfind("Answer: ")] if "Answer: " in raw else raw
    if arg and not arg.endswith("\n"):
        arg = arg + "\n"
    return f"{arg}Answer: {answer}"


def records_from_debate(d: dict):
    hs = D.honest_seats(d["seat"])
    st = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    rounds = [d["history"][r] for r in range(len(d["history"]))]
    for r in range(D.N_ROUNDS + 1):
        st.history = [[P.Reply(**rep) for rep in rnd] for rnd in rounds[:r]]
        st.round = r
        if r < D.N_ROUNDS:
            reps = {i: rounds[r][i] for i in hs}
        else:
            reps = dict(zip(hs, d["final"]))
        probs = d["beliefs"][r]                      # honest seats only, ordered as hs
        for k, a in enumerate(hs):
            rep = reps[a]
            yield {
                "messages": D.chat(st, a),
                "target": target_text(rep["raw"], rep["answer"]),
                "answer": rep["answer"],
                "probs": probs[k],
                "n_options": d["n_options"],
                "meta": {"task_id": d["task_id"], "gold": d["gold"], "persona": d["personalities"][a],
                         "round": r + 1, "cond": d["cond"], "seed": d["seed"], "seat": d["seat"],
                         "agent": a},
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for sp in a.states:
            st = json.load(open(sp))
            D.set_profile_set(st["config"].get("profile_set", "v1"))
            for d in st["debates"]:
                if d.get("final") is None:
                    continue
                for rec in records_from_debate(d):
                    f.write(json.dumps(rec) + "\n")
                    n += 1
    print(f"{n} records -> {a.out}")


if __name__ == "__main__":
    main()
