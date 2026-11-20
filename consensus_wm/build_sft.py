#!/usr/bin/env python
"""Turn collect_9b.py records into SFT records for one arm.

arm A ("persona"): base role prompt + "You are Agent i." + the true persona suffix.
arm B/C ("id"):    base role prompt + "You are Agent i." only; B is built from fixed-seat games,
                   C from shuffled-seat games (the data file decides fixed vs shuffled).
All arms prepend the PUBLIC history of the game (every agent's round-start positions and
broadcasts in earlier rounds) to the user prompt. The 9B agents are memoryless across rounds,
but an observer of the MAS sees this history, and it is the only place an ID-only WM can infer
an agent's type from (its own past moves/messages).
Target = the 9B's raw reply, normalised to "Message: <m>\\nPosition: <k>"; invalid replies dropped.

    python -m consensus_wm.build_sft --mode id --inp fixed_train.jsonl --out B_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import defaultdict  # noqa: E402

from attacker.prompts import BASE_ROLE_PROMPT, SUFFIX_BY_PERSONALITY  # noqa: E402


def wm_system(agent: int, persona: str | None) -> str:
    s = BASE_ROLE_PROMPT + "\n" + f"You are Agent {agent}."
    return s if persona is None else s + "\n" + SUFFIX_BY_PERSONALITY[persona]


def history_block(rounds) -> str:
    """rounds: list of (positions[4], messages[4]) for earlier rounds, oldest first."""
    if not rounds:
        return ""
    lines = ["Public history of this game (positions at the start of each earlier round, "
             "then what each agent broadcast in that round):"]
    for s, (pos, msgs) in enumerate(rounds):
        said = "; ".join(f'Agent {j}: "{m}"' for j, m in enumerate(msgs))
        lines.append(f"Round {s + 1}: positions {list(pos)}. {said}")
    return "\n".join(lines) + "\n\n"


def game_rounds(recs):
    """Per-gid list of (round-start positions, broadcasts that round) from benign records."""
    by_t = {}
    for r in recs:
        by_t.setdefault(r["t"], r)
    T = max(by_t)
    rounds = []
    for t in range(T):  # broadcasts of round t are the last_messages seen at t+1
        rounds.append((by_t[t]["positions"], by_t[t + 1]["last_messages"]))
    return rounds


def to_sft(rec, mode, rounds):
    system = wm_system(rec["agent"], rec["persona"] if mode == "persona" else None)
    user = history_block(rounds[:rec["t"]]) + rec["user"]
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "target": f"Message: {rec['message']}\nPosition: {rec['next_position']}",
            "meta": {k: rec[k] for k in ("seat_mode", "gid", "t", "agent", "persona", "attacker")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["persona", "id"], required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_gid = defaultdict(list)
    for line in open(a.inp):
        rec = json.loads(line)
        by_gid[rec["gid"]].append(rec)
    n = kept = 0
    with open(a.out, "w") as fo:
        for gid in sorted(by_gid):
            recs = by_gid[gid]
            rounds = game_rounds(recs)
            for rec in recs:
                n += 1
                if not rec["valid"]:
                    continue
                assert rec["system"].startswith(BASE_ROLE_PROMPT)
                fo.write(json.dumps(to_sft(rec, a.mode, rounds)) + "\n")
                kept += 1
    print(f"{kept}/{n} records -> {a.out}")


if __name__ == "__main__":
    main()
