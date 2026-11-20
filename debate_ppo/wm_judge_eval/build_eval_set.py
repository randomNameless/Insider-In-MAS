#!/usr/bin/env python
"""Held-out judge-evaluation set: honest Qwen3.5-9B turns from TEST-split debate logs.

Sources are eval_debate.py state.json files whose honest agents are the real Qwen3.5-9B (Lambda runs,
backed up locally). One record per honest turn (3 honest seats x rounds 1-3 + final submission):

    id, messages (the exact chat the 9B saw: persona system prompt + round prompt), n_options,
    gt: {answer (the letter 9B actually sampled at T=0.7), probs (its recorded P(A..J) after its own
         argument), top1, flip (answer != own previous-round answer; None in round 1)},
    meta: task_id, gold, target (assigned wrong option, targeted conds only), persona, round (1..4),
          cond, seed, seat (4th agent's seat), agent, prev_answer, fourth_prev (4th agent's answer the
          honest agent reads in this round), others_prev, source run, attacker_seen_in_wm_training.

Debates are sampled per condition with a fixed RNG so the set is reproducible; the manifest records
the source files' sha256. Only the test split (400 questions) is used; the WM was fitted on `fit`
and selected on `val`.

    python -m debate_ppo.wm_judge_eval.build_eval_set --out runs_local/wm_judge_eval/eval_turns.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo import dprotocol as D  # noqa: E402

P = D.P

# run dir (relative to repo) -> conditions to sample. All: test split, honest = Qwen3.5-9B, profile v1.
# "seen": the 4th-agent type appeared in the WM's SFT data (wm_collect_v1_fit: none / prompt /
# model_base / model_rl_ckpt20, all Qwen3-4B untargeted); everything else is unseen by the WM.
DEFAULT_SOURCES = {
    "artifacts/runs/exp9_v5_test_9b": ["none", "prompt", "model_base", "model_rl_oldwm_ckpt20"],
    "artifacts/runs/exp9_v6_targeted_test_9b": ["targeted_4b", "targeted_model_rl_c040",
                                                               "targeted_model_rl_c120"],
    "runs_lambda/exp9_ju_test_9b": ["ju_dpo_q3_4b"],
    "runs_lambda/exp9_strat_test_9b": ["stratdpo_q3_4b"],
    "artifacts/runs/exp9_upper_bound_9b": ["prompt_9b", "gold_9b"],
}
SEEN_CONDS = {"none", "prompt", "model_base", "model_rl_oldwm_ckpt20"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def turn_records(d: dict, source: str):
    """All honest turns of one finished debate, with the exact chat each honest agent saw."""
    hs = D.honest_seats(d["seat"])
    st = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    rounds = d["history"]
    for r in range(D.N_ROUNDS + 1):
        st.history = [[P.Reply(**rep) for rep in rnd] for rnd in rounds[:r]]
        st.round = r
        reps = {i: rounds[r][i] for i in hs} if r < D.N_ROUNDS else dict(zip(hs, d["final"]))
        probs_r = d["beliefs"][r]
        for k, a in enumerate(hs):
            rep = reps[a]
            probs = probs_r[k]
            prev = rounds[r - 1][a]["answer"] if r > 0 else None
            others = {j: rounds[r - 1][j]["answer"] for j in range(D.N_AGENTS) if j != a} if r > 0 else {}
            yield {
                "id": f"{d['cond']}|{d['task_id']}|s{d['seed']}|r{r + 1}|a{a}",
                "messages": D.chat(st, a),
                "n_options": d["n_options"],
                "gt": {"answer": rep["answer"], "probs": probs,
                       "top1": D.LETTERS[max(range(len(probs)), key=probs.__getitem__)],
                       "flip": (rep["answer"] != prev) if r > 0 else None},
                "meta": {"task_id": d["task_id"], "gold": d["gold"], "target": d.get("target"),
                         "persona": d["personalities"][a], "round": r + 1, "cond": d["cond"],
                         "seed": d["seed"], "seat": d["seat"], "agent": a, "prev_answer": prev,
                         "fourth_prev": others.get(d["seat"]), "others_prev": others,
                         "source": source, "seen": d["cond"] in SEEN_CONDS},
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cond", type=int, default=60, help="debates sampled per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=None, help="JSON {run_dir: [conds]} (default: DEFAULT_SOURCES)")
    a = ap.parse_args()
    sources = json.loads(a.sources) if a.sources else DEFAULT_SOURCES
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "per_cond": a.per_cond, "sources": {}, "counts": {}}
    n = 0
    with open(out, "w") as f:
        for run, conds in sources.items():
            sp = REPO / run / "state.json"
            st = json.load(open(sp))
            cfg = st["config"]
            assert cfg.get("split") == "test", (run, cfg.get("split"))
            assert "qwen3.5-9b" in cfg["victim"], (run, cfg["victim"])
            D.set_profile_set(cfg.get("profile_set", "v1"))
            manifest["sources"][run] = {"sha256": sha256_file(sp), "split": cfg["split"], "victim": cfg["victim"],
                                        "seeds": cfg["seeds"], "conds": conds, "honest_temp": cfg["honest_temp"],
                                        "honest_arg_tokens": cfg["honest_arg_tokens"],
                                        "split_digest": cfg.get("split_digest")}
            for c in conds:
                pool = [d for d in st["debates"] if d["cond"] == c and d.get("final")]
                assert pool, (run, c)
                pool.sort(key=lambda d: (d["task_id"], d["seed"]))
                pick = rng.sample(pool, min(a.per_cond, len(pool)))
                k = 0
                for d in pick:
                    for rec in turn_records(d, run):
                        f.write(json.dumps(rec) + "\n")
                        k += 1
                manifest["counts"][c] = {"debates": len(pick), "turns": k}
                n += k
    manifest["n_turns"] = n
    json.dump(manifest, open(out.with_suffix(".manifest.json"), "w"), indent=1)
    print(f"{n} turns -> {out}")
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()
