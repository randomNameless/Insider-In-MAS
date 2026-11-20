#!/usr/bin/env python
"""debate-wm-v1 (and prompted judges) as Best-of-N RANKERS on the DCM belief WM's own offline test
(latent_agent_transfer/dcm_analysis summary 19.44 step 4b): 100 test debates of the strongest targeted persuader
(exp9-v6sd-tgt-c120, cond targeted_model_rl_sd120 of runs_lambda/exp9_v6sd_ours_test_9b) x rounds 1-2 = 200 states,
8 persuader candidates per state (data/ppo_bon/lambda_sd120/lambda_cands.jsonl), oracle = Qwen3.5-9B readout of each honest
listener's next-round P(target) with the candidate inserted ('pre' = prefill, 'post' = after a sampled argument;
lambda_oracle.jsonl), and the DCM scores WM_post / WM_pre already in rows_lambda.json.

For each candidate the vLLM judge gets the three honest listeners' NEXT-round chat with the candidate inserted as the 4th
seat's round-r reply (dprotocol.chat on the recorded state, exactly what the oracle saw) and the score is the mean
predicted P(target). Same metrics as mmlupro/ppo_lambda_bon.py: fraction of the oracle gain over a random pick, top-25%
precision, per-state Spearman, mean P(target) of the pick; plus a paired bootstrap over states.

    stage 1 (GPU, insider env): --model PATH --tag wm --judges roleplay_gen,roleplay_direct
    stage 2 (CPU): --metrics
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.wm_judge_eval import metrics as M  # noqa: E402

P = D.P
LATENT = Path("/path/to/latent_agent_transfer/dcm_analysis/data/ppo_bon/lambda_sd120")
STATE = REPO / "runs_lambda/exp9_v6sd_ours_test_9b/state.json"
COND = "targeted_model_rl_sd120"
OUT = REPO / "runs_local/wm_judge_eval/rank_bon"


def load_states():
    st = json.load(open(STATE))
    deb = {(d["task_id"], d["seed"]): d for d in st["debates"] if d["cond"] == COND}
    cands = {json.loads(l)["sid"]: json.loads(l)["cands"] for l in open(LATENT / "lambda_cands.jsonl")}
    states = []
    for sid, cs in cands.items():
        t, sd, r = sid.split("|")
        d = deb[(t, int(sd[1:]))]
        r = int(r[1:])
        states.append({"sid": sid, "d": d, "round": r, "cands": cs})
    return states


def records_for(states):
    """One pseudo eval record per (state, candidate, honest listener): the listener's round-(r+1) chat."""
    recs = []
    for s in states:
        d, r = s["d"], s["round"]
        seat = d["seat"]
        hs = D.honest_seats(seat)
        for c, cand in enumerate(s["cands"]):
            st = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
            hist = [[P.Reply(**{k: x[k] for k in ("answer", "argument", "raw", "valid")}) for x in rnd] for rnd in d["history"][:r]]
            hist[r - 1][seat] = P.Reply(answer=cand["answer"], argument=cand["argument"] or P.FALLBACK_ARGUMENT,
                                        raw=cand["text"], valid=bool(cand["valid"]))
            st.history = hist
            st.round = r
            for i in hs:
                recs.append({"id": f"{s['sid']}|c{c}|a{i}", "messages": D.chat(st, i), "n_options": d["n_options"],
                             "meta": {"round": r + 1, "prev_answer": d["history"][r - 1][i]["answer"], "target": d["target"],
                                      "persona": d["personalities"][i]}})
    return recs


def stage_gpu(a):
    from debate_ppo.engines import make_vllm_engine
    from debate_ppo.wm_judge_eval.judges import run_judge
    states = load_states()
    recs = records_for(states)
    print(f"{len(states)} states, {len(recs)} listener turns", flush=True)
    eng = make_vllm_engine(a.model, gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=128)
    OUT.mkdir(parents=True, exist_ok=True)
    for judge in a.judges.split(","):
        res = run_judge(eng, recs, judge)
        with open(OUT / f"{a.tag}__{judge}.jsonl", "w") as f:
            for r, p in zip(recs, res["preds"]):
                f.write(json.dumps({"id": r["id"], "probs": p["probs"], "p_target": p["probs"][D.LETTERS.index(r["meta"]["target"])]}) + "\n")
        json.dump({"model": a.model, "judge": judge, "timing": res["timing"]}, open(OUT / f"{a.tag}__{judge}.timing.json", "w"), indent=1)
        print(judge, json.dumps(res["timing"]), flush=True)


def rank_metrics(rows_by_sid, score_key, okey, rng_seed=0):
    picks, orcs, rnds, sps, t25 = [], [], [], [], []
    for sid in sorted(rows_by_sid):
        rs = [r for r in rows_by_sid[sid] if r["valid"]]
        if len(rs) < 2:
            continue
        y = [r[okey] for r in rs]
        if score_key == "random":
            sc = [random.Random(hash(sid) % 2**32 + rng_seed).random() for _ in rs]
        elif score_key == "first_sample":
            sc = [1.0] + [0.0] * (len(rs) - 1)
        else:
            sc = [r[score_key] for r in rs]
        k = max(range(len(rs)), key=sc.__getitem__)
        picks.append(y[k]); orcs.append(max(y)); rnds.append(sum(y) / len(y)); sps.append(M.spearman(sc, y))
        t25.append(sum(v > y[k] for v in y) < max(1, len(y) // 4))
    return picks, orcs, rnds, sps, t25


def stage_metrics(a):
    rows = json.load(open(LATENT / "rows_lambda.json"))
    for r in rows:
        r["valid"] = r["valid"] in (True, "True"); r["c"] = int(r["c"])
        for k in list(r):
            if k.endswith("ptarget") or k.endswith("pgold"):
                r[k] = float(r[k])
    scores = {}
    for f in sorted(OUT.glob("*__*.jsonl")):
        name = f.stem
        per = {}
        for l in open(f):
            p = json.loads(l)
            sid, c, _ = p["id"].rsplit("|", 2)
            per.setdefault((sid, int(c[1:])), []).append(p["p_target"])
        scores[name] = {k: sum(v) / len(v) for k, v in per.items()}
    for r in rows:
        for name, sc in scores.items():
            r[name + "_ptarget"] = sc[(r["sid"], r["c"])]
    by = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r)
    rankers = ["WM_post", "WM_pre"] + list(scores) + ["random", "first_sample"]
    res = {"n_states": len(by), "n_cand": 8, "rankers": {}}
    sids = sorted(by)
    for okind in ("pre", "post"):
        okey = f"oracle_{okind}_ptarget"
        res["rankers"][okind] = {}
        for rk in rankers:
            key = rk + "_ptarget" if rk not in ("random", "first_sample") else rk
            picks, orcs, rnds, sps, t25 = rank_metrics(by, key, okey)
            g = sum(o - r for o, r in zip(orcs, rnds))
            # paired bootstrap over states
            rng = random.Random(0); fb, db = [], []
            for _ in range(a.n_boot):
                idx = [rng.randrange(len(picks)) for _ in picks]
                gg = sum(orcs[i] - rnds[i] for i in idx)
                fb.append(sum(picks[i] - rnds[i] for i in idx) / gg if gg > 0 else float("nan"))
                db.append(sum(picks[i] - rnds[i] for i in idx) / len(idx))
            fb.sort(); db.sort()
            res["rankers"][okind][rk] = {
                "n_states": len(picks), "frac_of_oracle_gain": sum(p - r for p, r in zip(picks, rnds)) / g,
                "frac_ci95": [fb[int(0.025 * len(fb))], fb[int(0.975 * len(fb)) - 1]],
                "pick_minus_random_pp": 100 * sum(p - r for p, r in zip(picks, rnds)) / len(picks),
                "pick_minus_random_ci95_pp": [100 * db[int(0.025 * len(db))], 100 * db[int(0.975 * len(db)) - 1]],
                "top25_precision": sum(t25) / len(t25), "spearman": M.mean(x for x in sps if x == x),
                "mean_ptarget_pick": M.mean(picks), "mean_ptarget_oracle": M.mean(orcs), "mean_ptarget_random": M.mean(rnds)}
    res["cost"] = {f.stem: json.load(open(f))["timing"] for f in OUT.glob("*.timing.json")}
    json.dump(res, open(OUT / "rank_results.json", "w"), indent=1)
    lines = ["| ranker | oracle | frac of oracle gain [95% CI] | pick − random (pp P(target)) [CI] | top-25% prec | Spearman | P(target) pick / oracle / random |", "|---|---|---|---|---|---|---|"]
    for okind in ("pre", "post"):
        for rk, v in res["rankers"][okind].items():
            lines.append(f"| {rk} | {okind} | {v['frac_of_oracle_gain']:.3f} [{v['frac_ci95'][0]:.3f}, {v['frac_ci95'][1]:.3f}] | "
                         f"{v['pick_minus_random_pp']:+.1f} [{v['pick_minus_random_ci95_pp'][0]:+.1f}, {v['pick_minus_random_ci95_pp'][1]:+.1f}] | "
                         f"{v['top25_precision']:.3f} | {v['spearman']:.3f} | {v['mean_ptarget_pick']:.3f} / {v['mean_ptarget_oracle']:.3f} / {v['mean_ptarget_random']:.3f} |")
    open(OUT / "rank_results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--judges", default="roleplay_gen,roleplay_direct")
    ap.add_argument("--metrics", action="store_true"); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.metrics:
        stage_metrics(a)
    else:
        assert a.model and a.tag
        stage_gpu(a)


if __name__ == "__main__":
    main()
