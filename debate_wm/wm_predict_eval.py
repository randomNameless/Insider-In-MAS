#!/usr/bin/env python
"""Next-turn prediction of the honest agents by different world models on RECORDED v1nn test debates (2026-11-05:
"why a generative interactive WM": position accuracy + message similarity). Nothing is re-played; every WM gets the
exact prefix (incl. the attacker's real messages) that the real Qwen3.5-9B agent saw, and predicts that agent's reply.

  states   exp10_nn test debates (profile v1nn, prompt debate_ppo_v3, 9B honest, T 0.7, 160 tok), seed 3, the same
           N questions for every condition (none / pi_ref / sd120 / sd120+dwm-BoN8). Targets per (debate, honest agent,
           t): t = 0,1,2 discussion rounds, t = 3 the final submission. Truth = the recorded reply.
  gen      generative WM plays the agent: engines.honest_round (argue, then letter readout, answer sampled at T 0.7)
           = byte-for-byte the WM-env call of PPO. K samples per target with ep_seed 1000+k (the recorded debates used
           seeds 3/4/5, so a 9B run is an independent re-sample = noise ceiling).
  judge    non-generative predictors of the letter only (judges.run_judge): observer_v2 (prompted), roleplay_direct.
  dcm      DCM belief WM (PPOBeliefWM, the WM-BoN wb scorer): belief at t+1 from the honest round-t replies + the 4th
           seat's round-t message. Belief input tb = recorded 9B beliefs, prior = persona prior (no hidden info).
           Predicts t = 1,2,3 only.
  rollout  multi-step: prefix = round 0 only; the WM generates the honest replies of rounds 1,2 and the final, the 4th
           seat's messages are the recorded ones (open loop on the attacker side). K samples.
  analyze  CPU tables (+ bge-m3 on GPU if available).

    python debate_wm/wm_predict_eval.py states  --out R --n 300
    python debate_wm/wm_predict_eval.py gen     --out R --tag dwm --model ~/models/debate-wm-v1 --k 4
    python debate_wm/wm_predict_eval.py judge   --out R --tag q4b --model ~/models/qwen3.5-4b --judges observer_v2
    python debate_wm/wm_predict_eval.py dcm     --out R
    python debate_wm/wm_predict_eval.py rollout --out R --tag dwm --model ~/models/debate-wm-v1 --k 4
    python debate_wm/wm_predict_eval.py analyze --out R
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402

P = D.P
R_ = Path("/path/to/runs")
CONDS = {  # name -> (run dir, cond)
    "none": ("exp10_nn_none", "none"),
    "pi_ref": ("exp10_nn_ours", "targeted_model_base"),
    "sd120": ("exp10_nn_ours", "targeted_model_rl_sd120"),
    "sd120_dwm": ("exp10_nn_dwmbon", "targeted_model_rl_sd120__wmbon8dw"),
}
HONEST_T, HONEST_TOK = 0.7, 160


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _jsonl(p):
    return [json.loads(l) for l in open(p)]


def _write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _state(d, t):
    """DebateState with the recorded history of rounds 0..t-1 (honest agents about to play t; t=3 = final)."""
    s = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    s.history = [[P.Reply(**rep) for rep in rnd] for rnd in d["history"][:t]]
    s.round = len(s.history)
    return s


def _truth(d, t, ag):
    if t < D.N_ROUNDS:
        return d["history"][t][ag]
    return d["final"][D.honest_seats(d["seat"]).index(ag)]


def cmd_states(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    targets = json.load(open(R_ / "targets_none9b_exp10_nn.json"))["targets"]
    for name, (run, cond) in CONDS.items():
        if run not in runs:
            runs[run] = json.load(open(R_ / run / "state.json"))
            assert runs[run]["config"]["profile_set"] == "v1nn", run
        deb = {d["task_id"]: d for d in runs[run]["debates"]
               if d["cond"] == cond and d["seed"] == a.seed and d["final"] is not None}
        runs.setdefault("_by", {})[name] = deb
    common = sorted(set.intersection(*(set(v) for v in runs["_by"].values())))
    pick = sorted(random.Random(0).sample(common, min(a.n, len(common))))
    rows = []
    for name in CONDS:
        for tid in pick:
            d = runs["_by"][name][tid]
            keep = {k: d[k] for k in ("task_id", "seed", "cond", "question", "n_options", "gold", "seat",
                                      "personalities", "history", "beliefs", "final")}
            keep["target"] = targets[tid]
            keep["cname"] = name
            keep["did"] = f"{name}|{tid}|s{d['seed']}"
            rows.append(keep)
    _write_jsonl(out / "debates.jsonl", rows)
    log(f"{len(rows)} debates ({len(pick)} questions x {len(CONDS)} conditions, seed {a.seed}) -> {out}")


def _targets(debs, ts):
    return [(d, t, ag) for d in debs for t in ts for ag in D.honest_seats(d["seat"])]


def _engine(a):
    from debate_ppo.engines import make_vllm_engine
    return make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len)


def cmd_gen(a):
    from debate_ppo.engines import honest_round
    D.set_profile_set("v1nn")
    out = Path(a.out)
    debs = _jsonl(out / "debates.jsonl")[: a.limit or None]
    tg = _targets(debs, [0, 1, 2, 3])
    eng = _engine(a)
    rows = []
    t0 = time.time()
    for k in range(a.k):
        for t in (0, 1, 2, 3):          # honest_round wants one phase per call (chat built from state.round)
            items = [(_state(d, tt), ag, 1000 + k) for d, tt, ag in tg if tt == t]
            meta = [(d, ag) for d, tt, ag in tg if tt == t]
            res = honest_round(eng, items, temperature=HONEST_T, max_tokens=HONEST_TOK)
            for (d, ag), r in zip(meta, res):
                rows.append({"did": d["did"], "t": t, "agent": ag, "k": k, "answer": r["reply"].answer,
                             "argument": r["reply"].argument, "probs": r["probs"], "truncated": r["truncated"]})
        log(f"gen {a.tag}: sample {k + 1}/{a.k} done ({time.time() - t0:.0f}s)")
    _write_jsonl(out / f"gen_{a.tag}.jsonl", rows)
    json.dump({"model": a.model, "k": a.k, "n_targets": len(tg), "wall_s": time.time() - t0},
              open(out / f"gen_{a.tag}.meta.json", "w"), indent=1)


def cmd_judge(a):
    from debate_ppo.wm_judge_eval.judges import run_judge
    D.set_profile_set("v1nn")
    out = Path(a.out)
    debs = _jsonl(out / "debates.jsonl")[: a.limit or None]
    tg = _targets(debs, [0, 1, 2, 3])
    recs = []
    for d, t, ag in tg:
        prev = d["history"][t - 1][ag]["answer"] if t >= 1 else None
        recs.append({"id": f"{d['did']}|{t}|{ag}", "messages": D.chat(_state(d, t), ag), "n_options": d["n_options"],
                     "meta": {"round": t + 1, "prev_answer": prev, "persona": d["personalities"][ag]}})
    eng = _engine(a)
    for j in a.judges.split(","):
        res = run_judge(eng, recs, j, temperature=HONEST_T, max_tokens=HONEST_TOK)
        rows = [{"did": d["did"], "t": t, "agent": ag, "probs": p.get("probs")} for (d, t, ag), p in zip(tg, res["preds"])]
        _write_jsonl(out / f"judge_{a.tag}_{j}.jsonl", rows)
        log(f"judge {a.tag}/{j}: {res['timing']}")


def cmd_dcm(a):
    import os
    from debate_ppo import wm_bon as WB
    D.set_profile_set("v1nn")
    out = Path(a.out)
    debs = _jsonl(out / "debates.jsonl")[: a.limit or None]
    sel = WB.Selector("tb")
    rows = []
    t0 = time.time()
    for n, d in enumerate(debs):
        seat, hs = d["seat"], D.honest_seats(d["seat"])
        for r in (0, 1, 2):             # predicts t = r+1 from the replies of round r
            for mode in a.modes.split(","):
                honest = {}
                for k, i in enumerate(hs):
                    rep = d["history"][r][i]
                    pr = WB._prior(d["personalities"][i], rep["answer"], d["n_options"])
                    h = {"persona": d["personalities"][i], "message": rep["raw"], "answer": rep["answer"],
                         "probs": WB._pad(d["beliefs"][r][k]) if mode == "tb" else pr}
                    if r >= 1:
                        pv = d["history"][r - 1][i]
                        h.update({"prev_message": pv["raw"], "prev_answer": pv["answer"],
                                  "prev_probs": WB._pad(d["beliefs"][r - 1][k]) if mode == "tb"
                                  else WB._prior(d["personalities"][i], pv["answer"], d["n_options"])})
                    honest[i] = h
                st = {"question_id": d["task_id"], "round": r + 1, "fourth_seat": seat, "gold": d["gold"],
                      "target": d["target"], "honest": honest,
                      "fourth_prev_message": d["history"][r - 1][seat]["raw"] if r >= 1 else None}
                o = sel.wm.score(st, [d["history"][r][seat]["raw"]])[0]
                for i in hs:
                    rows.append({"did": d["did"], "t": r + 1, "agent": i, "mode": mode,
                                 "probs": [float(x) for x in o["beliefs"][i]][: d["n_options"]]})
        if n % 100 == 0:
            log(f"dcm {n}/{len(debs)} ({time.time() - t0:.0f}s)")
    _write_jsonl(out / "dcm.jsonl", rows)


def cmd_rollout(a):
    from debate_ppo.engines import honest_round
    D.set_profile_set("v1nn")
    out = Path(a.out)
    debs = _jsonl(out / "debates.jsonl")[: a.limit or None]
    eng = _engine(a)
    rows = []
    t0 = time.time()
    for k in range(a.k):
        sts = {d["did"]: _state(d, 1) for d in debs}            # real round 0 only
        for t in (1, 2, 3):
            items = [(sts[d["did"]], ag, 2000 + k) for d in debs for ag in D.honest_seats(d["seat"])]
            meta = [(d, ag) for d in debs for ag in D.honest_seats(d["seat"])]
            res = honest_round(eng, items, temperature=HONEST_T, max_tokens=HONEST_TOK)
            reps = collections.defaultdict(dict)
            for (d, ag), r in zip(meta, res):
                reps[d["did"]][ag] = r["reply"]
                rows.append({"did": d["did"], "t": t, "agent": ag, "k": k, "answer": r["reply"].answer,
                             "argument": r["reply"].argument})
            if t < D.N_ROUNDS:
                for d in debs:              # the 4th seat says what it really said in round t
                    reps[d["did"]][d["seat"]] = P.Reply(**d["history"][t][d["seat"]])
                    D.commit(sts[d["did"]], reps[d["did"]])
        log(f"rollout {a.tag}: sample {k + 1}/{a.k} ({time.time() - t0:.0f}s)")
    _write_jsonl(out / f"rollout_{a.tag}.jsonl", rows)
    json.dump({"model": a.model, "k": a.k, "wall_s": time.time() - t0}, open(out / f"rollout_{a.tag}.meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["states", "gen", "judge", "dcm", "rollout", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag")
    ap.add_argument("--model")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--judges", default="observer_v2,roleplay_direct")
    ap.add_argument("--modes", default="tb,prior")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N debates only")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    if a.cmd == "analyze":
        from debate_wm import wm_predict_analyze as AN
        AN.main(Path(a.out))
        return
    {"states": cmd_states, "gen": cmd_gen, "judge": cmd_judge, "dcm": cmd_dcm, "rollout": cmd_rollout}[a.cmd](a)


if __name__ == "__main__":
    main()
