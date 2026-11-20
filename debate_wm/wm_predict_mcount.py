#!/usr/bin/env python
"""M_count ("Latent Belief MLP": dcm_analysis/models.MCount, a vote-only conditional-logit MLP over opinion scalars —
own belief / one-hot answer, peer vote counts, their histories, round, K) re-fit on the v1nn PPO-protocol 9B debates and
evaluated as a next-turn predictor on the wm_predict_eval set (2026-11-05). CPU only.

  fit  : stratdpo6_collect_v1nn_{fit,val} `none` debates (Qwen3.5-9B honest, 4th seat = neutral honest agent; the same
         population the attacker's WM was distilled from). Two feature modes, as in transitions.expand:
           belief : own = the listener's recorded belief b_t (hidden information, like DCM `tb`)
           answer : own = one-hot of its stated answer (observable only, deployable)
  eval : every honest (debate, t>=1, agent) target of R/debates.jsonl -> R/mcount.jsonl rows {did,t,agent,mode,probs},
         read by wm_predict_analyze like dcm.jsonl.

    python debate_wm/wm_predict_mcount.py --out ~/runs/wmpred_v1nn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LATENT = Path("/path/to/latent_agent_transfer/dcm_analysis")
sys.path.insert(0, str(LATENT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from fit import fit_mcount  # noqa: E402  (dcm_analysis)
from models import Batch, MCount, xent  # noqa: E402

LET = "ABCDEFGHIJ"
R_ = Path("/path/to/runs")


def _onehot(a, K):
    v = [0.0] * K
    if a in LET[:K]:
        v[LET.index(a)] = 1.0
    return v


def samples(d, mode, seat_key="seat", with_final=True):
    """One sample per honest agent and transition t -> t+1 (t = 0,1,2; t=2 -> final when with_final)."""
    K = d["n_options"]
    seat = d[seat_key]
    hs = [i for i in range(4) if i != seat]
    H = d["history"]
    out = []
    for i in hs:
        hist_own, hist_n = [], []
        for t in range(len(H)):
            a_t = H[t][i]["answer"]
            n = [0.0] * K
            for j in range(4):
                if j != i and H[t][j]["answer"] in LET[:K]:
                    n[LET.index(H[t][j]["answer"])] += 1.0
            bel = d["beliefs"][t][hs.index(i)][:K] if d.get("beliefs") and len(d["beliefs"]) > t else None
            own = list(bel) if (mode == "belief" and bel is not None) else _onehot(a_t, K)
            hist_own.append(own); hist_n.append(n)
            if t + 1 < len(H):
                a_next = H[t + 1][i]["answer"]
            elif with_final and d.get("final"):
                a_next = d["final"][hs.index(i)]["answer"]
            else:
                continue
            if a_t not in LET[:K] or a_next not in LET[:K]:
                continue
            out.append({"own": own, "n": n, "hist_own": [list(x) for x in hist_own], "hist_n": [list(x) for x in hist_n],
                        "target": LET.index(a_next), "prev": LET.index(a_t), "K": K, "t": t, "n_visible": 3,
                        "meta": {"persona": d["personalities"][i], "did": d.get("did"), "agent": i, "t_next": t + 1}})
    return out


def load_run(name):
    s = json.load(open(R_ / name / "state.json"))
    assert s["config"]["profile_set"] == "v1nn"
    return [d for d in s["debates"] if d["cond"] == "none" and d["final"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="belief,answer")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    fit, val = load_run("stratdpo6_collect_v1nn_fit"), load_run("stratdpo6_collect_v1nn_val")
    ev = [json.loads(l) for l in open(out / "debates.jsonl")]
    rows, info = [], {}
    for mode in a.modes.split(","):
        tr = [s for d in fit for s in samples(d, mode)]
        va = [s for d in val for s in samples(d, mode)]
        bt, bv = Batch(tr), Batch(va)
        t0 = time.time()
        models = []
        for seed in range(a.seeds):
            m, inf = fit_mcount(bt, bv, seed=seed)
            models.append(m)
            info[f"{mode}/seed{seed}"] = inf
        print(f"{mode}: {len(tr)} train / {len(va)} val samples, fit {time.time() - t0:.0f}s, "
              f"val nll {[round(info[f'{mode}/seed{s}']['best_val_nll'], 4) for s in range(a.seeds)]}", flush=True)
        es = [s for d in ev for s in samples(d, mode)]
        be = Batch(es)
        with torch.no_grad():
            lp = torch.stack([m(be) for m in models]).logsumexp(0) - torch.log(torch.tensor(float(len(models))))
            info[f"{mode}/eval_nll"] = xent(lp, be).mean().item()
            info[f"{mode}/eval_acc"] = (lp.argmax(1) == be.target).double().mean().item()
        for s, l in zip(es, lp):
            p = l[: s["K"]].exp().tolist()
            rows.append({"did": s["meta"]["did"], "t": s["meta"]["t_next"], "agent": s["meta"]["agent"], "mode": mode,
                         "probs": [x / sum(p) for x in p]})
        print(f"{mode}: eval {len(es)} samples, nll {info[f'{mode}/eval_nll']:.4f}, top1 {info[f'{mode}/eval_acc']:.3f}",
              flush=True)
    with open(out / "mcount.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"train": "stratdpo6_collect_v1nn_fit none", "val": "stratdpo6_collect_v1nn_val none",
               "seeds": a.seeds, "info": info}, open(out / "mcount.meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
