#!/usr/bin/env python
"""Paired tables for the debate defenses (2026-11-08).

Reference (no defense, v1nn, PPO protocol, test 400 x seeds 3/4/5): exp10_nn_none / exp10_nn_ours[targeted_model_rl_sd120]
/ exp10_nn_dwmbon[targeted_model_rl_sd120__wmbon8dw]. A defended set <tag>_<def>_{none,sd120,dwmbon} shares task ids,
seeds, seats and honest seeds, so every debate pairs by (task, seed). Reported per defense, with per-question
bootstrap 95% CIs (10k resamples, seed 0):
  * final_acc under none / sd120 / sd120+dwm, and the attack Δ vs the defended none;
  * defended − undefended under each condition (clean cost under none, protection under attack);
  * difference-in-differences: (Δ_attack defended) − (Δ_attack undefended)  (positive = attack weaker);
  * group accuracy (majority == gold) and the defense diagnostics (attacker hidden rate).

    python debate_wm/def_paired.py --tag def            # filters: def_{wmsr,cpwbft,sac}_*
    python debate_wm/def_paired.py --tag at --defs cons,gold   # adversarially trained victims: at_{cons,gold}_*
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.eval_debate import debate_metrics  # noqa: E402

R = Path("/path/to/runs")
REF = {"none": ("exp10_nn_none", "none"), "sd120": ("exp10_nn_ours", "targeted_model_rl_sd120"),
       "sd120+dwm": ("exp10_nn_dwmbon", "targeted_model_rl_sd120__wmbon8dw")}
COND = {"none": "none", "sd120": "targeted_model_rl_sd120", "sd120+dwm": "targeted_model_rl_sd120__wmbon8dw"}
DIRSUF = {"none": "none", "sd120": "sd120", "sd120+dwm": "dwmbon"}
N_BOOT = 10000


def load(run: str, cond: str):
    """(task, seed) -> metrics for one condition of a run; None if the run is missing / unfinished."""
    p = R / run / "state.json"
    if not p.exists():
        return None
    st = json.load(open(p))
    D.set_profile_set(st["config"].get("profile_set", "v1"))
    out = {}
    for d in st["debates"]:
        if d["cond"] != cond or d["final"] is None:
            continue
        m = debate_metrics(d)
        m["group_acc"] = float(D.majority([x["answer"] for x in d["final"]]) == d["gold"])
        out[(d["task_id"], d["seed"])] = m
    return out or None


def per_q(m: dict, key: str):
    by = defaultdict(list)
    for (t, s), x in m.items():
        if key in x:
            by[t].append(x[key])
    return {t: sum(v) / len(v) for t, v in by.items()}


def boot(vals, seed=0):
    n = len(vals)
    if n == 0:
        return None
    rng = random.Random(seed)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return {"mean": sum(vals) / n, "lo": b[int(.025 * N_BOOT)], "hi": b[int(.975 * N_BOOT) - 1], "n": n}


def diff(a: dict, b: dict):
    qs = sorted(set(a) & set(b))
    return boot([a[q] - b[q] for q in qs])


def f(x, sign=False):
    if x is None:
        return "–"
    s = "+" if sign else ""
    return f"{x['mean']:{s}.3f} [{x['lo']:{s}.3f}, {x['hi']:{s}.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="def")
    ap.add_argument("--defs", default="wmsr,cpwbft,sac")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    ref = {k: load(*v) for k, v in REF.items()}
    assert all(ref.values()), "reference runs missing"
    res = {}
    lines = [f"# Debate defenses: {a.tag} (v1nn, PPO protocol, MMLU-Pro test 400 x seeds 3/4/5, honest Qwen3.5-9B)", "",
             "Reference (no defense): " + ", ".join(f"{k} {boot(list(per_q(v, 'final_acc').values()))['mean']:.3f}" for k, v in ref.items()), ""]
    hdr = ["defense", "none", "sd120", "sd120+dwm", "Δ sd120 vs none", "Δ sd120+dwm vs none",
           "DiD sd120 (vs undefended)", "DiD sd120+dwm (vs undefended)"]
    lines += ["## final_acc (per-agent accuracy at the final submission)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    ref_d = {k: diff(per_q(ref[k], "final_acc"), per_q(ref["none"], "final_acc")) for k in ("sd120", "sd120+dwm")}
    lines.append("| none (reference) | " + " | ".join(f(boot(list(per_q(ref[k], 'final_acc').values()))) for k in REF)
                 + f" | {f(ref_d['sd120'], True)} | {f(ref_d['sd120+dwm'], True)} | – | – |")
    rows2 = []
    for df in a.defs.split(","):
        runs = {k: load(f"{a.tag}_{df}_{DIRSUF[k]}", COND[k]) for k in REF}
        res[df] = {"runs": {k: f"{a.tag}_{df}_{DIRSUF[k]}" for k in REF}}
        acc = {k: boot(list(per_q(v, "final_acc").values())) if v else None for k, v in runs.items()}
        dl = {k: diff(per_q(runs[k], "final_acc"), per_q(runs["none"], "final_acc")) if runs[k] and runs["none"] else None
              for k in ("sd120", "sd120+dwm")}
        did = {}
        for k in ("sd120", "sd120+dwm"):
            if runs[k] and runs["none"]:
                dq = {q: per_q(runs[k], "final_acc")[q] - per_q(runs["none"], "final_acc")[q]
                      for q in set(per_q(runs[k], "final_acc")) & set(per_q(runs["none"], "final_acc"))}
                rq = {q: per_q(ref[k], "final_acc")[q] - per_q(ref["none"], "final_acc")[q] for q in dq}
                did[k] = diff(dq, rq)
            else:
                did[k] = None
        res[df].update({"final_acc": acc, "delta_vs_none": dl, "did": did})
        lines.append(f"| {df} | " + " | ".join(f(acc[k]) for k in REF) + f" | {f(dl['sd120'], True)} | {f(dl['sd120+dwm'], True)}"
                     f" | {f(did['sd120'], True)} | {f(did['sd120+dwm'], True)} |")
        # defended - undefended per condition, group acc, diagnostics
        dvu = {k: diff(per_q(runs[k], "final_acc"), per_q(ref[k], "final_acc")) if runs[k] else None for k in REF}
        gacc = {k: boot(list(per_q(v, "group_acc").values())) if v else None for k, v in runs.items()}
        hid = {k: boot(list(per_q(v, "def_attacker_hidden").values())) if v and k != "none" else None for k, v in runs.items()}
        mh = {k: boot(list(per_q(v, "def_mean_hidden").values())) if v else None for k, v in runs.items()}
        res[df].update({"defended_minus_undefended": dvu, "group_acc": gacc, "attacker_hidden": hid, "mean_hidden": mh})
        rows2.append(f"| {df} | " + " | ".join(f(dvu[k], True) for k in REF) + " | " + " | ".join(f(gacc[k]) for k in REF)
                     + " | " + " | ".join(f(hid[k]) if k != "none" else "–" for k in REF) + " | " + " | ".join(f(mh[k]) for k in REF) + " |")
    hdr2 = ["defense", "defended − undefended: none", "sd120", "sd120+dwm", "group acc: none", "sd120", "sd120+dwm",
            "attacker hidden: none", "sd120", "sd120+dwm", "mean neighbors hidden: none", "sd120", "sd120+dwm"]
    lines += ["", "## clean cost / protection, group accuracy (majority == gold), filter diagnostics", "",
              "| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)] + rows2
    lines += ["", "Reference group acc: " + ", ".join(f"{k} {boot(list(per_q(v, 'group_acc').values()))['mean']:.3f}" for k, v in ref.items()),
              "", "All CIs: per-question bootstrap, 10k resamples. DiD = (Δ_attack with defense) − (Δ_attack without), paired by question; positive = defense weakens the attack."]
    print("\n".join(lines))
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
