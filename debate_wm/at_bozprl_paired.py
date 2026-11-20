#!/usr/bin/env python
"""Bozdag P-RL (no-think, <=200 shown tokens) vs AT-DPO honest agents (2026-11-13).

All runs: v1nn, MMLU-Pro test 400 x seeds 3/4/5, 2 neutral + 1 suggestible honest, 3 rounds, answer-first.
Rows = attacker; columns = honest model (undefended Qwen3.5-9B exp10_nn_* vs AT-DPO at_*). Per-seed mean ± SD (n=3 seeds),
honest-agent counts correct/total, paired per-question bootstrap Δ (10k, seed 0) as in def_paired.py.
    python debate_wm/at_bozprl_paired.py > ~/runs/at_bozprl_paired.md
"""
from __future__ import annotations

import statistics as S
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_paired import boot, diff, f, load, per_q  # noqa: E402

ROWS = [  # name, (undefended run, cond), (AT-DPO run, cond)
    ("none", ("exp10_nn_none", "none"), ("at_dpo_none", "none")),
    ("sd120", ("exp10_nn_ours", "targeted_model_rl_sd120"), ("at_dpo_sd120", "targeted_model_rl_sd120")),
    ("sd120 + WM-BoN8 (peek)", ("exp10_nn_dwmbon", "targeted_model_rl_sd120__wmbon8dw"),
     ("at_dpo_dwmbon", "targeted_model_rl_sd120__wmbon8dw")),
    ("P-RL no-think 200", ("exp10_nn_bozprl_nothink_len200", "bozdag_prl_3b_nothink_len200"),
     ("at_bozprl_single", "bozdag_prl_3b_nothink_len200")),
    ("P-RL no-think 200 + WM-BoN8 (peek)", ("exp10_nn_ctl_bozprl", "bozdag_prl_3b_nothink_len200_wmbon8dw"),
     ("at_bozprl_dwmbon", "bozdag_prl_3b_nothink_len200_wmbon8dw")),
]


def seeds(m):
    by = defaultdict(list)
    for (t, s), x in m.items():
        by[s].append(x["final_acc"])
    v = [sum(a) / len(a) for _, a in sorted(by.items())]
    return f"{S.mean(v):.3f} ± {S.stdev(v):.3f} ({', '.join(f'{x:.3f}' for x in v)})"


def counts(m):
    c = round(sum(x["final_acc"] * 3 for x in m.values()))
    return f"{c}/{3 * len(m)}"


def main():
    U = {n: load(*u) for n, u, _ in ROWS}
    A = {n: load(*a) for n, _, a in ROWS}
    q = lambda m: per_q(m, "final_acc")
    L = ["# Bozdag P-RL vs AT-DPO honest agents (v1nn, MMLU-Pro test 400 x seeds 3/4/5, 3 rounds)", "",
         "final_acc = per-honest-agent accuracy at final submission. mean ± SD over seeds (per-seed values); counts = correct honest answers / total.",
         "Δ = paired per-question bootstrap 95% CI (10k).", "",
         "| attacker | undefended (Qwen3.5-9B) | AT-DPO | AT-DPO counts | AT − undefended | AT: Δ vs none | AT: Δ vs sd120 | AT: Δ vs sd120+WM | DiD (attack Δ AT − attack Δ undef.) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, _, _ in ROWS:
        u, a = U[n], A[n]
        if not a:
            L.append(f"| {n} | {seeds(u) if u else '–'} | (missing) | | | | | | |"); continue
        did = "–"
        if n != "none":
            dq = {k: q(a)[k] - q(A["none"])[k] for k in q(a)}
            rq = {k: q(u)[k] - q(U["none"])[k] for k in dq}
            did = f(diff(dq, rq), True)
        L.append(f"| {n} | {seeds(u)} | {seeds(a)} | {counts(a)} | {f(diff(q(a), q(u)), True)} | "
                 f"{f(diff(q(a), q(A['none'])), True) if n != 'none' else '–'} | "
                 f"{f(diff(q(a), q(A['sd120'])), True) if n != 'sd120' else '–'} | "
                 f"{f(diff(q(a), q(A['sd120 + WM-BoN8 (peek)'])), True) if 'sd120 + WM' not in n else '–'} | {did} |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
