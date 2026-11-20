"""No-peek selector tables (2026-11-12): exp10_nn_np_{ours,ju,kraidia} (dwm simulates the honest round-r replies,
debate_ppo/eval_debate_dwm_np.py) paired against the peeking selector runs exp10_nn_{dwmbon,ctl_ju,ctl_kraidia}, the
single-sample runs exp10_nn_{ours,dpo,lit} and exp10_nn_none. Question-level paired bootstrap via eval_debate._boot_mean /
_boot_diff (2,000), SD across seeds 3/4/5. Also the simulation fidelity: agreement of the simulated round-r answers with the
real honest agents' round-r answers on the same history.
python debate_wm/np_paired.py > ~/runs/exp10_nn_np_paired.md"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

R = Path("/path/to/runs")
SD, JU, KR, W = "targeted_model_rl_sd120", "ju_dpo_q3_4b", "kraidia_4b", "__wmbon8dw"
# label -> (run, cond)
RUNS = {
    "none": ("none", "none"),
    "Ju DPO": ("dpo", JU), "Ju DPO + WM-BoN8 (peek)": ("ctl_ju", JU + "_wmbon8dw"),
    "Ju DPO + WM-BoN8 (no-peek)": ("np_ju", JU + "_wmbon8dw"),
    "Kraidia PPA": ("lit", KR), "Kraidia PPA + WM-BoN8 (peek)": ("ctl_kraidia", KR + "_wmbon8dw"),
    "Kraidia PPA + WM-BoN8 (no-peek)": ("np_kraidia", KR + "_wmbon8dw"),
    "sd120": ("ours", SD), "sd120 + WM-BoN8 (peek)": ("dwmbon", SD + W), "sd120 + WM-BoN8 (no-peek)": ("np_ours", SD + W),
}
PAIRS = [("sd120 + WM-BoN8 (no-peek)", "sd120 + WM-BoN8 (peek)"), ("sd120 + WM-BoN8 (no-peek)", "sd120"),
         ("sd120 + WM-BoN8 (no-peek)", "none"), ("sd120 + WM-BoN8 (peek)", "sd120"),
         ("Ju DPO + WM-BoN8 (no-peek)", "Ju DPO + WM-BoN8 (peek)"), ("Ju DPO + WM-BoN8 (no-peek)", "Ju DPO"),
         ("Kraidia PPA + WM-BoN8 (no-peek)", "Kraidia PPA + WM-BoN8 (peek)"), ("Kraidia PPA + WM-BoN8 (no-peek)", "Kraidia PPA"),
         ("sd120 + WM-BoN8 (no-peek)", "Ju DPO + WM-BoN8 (no-peek)"), ("sd120 + WM-BoN8 (no-peek)", "Kraidia PPA + WM-BoN8 (no-peek)")]
KEYS = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target"]

per, per_seed, fid, cache = {}, {}, {}, {}
for lab, (run, cond) in RUNS.items():
    p = R / f"exp10_nn_{run}" / "state.json"
    if not p.exists():
        continue
    st = cache.setdefault(run, json.load(open(p)))
    ds = [d for d in st["debates"] if d["cond"] == cond]
    if not ds or any(d["final"] is None for d in ds):
        print(f"(skipping unfinished exp10_nn_{run})", file=sys.stderr)
        continue
    for d in ds:
        m = E.debate_metrics(d)
        per.setdefault(lab, {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(lab, {}).setdefault(d["seed"], []).append(m)
        for r, sim in (d.get("np_sim") or {}).items():
            f = fid.setdefault(lab, {}).setdefault(int(r), [0, 0, 0, 0])
            for i, rep in sim.items():
                real = d["history"][int(r)][int(i)]["answer"]
                f[0] += rep["answer"] == real
                f[1] += 1
                f[2] += rep["answer"] == d["target"]
                f[3] += real == d["target"]


def seed_sd(c, k):
    xs = [statistics.mean(m[k] for m in ms if k in m) for ms in per_seed[c].values() if any(k in m for m in ms)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def cell(c, k):
    if not any(k in m for ms in per[c].values() for m in ms):
        return "–"
    b = E._boot_mean({q: [m for m in ms if k in m] for q, ms in per[c].items()}, k)
    return f"{b['mean']:.3f} ± {seed_sd(c, k):.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"


print("Protocol: PPO protocol (debate_ppo_v3, answer-first), honest set v1nn (suggestible/neutral/neutral, Qwen3.5-9B), "
      "MMLU-Pro test 400 q x seeds 3/4/5, targets = most-voted wrong option of `none` round 1. Cells: mean ± SD over seeds "
      "[question bootstrap 95% CI]. peek = selector reads the honest agents' real round-r replies; no-peek = debate-wm-v1 "
      "simulates them (1 sample).\n")
print("| condition | source run | n | " + " | ".join(KEYS) + " | final_acc seeds 3/4/5 |")
print("|---|---|---|" + "---|" * (len(KEYS) + 1))
for c in RUNS:
    if c not in per:
        continue
    n = sum(len(v) for v in per[c].values())
    ps = " / ".join(f"{statistics.mean(m['final_acc'] for m in per_seed[c][s]):.3f}" for s in sorted(per_seed[c]))
    print(f"| {c} | exp10_nn_{RUNS[c][0]} | {n} | " + " | ".join(cell(c, k) for k in KEYS) + f" | {ps} |")
print()
print("| a − b | Δ final_acc [95% CI] | Δ outcome_wrong | Δ all_wrong_consensus | Δ majority_on_target | Δ final_acc per seed |")
print("|---|---|---|---|---|---|")
for a, b in PAIRS:
    if a not in per or b not in per:
        continue
    out = []
    for k in KEYS:
        if k == "majority_on_target" and "none" in (a, b):
            out.append("–")
            continue
        r = E._boot_diff(per[a], per[b], k)
        out.append(f"{r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    ps = " / ".join(f"{statistics.mean(m['final_acc'] for m in per_seed[a][s]) - statistics.mean(m['final_acc'] for m in per_seed[b][s]):+.3f}"
                    for s in sorted(per_seed[a]))
    print(f"| {a} − {b} | " + " | ".join(out) + f" | {ps} |")
print()
print("Simulation fidelity (no-peek runs): simulated vs real honest answer in the same round, on the same history.\n")
print("| condition | round | n turns | sim = real answer | sim says target | real says target |")
print("|---|---|---|---|---|---|")
for c, rs in fid.items():
    for r in sorted(rs):
        a, n, st_, rt = rs[r]
        print(f"| {c} | {r + 1} | {n} | {a / n:.3f} | {st_ / n:.3f} | {rt / n:.3f} |")
