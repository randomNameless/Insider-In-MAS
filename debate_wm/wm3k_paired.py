"""Cheap-WM (3,000 real debates) selector rows (2026-11-11), as nn_paired_dwm.py: question-level paired bootstrap via
eval_debate._boot_mean/_boot_diff (2,000), SD = across seeds 3/4/5 of the per-seed mean. All arms: PPO protocol, v1nn,
Qwen3.5-9B honest, MMLU-Pro test 400 x seeds 3/4/5. python debate_wm/wm3k_paired.py > out.md"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

R = Path("/path/to/runs")
SD = "targeted_model_rl_sd120"
ARMS = [("none", "exp10_nn_none", "none"), ("sd120", "exp10_nn_ours", SD),
        ("sd120+BoN8 WM-full-4B (11,528)", "exp10_nn_dwmbon", SD + "__wmbon8dw"),
        ("sd120+BoN8 WM-3k-4B (3,000)", "exp10_nn_dwmbon3k_4b", SD + "__wmbon8dw"),
        ("sd120+BoN8 WM-3k-1.7B (3,000)", "exp10_nn_dwmbon3k_1p7b", SD + "__wmbon8dw")]
per, per_seed = {}, {}
for name, run, cond in ARMS:
    p = R / run / "state.json"
    if not p.exists():
        continue
    for d in json.load(open(p))["debates"]:
        if d["cond"] != cond or d["final"] is None:
            continue
        m = E.debate_metrics(d)
        per.setdefault(name, {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(name, {}).setdefault(d["seed"], []).append(m)
KEYS = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target"]


def seed_sd(c, k):
    xs = [statistics.mean(m[k] for m in ms if k in m) for ms in per_seed[c].values() if any(k in m for m in ms)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def sub(c, k):
    return {q: v for q, v in ((q, [m for m in ms if k in m]) for q, ms in per[c].items()) if v}


print("| arm (PPO protocol) | source run | n | final_acc ± SD [95% CI] | outcome_wrong | all_wrong_consensus | majority_on_target | final_acc seeds 3 / 4 / 5 |")
print("|---|---|---|---|---|---|---|---|")
for name, run, _ in ARMS:
    if name not in per:
        print(f"| {name} | {run} missing |"); continue
    cells = []
    for k in KEYS:
        s = sub(name, k)
        cells.append("%.3f ± %.3f [%.3f, %.3f]" % (E._boot_mean(s, k)["mean"], seed_sd(name, k), *E._boot_mean(s, k)["ci95"]) if s else "–")
    seeds = " / ".join("%.3f" % statistics.mean(m["final_acc"] for m in per_seed[name][s]) for s in sorted(per_seed[name]))
    print(f"| {name} | {run} | {sum(len(v) for v in per[name].values())} | {' | '.join(cells)} | {seeds} |")
print()
print("| a − b | Δ final_acc [95% CI] | Δ majority_on_target [95% CI] |")
print("|---|---|---|")
for a in [x[0] for x in ARMS[3:]]:
    for b in ["none", "sd120", ARMS[2][0]]:
        if a in per and b in per:
            r = E._boot_diff(per[a], per[b], "final_acc")
            mt = "–" if b == "none" else "%+.3f [%+.3f, %+.3f]" % (lambda r: (r["diff"], *r["ci95"]))(E._boot_diff(sub(a, "majority_on_target"), sub(b, "majority_on_target"), "majority_on_target"))
            print(f"| {a} − {b} | {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {mt} |")
if all(x[0] in per for x in ARMS[3:]):
    a, b = ARMS[4][0], ARMS[3][0]
    r = E._boot_diff(per[a], per[b], "final_acc")
    print(f"| {a} − {b} | {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | |")
