"""RQ4 ablation table (2026-11-05): the 2x2x2 factorial (WM-scored init x RL vs WM x dwm-BoN8) + RL checkpoints, from
exp10_nn_{none,ours,dpo6,dwmbon,wmbon,abl_dwm,abl_rlcurve}. Copy of nn_paired_dwm.py with other conds/pairs: question-level
paired bootstrap (2,000), SD = across seeds 3/4/5 of the per-seed mean. Markdown out.
python debate_wm/abl_paired.py [TAG] > out.md"""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

tag = sys.argv[1] if len(sys.argv) > 1 else "exp10_nn"
R = "/path/to/runs"
per, per_seed, src = {}, {}, {}
for name in ("none", "ours", "dpo6", "wmbon", "dwmbon", "abl_dwm", "abl_rlcurve"):
    p = Path(R) / f"{tag}_{name}" / "state.json"
    for d in json.load(open(p))["debates"]:
        if d["final"] is None:
            continue
        m = E.debate_metrics(d)
        per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(d["cond"], {}).setdefault(d["seed"], []).append(m)
        src[d["cond"]] = f"{tag}_{name}"
B, Z, SD, C = "targeted_model_base", "targeted_model_stratdpo", "targeted_model_rl_sd120", "targeted_model_rl_c120"
DW = "__wmbon8dw"
CONDS = ["none", B, B + DW, Z, Z + DW, C, C + DW, SD, SD + DW, C + "__wmbon8wb", SD + "__wmbon8wb",
         "targeted_model_rl_c040", "targeted_model_rl_c080", "targeted_model_rl_sd040", "targeted_model_rl_sd080"]
PAIRS = [(Z, B), (Z + DW, B + DW), (SD, C), (SD + DW, C + DW),                      # WM-scored init
         (C, B), (C + DW, B + DW), (SD, Z), (SD + DW, Z + DW),                      # RL vs WM
         (B + DW, B), (Z + DW, Z), (C + DW, C), (SD + DW, SD),                      # dwm selection
         (SD + DW, B),                                                              # full method vs pi_ref
         ("targeted_model_rl_c040", B), ("targeted_model_rl_c080", B), (C, "targeted_model_rl_c080"),
         ("targeted_model_rl_sd040", Z), ("targeted_model_rl_sd080", Z), (SD, "targeted_model_rl_sd080")]
KEYS = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target"]


def seed_sd(c, k):
    xs = [statistics.mean(m[k] for m in ms if k in m) for ms in per_seed[c].values() if any(k in m for m in ms)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def sub(c, k):
    s = {q: [m for m in ms if k in m] for q, ms in per[c].items()}
    return {q: v for q, v in s.items() if v}


def seeds_str(c):
    return " / ".join("%.3f" % statistics.mean(m["final_acc"] for m in per_seed[c][s]) for s in sorted(per_seed[c]))


print("| condition | source run | n | final_acc ± SD [95% CI] | Δ final_acc vs none [95% CI] | outcome_wrong | all_wrong_consensus | majority_on_target | final_acc seeds 3 / 4 / 5 |")
print("|---|---|---|---|---|---|---|---|---|")
for c in CONDS:
    if c not in per:
        print(f"| {c} | missing |"); continue
    cells = []
    for k in KEYS:
        s = sub(c, k)
        cells.append("%.3f ± %.3f [%.3f, %.3f]" % (E._boot_mean(s, k)["mean"], seed_sd(c, k), *E._boot_mean(s, k)["ci95"]) if s else "–")
    dn = "–" if c == "none" else "%+.3f [%+.3f, %+.3f]" % (lambda r: (r["diff"], *r["ci95"]))(E._boot_diff(per[c], per["none"], "final_acc"))
    n = sum(len(v) for v in per[c].values())
    print(f"| {c} | {src[c]} | {n} | {cells[0]} | {dn} | {cells[1]} | {cells[2]} | {cells[3]} | {seeds_str(c)} |")
print()
print("| a − b | Δ final_acc [95% CI] | Δ outcome_wrong | Δ all_wrong_consensus | Δ majority_on_target | Δ final_acc per seed 3/4/5 |")
print("|---|---|---|---|---|---|")
for a, b in PAIRS:
    if a not in per or b not in per:
        print(f"| {a} − {b} | missing |"); continue
    cells = []
    for k in KEYS:
        A, B = sub(a, k), sub(b, k)
        cells.append("%+.3f [%+.3f, %+.3f]" % (lambda r: (r["diff"], *r["ci95"]))(E._boot_diff(A, B, k)) if set(A) & set(B) else "–")
    ps = [statistics.mean(m["final_acc"] for m in per_seed[a][s]) - statistics.mean(m["final_acc"] for m in per_seed[b][s])
          for s in sorted(set(per_seed[a]) & set(per_seed[b]))]
    print(f"| {a} − {b} | " + " | ".join(cells) + " | " + " / ".join("%+.3f" % x for x in ps) + " |")
