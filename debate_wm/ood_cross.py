"""Cross-dataset OOD-transfer summary (2026-11-04): one row per dataset (MMLU-Pro ID = exp10_nn runs), final_acc per cond,
Δ vs none, RL gain (sd120 − base), WM gain (sd120+dw − sd120) with question-level paired bootstrap CIs (eval_debate._boot_diff).
python debate_wm/ood_cross.py > ~/runs/ood_cross.md"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

R = Path("/path/to/runs")
SD, B, P4 = "targeted_model_rl_sd120", "targeted_model_base", "targeted_4b"
SDW, BW = SD + "__wmbon8dw", B + "__wmbon8dw"
SETS = [("MMLU-Pro (ID)", "exp10_nn", ("none", "ours", "dwmbon")), ("TruthfulQA-MC1", "ood_truthfulqa_mc1", ("none", "ours", "dwmbon")),
        ("GPQA-Diamond", "ood_gpqa_diamond", ("none", "ours", "dwmbon")), ("LogiQA 2.0", "ood_logiqa2", ("none", "ours", "dwmbon")),
        ("MedQA", "ood_medqa", ("none", "ours", "dwmbon"))]


def load(tag, names):
    per = {}
    for n in names:
        p = R / f"{tag}_{n}" / "state.json"
        if not p.exists():
            return None
        for d in json.load(open(p))["debates"]:
            if d["final"] is not None:
                per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(E.debate_metrics(d))
    return per


def mean(per, c):
    return E._boot_mean(per[c], "final_acc")["mean"] if c in per else float("nan")


def diff(per, a, b):
    if a not in per or b not in per:
        return "–"
    r = E._boot_diff(per[a], per[b], "final_acc")
    return "%+.3f [%+.3f, %+.3f]" % (r["diff"], *r["ci95"])


print("| dataset | n Q | none | 4b | base | sd120 | base+dw | sd120+dw | Δ sd120 vs none | Δ sd120+dw vs none | RL gain sd120 − base | WM gain sd120+dw − sd120 | WM gain on base | rel. Δ sd120 / sd120+dw |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, tag, names in SETS:
    per = load(tag, names)
    if per is None:
        print(f"| {name} | missing |"); continue
    nq = len(per["none"]); n0 = mean(per, "none")
    accs = " | ".join("%.3f" % mean(per, c) if c in per else "–" for c in ("none", P4, B, SD, BW, SDW))
    rel = "%.0f%% / %.0f%%" % (100 * (mean(per, SD) - n0) / n0, 100 * (mean(per, SDW) - n0) / n0)
    print(f"| {name} | {nq} | {accs} | {diff(per, SD, 'none')} | {diff(per, SDW, 'none')} | {diff(per, SD, B)} | {diff(per, SDW, SD)} | {diff(per, BW, B)} | {rel} |")
