"""Rule-paraphrase mutation summary (2026-11-05): one row per honest system-prompt wording (ID = exp10_nn v1nn runs,
R1/R2 = debate_wm/mut_chain_lambda.sh runs on the same 400 test questions x seeds 3/4/5), final_acc per cond, Δ vs the
set's own none, RL gain (sd120 − base), WM gain (sd120+dw − sd120), question-level paired bootstrap CIs (eval_debate._boot_diff).
Second table: difference-in-differences vs ID, paired by question (per-question Δ averaged over seeds, 2,000 bootstraps).
python debate_wm/mut_cross.py [TAG ...] > ~/runs/mut_cross.md"""
import json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

R = Path("/path/to/runs")
SD, B, P4 = "targeted_model_rl_sd120", "targeted_model_base", "targeted_4b"
SDW = SD + "__wmbon8dw"
SETS = [("ID wording (v1nn)", "exp10_nn")] + [(f"{t} (profile {t.replace('mut_', 'v1nn_')})", t) for t in (sys.argv[1:] or ["mut_r1", "mut_r2"])]


def load(tag):
    per = {}
    for n in ("none", "ours", "dwmbon"):
        p = R / f"{tag}_{n}" / "state.json"
        if not p.exists():
            continue
        for d in json.load(open(p))["debates"]:
            if d["final"] is not None:
                per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(E.debate_metrics(d))
    return per


def mean(per, c):
    return "%.3f" % E._boot_mean(per[c], "final_acc")["mean"] if c in per else "–"


def diff(per, a, b):
    if a not in per or b not in per:
        return "–"
    r = E._boot_diff(per[a], per[b], "final_acc")
    return "%+.3f [%+.3f, %+.3f]" % (r["diff"], *r["ci95"])


def qdelta(per, a, b):
    """per-question Δ final_acc (mean over seeds) a − b."""
    return {q: statistics.mean(m["final_acc"] for m in per[a][q]) - statistics.mean(m["final_acc"] for m in per[b][q])
            for q in set(per[a]) & set(per[b])}


def did(x, y, n=2000):
    """x, y: per-question Δ dicts; returns mean(x − y) over common questions and bootstrap CI."""
    qs = sorted(set(x) & set(y))
    if not qs:
        return "–"
    d = [x[q] - y[q] for q in qs]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(n))
    return "%+.3f [%+.3f, %+.3f] (n=%d)" % (statistics.mean(d), bs[int(0.025 * n)], bs[int(0.975 * n)], len(qs))


data = {name: load(tag) for name, tag in SETS}
print("| rule wording | n Q | none | 4b | base | sd120 | sd120+dw | Δ sd120 vs none | Δ sd120+dw vs none | RL gain sd120 − base | WM gain sd120+dw − sd120 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, per in data.items():
    if "none" not in per:
        print(f"| {name} | missing |"); continue
    print(f"| {name} | {len(per['none'])} | {mean(per, 'none')} | {mean(per, P4)} | {mean(per, B)} | {mean(per, SD)} | {mean(per, SDW)} | "
          f"{diff(per, SD, 'none')} | {diff(per, SDW, 'none')} | {diff(per, SD, B)} | {diff(per, SDW, SD)} |")
print()
print("Difference-in-differences vs ID wording, paired by question (positive = attack weaker under the paraphrase):")
print()
print("| rule wording | none − none(ID) | Δsd120 − Δsd120(ID) | Δsd120+dw − Δsd120+dw(ID) | RL gain − RL gain(ID) | WM gain − WM gain(ID) |")
print("|---|---|---|---|---|---|")
ref = data[SETS[0][0]]
for name, per in list(data.items())[1:]:
    if "none" not in per or "none" not in ref:
        print(f"| {name} | missing |"); continue
    cells = []
    for a, b in (("none", None), (SD, "none"), (SDW, "none"), (SD, B), (SDW, SD)):
        if a not in per or a not in ref or (b and (b not in per or b not in ref)):
            cells.append("–"); continue
        if b is None:
            x = {q: statistics.mean(m["final_acc"] for m in per[a][q]) for q in per[a]}
            y = {q: statistics.mean(m["final_acc"] for m in ref[a][q]) for q in ref[a]}
        else:
            x, y = qdelta(per, a, b), qdelta(ref, a, b)
        cells.append(did(x, y))
    print(f"| {name} | " + " | ".join(cells) + " |")
