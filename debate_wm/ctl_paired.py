"""Same-selector control tables (2026-11-09): exp10_nn_ctl_{ju,kraidia} paired against exp10_nn_{none,dpo,lit,ours,dwmbon,
abl_dwm}; question-level paired bootstrap via eval_debate._boot_mean/_boot_diff (2,000), SD across seeds 3/4/5. Also the
inference-cost columns from the run logs / state (generated tokens of all candidates, spoken tokens, WM prompt tokens).
python debate_wm/ctl_paired.py > ~/runs/exp10_nn_ctl_paired.md"""
import json, re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E

R = Path("/path/to/runs")
per, per_seed, src, cost = {}, {}, {}, {}
for name in ("none", "dpo", "lit", "ours", "dwmbon", "abl_dwm", "ctl_ju", "ctl_kraidia"):
    p = R / f"exp10_nn_{name}" / "state.json"
    if not p.exists():
        continue
    st = json.load(open(p))
    if not all(d["final"] is not None for d in st["debates"]):
        print(f"(skipping unfinished exp10_nn_{name})", file=sys.stderr)
        continue
    for d in st["debates"]:
        m = E.debate_metrics(d)
        c = d["cond"]
        per.setdefault(c, {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(c, {}).setdefault(d["seed"], []).append(m)
        src[c] = f"exp10_nn_{name}"
        k = cost.setdefault(c, {"spoken": [], "cand": [], "turns": 0})
        k["turns"] += len(d.get("attacker_meta") or [])
        k["spoken"] += [a["arg_tokens"] for a in d.get("attacker_meta") or [] if a.get("arg_tokens")]
        for w in d.get("wmbon") or []:
            if "cand_tokens" in w:
                k["cand"].append(sum(w["cand_tokens"]))
    log = (R / f"exp10_nn_{name}" / "play.log").read_text(errors="ignore")
    for cnd, js in re.findall(r"dwm\[(\S+?)\] r=\d+: (\{.*\})", log):
        j = json.loads(js)
        k = cost.setdefault(cnd, {"spoken": [], "cand": [], "turns": 0})
        k["wm_prompt"] = k.get("wm_prompt", 0) + j["prompt_tokens"]
        k["wm_wall"] = k.get("wm_wall", 0) + j["wall_s"]
        k["wm_n"] = k.get("wm_n", 0) + j["n_turns"]
    for cnd, js in re.findall(r"attacker\[(\S+?)\] r=\d+: (\{.*\})", log):
        j = json.loads(js)
        k = cost.setdefault(cnd, {"spoken": [], "cand": [], "turns": 0})
        k["gen_wall"] = k.get("gen_wall", 0) + j["gen_wall_s"]
SD, JU, KR = "targeted_model_rl_sd120", "ju_dpo_q3_4b", "kraidia_4b"
W = "__wmbon8dw"
CONDS = ["none", JU, JU + "_wmbon8dw", KR, KR + "_wmbon8dw", "stratdpo_q3_4b", "targeted_model_stratdpo", "targeted_model_stratdpo" + W,
         "targeted_model_base", "targeted_model_base" + W, SD, SD + W]
PAIRS = [(JU + "_wmbon8dw", JU), (KR + "_wmbon8dw", KR), (SD + W, SD), ("targeted_model_stratdpo" + W, "targeted_model_stratdpo"),
         ("targeted_model_base" + W, "targeted_model_base"),
         (SD + W, JU + "_wmbon8dw"), (SD + W, KR + "_wmbon8dw"), (SD, JU + "_wmbon8dw"), (SD, KR + "_wmbon8dw"),
         ("targeted_model_stratdpo" + W, JU + "_wmbon8dw"), (JU + "_wmbon8dw", "none"), (KR + "_wmbon8dw", "none"), (SD + W, "none"),
         (SD, JU), (SD, KR)]
KEYS = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target"]


def seed_sd(c, k):
    xs = [statistics.mean(m[k] for m in ms if k in m) for ms in per_seed[c].values() if any(k in m for m in ms)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def cell(c, k):
    if not any(k in m for ms in per[c].values() for m in ms):
        return "–"
    b = E._boot_mean({q: [m for m in ms if k in m] for q, ms in per[c].items()}, k)
    return f"{b['mean']:.3f} ± {seed_sd(c, k):.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"


print("| condition | source run | n | " + " | ".join(KEYS) + " | final_acc seeds 3/4/5 |")
print("|---|---|---|" + "---|" * (len(KEYS) + 1))
for c in CONDS:
    if c not in per:
        continue
    n = sum(len(v) for v in per[c].values())
    ps = " / ".join(f"{statistics.mean(m['final_acc'] for m in per_seed[c][s]):.3f}" for s in sorted(per_seed[c]))
    print(f"| {c} | {src[c]} | {n} | " + " | ".join(cell(c, k) for k in KEYS) + f" | {ps} |")
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
print("| condition | attacker turns | spoken tokens/turn | generated tokens/turn (all candidates) | WM prompt tokens/turn | candidate gen wall s (all turns) | WM scoring wall s (all turns) |")
print("|---|---|---|---|---|---|---|")
for c in CONDS:
    k = cost.get(c)
    if not k or not k["turns"]:
        continue
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"
    print(f"| {c} | {k['turns']} | {f(statistics.mean(k['spoken']) if k['spoken'] else None)} | "
          f"{f(statistics.mean(k['cand']) if k['cand'] else None)} | {f(k['wm_prompt'] / k['turns'] if 'wm_prompt' in k else None, 0)} | "
          f"{f(k.get('gen_wall'), 0)} | {f(k.get('wm_wall'), 0)} |")
