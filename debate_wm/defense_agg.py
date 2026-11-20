"""Post-hoc aggregation-rule defense for the debate (2026-11-08). Reads existing run states (read-only) and
re-aggregates the three honest agents' FINAL submissions with different rules. Per-agent final_acc is unaffected by
the aggregation rule, so the metric here is group accuracy = P[aggregated answer == gold].
Rules: maj (>=2 of 3, deadlock counts as not correct; = protocol rule), plural_conf (plurality, ties by summed own-answer
prob), conf_vote (each agent votes its sampled answer with weight = its prob of that answer; CP-WBFT stage-2 style),
soft (argmax of mean final belief)."""
import json, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, "/path/to/consensus-attacker")
from debate_ppo import dprotocol as D

RUNS = {"none": ("exp10_nn_none", "none"), "base": ("exp10_nn_ours", "targeted_model_base"),
        "sd120": ("exp10_nn_ours", "targeted_model_rl_sd120"),
        "sd120+dwm": ("exp10_nn_dwmbon", "targeted_model_rl_sd120__wmbon8dw")}
RULES = ["maj", "plural_conf", "conf_vote", "soft"]
N_BOOT = 10000


def agg(d):
    L = D.LETTERS
    fin = [r["answer"] for r in d["final"]]
    probs = d["beliefs"][D.N_ROUNDS]                       # final-submission beliefs, honest seats in order
    own = [p[L.index(a)] if a else 0.0 for a, p in zip(fin, probs)]
    out = {"maj": D.majority(fin)}
    w = defaultdict(float); c = Counter(a for a in fin if a)
    for a, o in zip(fin, own):
        if a: w[a] += o
    top = max(c.values()) if c else 0
    out["plural_conf"] = max((a for a in c if c[a] == top), key=lambda a: w[a]) if c else None
    out["conf_vote"] = max(w, key=w.get) if w else None
    n = len(probs[0]); mean = [sum(p[k] for p in probs) / 3 for k in range(n)]
    out["soft"] = L[max(range(n), key=lambda k: mean[k])]
    return out


def boot(vals, seed=0):
    rng = random.Random(seed); n = len(vals)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(vals) / n, b[int(.025 * N_BOOT)], b[int(.975 * N_BOOT) - 1]


per = {}                                                  # cond -> rule -> task -> [0/1]
states = {}
for name, (run, cond) in RUNS.items():
    if run not in states:
        states[run] = json.load(open(f"/path/to/runs/{run}/state.json"))
    st = states[run]; D.set_profile_set(st["config"].get("profile_set", "v1"))
    assert st["config"]["profile_set"] == "v1nn" and st["config"]["split"] == "test"
    per[name] = {r: defaultdict(list) for r in RULES}
    for d in st["debates"]:
        if d["cond"] != cond: continue
        a = agg(d)
        for r in RULES:
            per[name][r][d["task_id"]].append(float(a[r] == d["gold"]))
qs = sorted(per["none"]["maj"])
def qmean(x): return [sum(x[q]) / len(x[q]) for q in qs]
res = {}
lines = ["| condition (v1nn, test 400 x seeds 3/4/5) | " + " | ".join(RULES) + " | Δ(rule − maj) conf_vote | Δ(rule − maj) soft |",
         "|---|" + "---|" * (len(RULES) + 2)]
for name in RUNS:
    row = {}
    for r in RULES:
        row[r] = boot(qmean(per[name][r]))
    dc = boot([a - b for a, b in zip(qmean(per[name]["conf_vote"]), qmean(per[name]["maj"]))])
    ds = boot([a - b for a, b in zip(qmean(per[name]["soft"]), qmean(per[name]["maj"]))])
    res[name] = {"rules": row, "conf_vote_minus_maj": dc, "soft_minus_maj": ds}
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    g = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    lines.append(f"| {name} | " + " | ".join(f(row[r]) for r in RULES) + f" | {g(dc)} | {g(ds)} |")
# attack effect under each rule (paired vs none, same rule)
lines += ["", "| attack Δ vs none, group acc (same rule) | " + " | ".join(RULES) + " |", "|---|" + "---|" * len(RULES)]
for name in ["base", "sd120", "sd120+dwm"]:
    cells = []
    for r in RULES:
        t = boot([a - b for a, b in zip(qmean(per[name][r]), qmean(per["none"][r]))])
        res[name][f"delta_vs_none_{r}"] = t
        cells.append(f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]")
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
json.dump(res, open("/path/to/runs/defense_agg/agg_posthoc.json", "w"), indent=1)
open("/path/to/runs/defense_agg/agg_posthoc.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
