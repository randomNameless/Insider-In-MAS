"""6r variant (2026-10-30): adds the 500-token batch exp10_nn_dpo6r (targeted_model_stratdpo6r, targeted_model_stratdpo500).
Tables for the exp10_nn v6-format strat-DPO batch (2026-10-28). Loads every ${TAG}_*/state.json (as nn_paired.py),
question-level paired bootstrap via eval_debate._boot_mean/_boot_diff, SD = across seeds 3/4/5 of the per-seed mean.
python3 nn_paired_dpo6.py [TAG] > out.tsv"""
import glob, json, statistics, sys
sys.path.insert(0, "/path/to/consensus-attacker")
from debate_ppo import eval_debate as E

tag = sys.argv[1] if len(sys.argv) > 1 else "exp10_nn"
per, per_seed = {}, {}
for p in sorted(glob.glob("/path/to/runs/%s_*/state.json" % tag)):
    for d in json.load(open(p))["debates"]:
        if d["final"] is None:
            continue
        m = E.debate_metrics(d)
        per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(d["cond"], {}).setdefault(d["seed"], []).append(m)
NEW, OLD6, SD120, NATIVE = "targeted_model_stratdpo6", "targeted_model_stratdpo", "targeted_model_rl_sd120", "stratdpo_q3_4b"
NEW6R, OLD500 = "targeted_model_stratdpo6r", "targeted_model_stratdpo500"  # 500-token batch exp10_nn_dpo6r (2026-10-30)
CONDS = ["none", NEW6R, OLD500, NEW, OLD6, NATIVE, SD120]
PAIRS = [(NEW6R, "none"), (OLD500, "none"), (NEW6R, OLD500), (NEW6R, NEW), (NEW6R, OLD6), (NEW6R, NATIVE), (NEW6R, SD120), (OLD500, OLD6), (SD120, OLD500)]
KEYS = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target", "attacker_arg_words"]


def seed_sd(c, k):
    xs = [statistics.mean(m[k] for m in ms if k in m) for ms in per_seed[c].values() if any(k in m for m in ms)]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def fmt_mean(r, sd):
    return "%.3f ± %.3f [%.3f, %.3f]" % (r["mean"], sd, *r["ci95"])


def fmt_diff(r):
    return "%+.3f [%+.3f, %+.3f]" % (r["diff"], *r["ci95"])


print("condition\tn_debates\t" + "\t".join(KEYS))
for c in CONDS:
    if c not in per:
        print(c + "\tmissing")
        continue
    row = [c, str(sum(len(v) for v in per[c].values()))]
    for k in KEYS:
        sub = {q: [m for m in ms if k in m] for q, ms in per[c].items()}
        sub = {q: ms for q, ms in sub.items() if ms}
        row.append(fmt_mean(E._boot_mean(sub, k), seed_sd(c, k)) if sub else "")
    print("\t".join(row))
print()
print("a\tb\t" + "\t".join(k + " diff [95% CI]" for k in KEYS) + "\tfinal_acc diff per seed\tSD(seeds)")
for a, b in PAIRS:
    if a not in per or b not in per:
        print(f"{a}\t{b}\tmissing")
        continue
    row = [a, b]
    for k in KEYS:
        A = {q: [m for m in ms if k in m] for q, ms in per[a].items()}
        B = {q: [m for m in ms if k in m] for q, ms in per[b].items()}
        A = {q: v for q, v in A.items() if v}; B = {q: v for q, v in B.items() if v}
        row.append(fmt_diff(E._boot_diff(A, B, k)) if set(A) & set(B) else "")
    ps = [statistics.mean(m["final_acc"] for m in per_seed[a][s]) - statistics.mean(m["final_acc"] for m in per_seed[b][s])
          for s in sorted(set(per_seed[a]) & set(per_seed[b]))]
    print("\t".join(row + ["/".join("%+.3f" % x for x in ps), "%.3f" % (statistics.pstdev(ps) if len(ps) > 1 else 0.0)]))
