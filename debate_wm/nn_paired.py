"""All conditions of the ${TAG}_{none,ours,dpo,lit,wmbon} runs, paired against ${TAG}_none (same task/seed/seat/honest seeds).
Question-level paired bootstrap (2,000 resamples, seeds averaged per question) via eval_debate._boot_mean/_boot_diff.
python3 nn_paired.py TAG > out.json   (also prints a tab table on stderr)"""
import glob, json, os, sys
sys.path.insert(0, "/path/to/consensus-attacker")
from debate_ppo import eval_debate as E

tag = sys.argv[1]
KEYS = ["final_acc", "outcome_wrong", "majority_wrong", "all_wrong", "all_wrong_consensus", "majority_on_target", "change_rate",
        "attacker_valid", "attacker_arg_words", "pgold_r2", "pgold_r3", "pgold_r4"]
per, runs = {}, {}
for p in sorted(glob.glob("/path/to/runs/%s_*/state.json" % tag)):
    st = json.load(open(p))
    for d in st["debates"]:
        if d["final"] is None:
            continue
        per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(E.debate_metrics(d))
        runs[d["cond"]] = os.path.basename(os.path.dirname(p))
CAT = {json.loads(l)["task_id"]: json.loads(l)["metadata"].get("category", "?") for l in open("/path/to/data/mmlupro_v1/tasks.jsonl")}
import statistics
def sd(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0.0
out = {"conditions": {}, "paired_vs_none": {}, "variance": {}}
# variance: per-seed final_acc (mean over the 400 questions for each seed) + SD across seeds; per-category mean + SD across seeds
for p in sorted(glob.glob("/path/to/runs/%s_*/state.json" % tag)):
    st = json.load(open(p))
    acc = {}
    for d in st["debates"]:
        if d["final"] is None:
            continue
        m = E.debate_metrics(d)
        acc.setdefault(d["cond"], {}).setdefault(d["seed"], {}).setdefault(CAT.get(d["task_id"], "?"), []).append((m["final_acc"], m["outcome_wrong"], m.get("all_wrong_consensus", 0.0), m.get("majority_on_target", float("nan"))))
    for c, by_seed in acc.items():
        seeds = sorted(by_seed)
        per_seed = {str(s): sum(v[0] for cat in by_seed[s] for v in by_seed[s][cat]) / sum(len(by_seed[s][cat]) for cat in by_seed[s]) for s in seeds}
        per_seed_ow = {str(s): sum(v[1] for cat in by_seed[s] for v in by_seed[s][cat]) / sum(len(by_seed[s][cat]) for cat in by_seed[s]) for s in seeds}
        cats = sorted({cat for s in seeds for cat in by_seed[s]})
        by_cat = {}
        for cat in cats:
            vals = [sum(v[0] for v in by_seed[s][cat]) / len(by_seed[s][cat]) for s in seeds if cat in by_seed[s]]
            n_q = max(len(by_seed[s][cat]) for s in seeds if cat in by_seed[s])
            by_cat[cat] = {"n_questions": n_q, "final_acc": sum(vals) / len(vals), "sd_seeds": sd(vals)}
        MET = ["final_acc", "outcome_wrong", "all_wrong_consensus", "majority_on_target"]
        sd_seeds = {}
        for mi, mk in enumerate(MET):
            vals = [sum(v[mi] for cat in by_seed[s] for v in by_seed[s][cat]) / sum(len(by_seed[s][cat]) for cat in by_seed[s]) for s in seeds]
            if all(x == x for x in vals): sd_seeds[mk] = sd(vals)
        out["variance"][c] = {"sd_seeds": sd_seeds, "final_acc_per_seed": per_seed, "final_acc_sd_seeds": sd(list(per_seed.values())),
                              "outcome_wrong_per_seed": per_seed_ow, "outcome_wrong_sd_seeds": sd(list(per_seed_ow.values())),
                              "final_acc_sd_categories": sd([v["final_acc"] for v in by_cat.values()]), "by_category": by_cat}
for c, by_q in per.items():
    out["conditions"][c] = {"run": runs[c], "n_debates": sum(len(v) for v in by_q.values()), "n_questions": len(by_q),
                            **{k: E._boot_mean(by_q, k) for k in KEYS if all(k in m for ms in by_q.values() for m in ms)}}
if "none" in per:
    for c in per:
        if c != "none":
            out["paired_vs_none"][c] = {k: E._boot_diff(per[c], per["none"], k) for k in KEYS if k in out["conditions"][c] and k in out["conditions"]["none"]}
json.dump(out, sys.stdout, indent=1)
def f(x): return "%.3f" % x
rows = sorted(out["conditions"], key=lambda c: out["conditions"][c]["final_acc"]["mean"])
print("cond\trun\tfinal_acc ± SD(seeds) [95% CI]\tdelta vs none [95% CI]\toutcome_wrong ± SD\tall_wrong_consensus ± SD\tmajority_on_target ± SD\tvalid\twords\tfinal_acc seed3/4/5\tSD seeds\tSD categories", file=sys.stderr)
for c in rows:
    v = out["conditions"][c]; d = out["paired_vs_none"].get(c, {}).get("final_acc")
    print("\t".join([c, v["run"], "%s ± %s [%s, %s]" % (f(v["final_acc"]["mean"]), f(out["variance"][c]["final_acc_sd_seeds"]), f(v["final_acc"]["ci95"][0]), f(v["final_acc"]["ci95"][1])),
                     ("%+.3f [%+.3f, %+.3f]" % (d["diff"], d["ci95"][0], d["ci95"][1])) if d else "-",
                     "%s ± %s" % (f(v["outcome_wrong"]["mean"]), f(out["variance"][c]["sd_seeds"].get("outcome_wrong", 0))),
                     "%s ± %s" % (f(v["all_wrong_consensus"]["mean"]), f(out["variance"][c]["sd_seeds"].get("all_wrong_consensus", 0))),
                     ("%s ± %s" % (f(v["majority_on_target"]["mean"]), f(out["variance"][c]["sd_seeds"].get("majority_on_target", 0)))) if "majority_on_target" in v else "-",
                     f(v["attacker_valid"]["mean"]) if "attacker_valid" in v else "-",
                     "%.1f" % v["attacker_arg_words"]["mean"] if "attacker_arg_words" in v else "-",
                     "/".join(f(x) for x in out["variance"][c]["final_acc_per_seed"].values()), f(out["variance"][c]["final_acc_sd_seeds"]),
                     f(out["variance"][c]["final_acc_sd_categories"])]), file=sys.stderr)
cats = sorted({cat for c in out["variance"] for cat in out["variance"][c]["by_category"]})
print("\ncategory table (final_acc, mean over seeds)\ncond\t" + "\t".join("%s(n=%d)" % (cat, out["variance"][rows[0]]["by_category"].get(cat, {}).get("n_questions", 0)) for cat in cats), file=sys.stderr)
for c in rows:
    print(c + "\t" + "\t".join(f(out["variance"][c]["by_category"][cat]["final_acc"]) if cat in out["variance"][c]["by_category"] else "-" for cat in cats), file=sys.stderr)
