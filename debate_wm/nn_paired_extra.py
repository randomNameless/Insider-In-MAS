"""Extra paired contrasts for exp10_nn (same loader as nn_paired.py): WM-BoN vs its single-sample base, ours vs strongest baseline, per-seed deltas."""
import glob, json, sys, statistics
sys.path.insert(0, "/path/to/consensus-attacker")
from debate_ppo import eval_debate as E
tag = sys.argv[1]
per, per_seed = {}, {}
for p in sorted(glob.glob("/path/to/runs/%s_*/state.json" % tag)):
    for d in json.load(open(p))["debates"]:
        if d["final"] is None: continue
        m = E.debate_metrics(d)
        per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(m)
        per_seed.setdefault(d["cond"], {}).setdefault(d["seed"], []).append(m)
PAIRS = [("targeted_model_rl_sd120__wmbon8wb","targeted_model_rl_sd120"),("targeted_model_rl_c120__wmbon8wb","targeted_model_rl_c120"),
         ("targeted_model_rl_sd120__wmbon8wb","stratdpo_q3_4b"),("targeted_model_rl_sd120","stratdpo_q3_4b"),("targeted_model_rl_c120","stratdpo_q3_4b"),
         ("targeted_model_rl_sd120","targeted_model_rl_c120"),("stratdpo_q3_4b","ju_dpo_q3_4b"),("stratdpo_q3_4b","kraidia_4b"),("targeted_model_rl_sd120__wmbon8wb","ju_dpo_q3_4b")]
KEYS = ["final_acc","outcome_wrong","all_wrong_consensus","majority_on_target"]
out = {}
print("a\tb\t" + "\t".join("%s: diff [95%% CI]" % k for k in KEYS) + "\tfinal_acc diff per seed 3/4/5\tSD(seeds)")
for a, b in PAIRS:
    r = {k: E._boot_diff(per[a], per[b], k) for k in KEYS}
    ps = []
    for s in sorted(per_seed[a]):
        ma = statistics.mean(m["final_acc"] for m in per_seed[a][s]); mb = statistics.mean(m["final_acc"] for m in per_seed[b][s]); ps.append(ma - mb)
    r["final_acc_diff_per_seed"] = ps; r["final_acc_diff_sd_seeds"] = statistics.pstdev(ps); out["%s vs %s" % (a, b)] = r
    print("\t".join([a, b] + ["%+.3f [%+.3f, %+.3f]" % (r[k]["diff"], r[k]["ci95"][0], r[k]["ci95"][1]) for k in KEYS] + ["/".join("%+.3f" % x for x in ps), "%.3f" % r["final_acc_diff_sd_seeds"]]))
json.dump(out, open("/path/to/runs/%s_paired_extra.json" % tag, "w"), indent=1)
