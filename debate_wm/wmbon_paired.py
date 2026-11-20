"""Paired comparison of the WM-BoN run with the single-sample baselines of exp9_v6sd_ours_test_9b (same tasks, seeds, seats,
honest sampling seeds). Question-level paired bootstrap (2,000 resamples over the 400 questions, seeds averaged per question),
as eval_debate.summarize does within one run.   python3 wmbon_paired.py BON_RUN BASE_RUN > out.json"""
import json, random, sys
sys.path.insert(0, "/path/to/consensus-attacker")
from debate_ppo import eval_debate as E

bon, base = sys.argv[1], sys.argv[2]
KEYS = ["final_acc", "outcome_wrong", "majority_wrong", "all_wrong", "all_wrong_consensus", "majority_on_target", "change_rate",
        "attacker_arg_words", "pgold_r2", "pgold_r3", "pgold_r4"]


def load(run):
    st = json.load(open(run + "/state.json"))
    per = {}
    for d in st["debates"]:
        if d["final"] is None:
            continue
        per.setdefault(d["cond"], {}).setdefault(d["task_id"], []).append(E.debate_metrics(d))
    return per


A, B = load(bon), load(base)
out = {"conditions": {}, "paired": {}}
for c, by_q in list(A.items()) + list(B.items()):
    out["conditions"][c] = {"n_debates": sum(len(v) for v in by_q.values()), "n_questions": len(by_q),
                            **{k: E._boot_mean(by_q, k)["mean"] for k in KEYS if all(k in m for ms in by_q.values() for m in ms)}}
for a in A:
    for b in ("targeted_model_rl_sd120", "none"):
        out["paired"][f"{a} - {b}"] = {k: E._boot_diff(A[a], B[b], k) for k in KEYS if k in out["conditions"][a] and k in out["conditions"][b]}
if len(A) == 2:
    a1, a2 = sorted(A)
    out["paired"][f"{a2} - {a1}"] = {k: E._boot_diff(A[a2], A[a1], k) for k in KEYS if k in out["conditions"][a1]}
# WM-BoN bookkeeping: candidate validity, how often the pick differs from the first sample, score spread
st = json.load(open(bon + "/state.json"))
for c in A:
    ds = [d for d in st["debates"] if d["cond"] == c and d.get("wmbon")]
    picks = [x for d in ds for x in d["wmbon"]]
    out["conditions"][c]["bon"] = {"turns": len(picks), "mean_valid_cands": sum(sum(x["valid"]) for x in picks) / len(picks),
                                   "pick_is_first_sample": sum(x["best"] == 0 for x in picks) / len(picks),
                                   "mean_score_pick": sum(x["scores"][x["best"]] for x in picks if x["scores"][x["best"]] is not None) / len(picks),
                                   "mean_score_all": sum(sum(s for s in x["scores"] if s is not None) / len(x["scores"]) for x in picks) / len(picks)}
json.dump(out, sys.stdout, indent=1)
