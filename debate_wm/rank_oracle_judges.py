#!/usr/bin/env python
"""Re-check of the rank_oracle_exp10 ranking experiment with the rankers of the authors' wm_judge_eval study (2026-10-31, RTX 5090).

Data = the Lambda run rank_oracle_exp10 unchanged (states.jsonl, cands_*.jsonl, oracle.jsonl = Qwen3.5-9B K=4 post-argument
readouts, wm.jsonl = DCM PPOBeliefWM scores, modes tb / prior). For every candidate each judge gets the three honest
listeners' round-(r+1) chat with the candidate in the 4th seat -- built exactly as rank_oracle.cmd_oracle builds the
oracle's input (make_data.scoring_state + rank_oracle._reply, dprotocol.chat under profile set v1nn) -- and the score is
the mean predicted P(target) over the three listeners. Judges are debate_ppo/wm_judge_eval/judges.run_judge, a verbatim
copy of the local study's judges.py (sha256 2a52b336...):
  wm  / roleplay_direct   debate-wm-v1, persona chat, letter read after 'Answer:' prefill   ("pre" readout)
  wm  / roleplay_gen      debate-wm-v1, writes its own argument (T 0.7, <=160 tok) then reads  ("post" readout)
  q4b / observer_v2       untrained Qwen3.5-4B zero-shot prompted judge
  q9b / observer_v2       Qwen3.5-9B zero-shot prompted judge (white-box upper bound)
  q9b / roleplay_direct   Qwen3.5-9B itself, prefill readout ("pre" readout of the listener model)
  q9b / roleplay_gen      Qwen3.5-9B re-sampled once (one more "post" rollout; independent seeds from the oracle's)

    gpu stage (insider env):  --out R --model PATH --tag wm --judges roleplay_direct,roleplay_gen
    analysis (CPU):           --out R --analyze [--res DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.strat_dpo import make_data as MD  # noqa: E402
from debate_wm import rank_oracle as RO  # noqa: E402

PERS = ["dpo", "ppo_c120", "dpo_ppo_sd120", "base_q3_4b"]
LABEL = {"dpo": "pure DPO (strat-dpo-q3-4b)", "ppo_c120": "PPO c120", "dpo_ppo_sd120": "DPO+PPO sd120",
         "base_q3_4b": "plain Qwen3-4B"}


def records(out: Path):
    D.set_profile_set("v1nn")
    states = {s["state_id"]: s for s in RO._jsonl(out / "states.jsonl")}
    recs = []
    for c in RO._all_cands(out):
        s = states[c["state_id"]]
        sst = MD.scoring_state(s, RO._reply(s, c))
        for ag in D.honest_seats(s["seat"]):
            recs.append({"id": f"{c['persuader']}|{c['state_id']}|{c['cand']}|a{ag}", "messages": D.chat(sst, ag),
                         "n_options": s["n_options"],
                         "meta": {"round": s["round"] + 2, "prev_answer": s["history"][s["round"]][ag]["answer"],
                                  "target": s["target"], "persona": s["personalities"][ag]}})
    return recs


def stage_gpu(a):
    from debate_ppo.engines import make_vllm_engine
    from debate_ppo.wm_judge_eval.judges import run_judge
    res_dir = Path(a.res)
    res_dir.mkdir(parents=True, exist_ok=True)
    recs = records(Path(a.out))
    if a.limit:
        recs = recs[: a.limit]
    print(f"{len(recs)} listener turns", flush=True)
    t0 = time.time()
    eng = make_vllm_engine(a.model, gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=128)
    load_s = time.time() - t0
    for judge in a.judges.split(","):
        f = res_dir / f"{a.tag}__{judge}.jsonl"
        if f.exists():
            print(f"{f} exists, skip", flush=True)
            continue
        r = run_judge(eng, recs, judge)
        with open(f.with_suffix(".tmp"), "w") as g:
            for rec, p in zip(recs, r["preds"]):
                g.write(json.dumps({"id": rec["id"], "probs": p["probs"],
                                    "p_target": p["probs"][D.LETTERS.index(rec["meta"]["target"])]}) + "\n")
        f.with_suffix(".tmp").rename(f)
        json.dump({"model": a.model, "judge": judge, "load_s": load_s, "timing": r["timing"]},
                  open(res_dir / f"{a.tag}__{judge}.timing.json", "w"), indent=1)
        print(judge, json.dumps(r["timing"]), flush=True)


# ---------------------------------------------------------------- analysis
def per_state(cs, score):
    """cs: 8 candidates with ptarget_k (K=4 oracle rollouts); score: ranker scores. Same definitions as rank_oracle."""
    o = [st.mean(c["ptarget_k"]) for c in cs]
    hA = [st.mean(c["ptarget_k"][:2]) for c in cs]
    hB = [st.mean(c["ptarget_k"][2:]) for c in cs]
    n = len(cs)

    def sel_gain(sc, ev):
        j = max(range(n), key=lambda i: sc[i])
        return ev[j] - st.mean(ev)
    pick = max(range(n), key=lambda i: score[i])
    return {"sp": RO.spearman(score, o), "sp_ceiling": RO.spearman(hA, hB),
            "top1": float(pick == max(range(n), key=lambda i: o[i])),
            "oracle_gain": (sel_gain(hA, hB) + sel_gain(hB, hA)) / 2,
            "gain": (sel_gain(score, hB) + sel_gain(score, hA)) / 2,
            "pick_ptarget": o[pick], "mean_ptarget": st.mean(o)}


def boot_idx(n, B, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(B)]


def ci(vals):
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def sb(r):            # Spearman-Brown: reliability of the K=4 mean from the split-half (2 vs 2) correlation
    return 2 * r / (1 + r) if r > 0 else float("nan")


def stage_analyze(a):
    out, res_dir = Path(a.out), Path(a.res)
    orc = {(r["persuader"], r["state_id"], r["cand"]): r for r in RO._jsonl(out / "oracle.jsonl")}
    scores = collections.defaultdict(dict)                  # ranker -> key -> score
    for r in RO._jsonl(out / "wm.jsonl"):
        scores[f"DCM_{r['mode']}"][(r["persuader"], r["state_id"], r["cand"])] = r["wm_ptarget"]
    for f in sorted(res_dir.glob("*__*.jsonl")):
        per = collections.defaultdict(list)
        for l in open(f):
            p = json.loads(l)
            pe, rest = p["id"].split("|", 1)              # state_id itself contains '|'
            sid, c, _ = rest.rsplit("|", 2)
            per[(pe, sid, int(c))].append(p["p_target"])
        assert all(len(v) == 3 for v in per.values()), f
        scores[f.stem] = {k: st.mean(v) for k, v in per.items()}
    groups = collections.defaultdict(list)
    for c in RO._all_cands(out):
        k = (c["persuader"], c["state_id"], c["cand"])
        if k in orc and all(k in s for s in scores.values()):
            groups[(c["persuader"], c["state_id"])].append({**c, **orc[k], "key": k})
    groups = {k: sorted(v, key=lambda c: c["cand"]) for k, v in groups.items() if len(v) == RO.N_CAND}
    common = sorted(set.intersection(*[{sid for (p, sid) in groups if p == pe} for pe in PERS]))
    rankers = ["DCM_tb", "DCM_prior"] + [n for n in ["wm__roleplay_direct", "wm__roleplay_gen", "q4b__observer_v2",
                                                       "q9b__observer_v2", "q9b__roleplay_direct", "q9b__roleplay_gen"]
                                         if n in scores]
    names = {"DCM_tb": "DCM WM_post, tb (true beliefs; existing)", "DCM_prior": "DCM WM_post, prior (existing)",
             "wm__roleplay_direct": "debate-wm-v1 direct readout (pre)", "wm__roleplay_gen": "debate-wm-v1 write-then-read (post)",
             "q4b__observer_v2": "untrained 4B observer v2", "q9b__observer_v2": "9B observer v2 (white-box)",
             "q9b__roleplay_direct": "9B direct readout (pre, white-box)", "q9b__roleplay_gen": "9B re-sample once (post, white-box)"}
    M = {rk: {pe: {sid: per_state(groups[(pe, sid)], [scores[rk][c["key"]] for c in groups[(pe, sid)]]) for sid in common}
              for pe in PERS} for rk in rankers}
    B = a.n_boot
    bi = boot_idx(len(common), B)
    res = {"n_states": len(common), "n_boot": B, "rankers": {}, "paired": {}}
    L = []
    pr = L.append
    pr(f"# rank_oracle_exp10 re-check with wm_judge_eval rankers ({len(common)} common states x 8 candidates x 4 persuaders)\n")
    pr("Oracle = Qwen3.5-9B honest listeners, K=4 rollouts (argue then read, T 0.7), mean P(target) over the 3 listeners "
       "(Lambda, unchanged). Spearman = within-state Spearman(ranker, K=4 oracle mean), mean over states. "
       "Noise-corrected Spearman = mean Spearman / sqrt(SB(ceiling)), SB(ρ) = 2ρ/(1+ρ) the Spearman-Brown reliability of "
       "the 4-rollout mean from the split-half (2 vs 2) ceiling. Top-1 = ranker's pick is the K=4 oracle argmax (chance 0.125). "
       "Shares above 1 are possible: the split-half oracle selects on 2 noisy rollouts, a ranker selects on its own score. "
       "Held-out gain = P(target) of the ranker's pick minus the state mean, evaluated on each oracle half (A, B) and averaged; "
       "share = held-out gain / split-half oracle gain (select on one half, evaluate on the other). 95% CIs: bootstrap over "
       f"states ({B}).\n")
    for pe in PERS:
        pr(f"## {LABEL[pe]} (`{pe}`)\n")
        ceil = st.mean(M[rankers[0]][pe][s]["sp_ceiling"] for s in common)
        og = [M[rankers[0]][pe][s]["oracle_gain"] for s in common]
        pr(f"noise ceiling Spearman(half A, half B) = {ceil:.3f} → reliability of K=4 mean SB = {sb(ceil):.3f}; "
           f"split-half oracle gain = {st.mean(og):+.3f}; oracle mean P(target) = "
           f"{st.mean(M[rankers[0]][pe][s]['mean_ptarget'] for s in common):.3f}\n")
        pr("| ranker | Spearman [95% CI] | noise-corrected Spearman | top-1 hit | held-out gain [95% CI] | share of oracle gain [95% CI] | pick P(target) |")
        pr("|---|---|---|---|---|---|---|")
        res["rankers"][pe] = {}
        for rk in rankers:
            m = M[rk][pe]
            sp = [m[s]["sp"] for s in common]
            g = [m[s]["gain"] for s in common]
            ogs = [m[s]["oracle_gain"] for s in common]
            spb = [st.mean(sp[i] for i in ix) for ix in bi]
            gb = [st.mean(g[i] for i in ix) for ix in bi]
            shb = [sum(g[i] for i in ix) / sum(ogs[i] for i in ix) for ix in bi]
            share = sum(g) / sum(ogs)
            row = {"spearman": st.mean(sp), "spearman_ci": ci(spb), "spearman_nc": st.mean(sp) / math.sqrt(sb(ceil)),
                   "top1": st.mean(m[s]["top1"] for s in common), "gain": st.mean(g), "gain_ci": ci(gb),
                   "share": share, "share_ci": ci(shb), "pick_ptarget": st.mean(m[s]["pick_ptarget"] for s in common),
                   "ceiling": ceil, "oracle_gain": st.mean(ogs)}
            res["rankers"][pe][rk] = row
            pr(f"| {names[rk]} | {row['spearman']:.3f} [{row['spearman_ci'][0]:.3f}, {row['spearman_ci'][1]:.3f}] | "
               f"{row['spearman_nc']:.3f} | {row['top1']:.3f} | {row['gain']:+.3f} [{row['gain_ci'][0]:+.3f}, {row['gain_ci'][1]:+.3f}] | "
               f"{row['share']:.2f} [{row['share_ci'][0]:.2f}, {row['share_ci'][1]:.2f}] | {row['pick_ptarget']:.3f} |")
        pr("")
    # paired: debate-wm-v1 direct - DCM (per persuader)
    if "wm__roleplay_direct" in rankers:
        pr("## Paired: debate-wm-v1 direct readout − DCM (same states, bootstrap over states)\n")
        pr("| persuader | vs | ΔSpearman [95% CI] | Δtop-1 [95% CI] | Δheld-out gain [95% CI] | Δshare of oracle gain [95% CI] |")
        pr("|---|---|---|---|---|---|")
        for pe in PERS:
            for dcm in ("DCM_tb", "DCM_prior"):
                a_, b_ = M["wm__roleplay_direct"][pe], M[dcm][pe]
                cells, rec = [], {}
                for k in ("sp", "top1", "gain"):
                    d = [a_[s][k] - b_[s][k] for s in common]
                    lo, hi = ci([st.mean(d[i] for i in ix) for ix in bi])
                    rec[k] = (st.mean(d), lo, hi)
                    cells.append(f"{st.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
                ga = [a_[s]["gain"] for s in common]; gb_ = [b_[s]["gain"] for s in common]
                ogs = [a_[s]["oracle_gain"] for s in common]
                ds = [(sum(ga[i] for i in ix) - sum(gb_[i] for i in ix)) / sum(ogs[i] for i in ix) for ix in bi]
                dsh = (sum(ga) - sum(gb_)) / sum(ogs)
                rec["share"] = (dsh, *ci(ds))
                cells.append(f"{dsh:+.2f} [{ci(ds)[0]:+.2f}, {ci(ds)[1]:+.2f}]")
                res["paired"][f"wm_direct - {dcm} | {pe}"] = rec
                pr(f"| {LABEL[pe]} | {dcm} | " + " | ".join(cells) + " |")
        pr("")
    # sd120 - c120 under the best deployable ranker (deployable = no 9B access; chosen by mean share over persuaders)
    dep = [rk for rk in rankers if rk.startswith(("wm__", "q4b__", "DCM_prior"))]
    best = max(dep, key=lambda rk: st.mean(res["rankers"][pe][rk]["share"] for pe in PERS))
    res["best_deployable"] = best
    pr(f"## Paired: DPO+PPO sd120 − PPO c120 under the best deployable ranker = {names[best]}\n")
    pr("(best deployable = highest mean share of oracle gain over the 4 persuaders among rankers that need no 9B access)\n")
    pr("| quantity | sd120 | c120 | Δ sd120 − c120 [95% CI] |")
    pr("|---|---|---|---|")
    x, y = M[best]["dpo_ppo_sd120"], M[best]["ppo_c120"]
    for k, nm in (("pick_ptarget", "9B oracle P(target) of the ranker's pick"), ("mean_ptarget", "9B oracle P(target), random candidate"),
                  ("gain", "held-out gain of the pick over random"), ("oracle_gain", "split-half oracle gain (what is there to pick)"),
                  ("sp", "within-state Spearman(ranker, 9B)"), ("top1", "top-1 hit")):
        d = [x[s][k] - y[s][k] for s in common]
        lo, hi = ci([st.mean(d[i] for i in ix) for ix in bi])
        res["paired"][f"sd120 - c120 | {best} | {k}"] = (st.mean(d), lo, hi)
        pr(f"| {nm} | {st.mean(x[s][k] for s in common):.3f} | {st.mean(y[s][k] for s in common):.3f} | {st.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    pr("")
    cost = {f.stem: json.load(open(f)) for f in sorted(res_dir.glob("*.timing.json"))}
    res["cost"] = cost
    pr("Cost (one batch of all listener turns, vLLM 0.23, bf16, GPU of the run): " + "; ".join(
        f"{k.removesuffix('.timing')} {v['timing']['s_per_turn']:.3f} s/turn" for k, v in cost.items()) + "\n")
    json.dump(res, open(res_dir / "rank_results.json", "w"), indent=1)
    (res_dir / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="rank_oracle_exp10 dir (read only)")
    ap.add_argument("--res", default=None, help="where to write judge scores / results (default <out>_judges)")
    ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--judges", default="roleplay_direct")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N listener turns only")
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.res = a.res or str(Path(a.out).with_name(Path(a.out).name + "_judges"))
    stage_analyze(a) if a.analyze else stage_gpu(a)


if __name__ == "__main__":
    main()
