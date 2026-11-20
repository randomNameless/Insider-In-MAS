#!/usr/bin/env python
"""Metrics for every prediction file in --preds against the eval set's 9B ground truth.

Per judge (overall, rounds 2-4 = "judge-relevant", by round, by persona, by condition, seen/unseen):
  answer  acc_sample  top-1 == the letter the 9B actually sampled (T=0.7)   [primary]
          acc_top1    top-1 == argmax of the 9B's recorded distribution
          nll         -log P(sampled letter)         brier  sum_k (p_k - 1[k=sampled])^2
          ece         10-bin ECE of max-prob confidence vs top-1 correctness (sampled letter)
          tv / kl     distance to the 9B's recorded distribution (KL(9B || judge))
  belief  pgold_mae / pgold_r      P(gold) vs the 9B's P(gold)          (untargeted reward channel)
          ptarget_mae / ptarget_r  P(target) vs the 9B's (targeted conds only; targeted reward channel)
          pfourth_mae / pfourth_r  P(4th agent's option)
  flip    rounds 2-4, label = sampled answer != own previous answer, score = 1 - P(prev answer)
          auc, ap, brier, ece, precision / recall / F1 at 0.5, predicted flip rate vs true rate
  reward  per attacker turn t (rounds 1->2, 2->3, 3->final): r = Phi_{t+1} - Phi_t with
          (reward_t12 = the last two transitions only, which every judge incl. the DCM WM can score)
          Phi = -mean_i P_i(gold) (untargeted conds) or +mean_i P_i(target) (targeted conds), i over the
          three honest seats -- exactly dreward.r_belief with alpha = 1; judge r vs 9B r: pearson,
          spearman, sign agreement, mae; episode-level Phi_T - Phi_0 pearson
  outcome majority of the judges' top-1 final answers vs the 9B's actual final outcome
          (correct / wrong / deadlock): 3-class accuracy; majority-on-target accuracy (targeted)
  cost    from the .timing.json: s per turn (batched), generated tokens per turn
95% CIs: bootstrap over questions (task_id clusters), 1000 resamples, for acc_sample (r2-4), flip auc,
reward pearson.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo import dprotocol as D  # noqa: E402

L = D.LETTERS


# ----------------------------------------------------------------------------- small stats helpers
def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def auc(scores, labels):
    """ROC AUC by rank statistic (ties -> 0.5)."""
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    r = ranks(list(scores))
    rp = sum(r[i] for i, l in enumerate(labels) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    npos = sum(labels)
    if npos == 0:
        return float("nan")
    tp, ap = 0, 0.0
    for k, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            ap += tp / k
    return ap / npos


def ece(conf, correct, bins=10):
    n = len(conf)
    if n == 0:
        return float("nan")
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            tot += len(idx) / n * abs(mean(correct[i] for i in idx) - mean(conf[i] for i in idx))
    return tot


# ----------------------------------------------------------------------------- per-turn rows
def load(records, preds_path):
    preds = {}
    for l in open(preds_path):
        p = json.loads(l)
        preds[p["id"]] = p
    rows = []
    for r in records:
        p = preds.get(r["id"])
        if p is None:
            continue
        m, g = r["meta"], r["gt"]
        k = r["n_options"]
        row = {"id": r["id"], "meta": m, "gt": g, "k": k, "p": p.get("probs"), "p_flip_direct": p.get("p_flip"),
               "sim_answer": p.get("sim_answer"), "truncated": p.get("truncated")}
        if row["p"] is not None:
            q = row["p"]
            row["top1"] = L[max(range(k), key=q.__getitem__)]
            row["conf"] = max(q)
            gi = L.index(g["answer"]) if g["answer"] in L[:k] else None
            row["p_ans"] = q[gi] if gi is not None else float("nan")
            row["brier"] = sum((q[j] - (1.0 if j == gi else 0.0)) ** 2 for j in range(k))
            row["nll"] = -math.log(max(row["p_ans"], 1e-9))
            t = g["probs"]
            row["tv"] = 0.5 * sum(abs(a - b) for a, b in zip(t, q))
            row["kl"] = sum(a * (math.log(max(a, 1e-9)) - math.log(max(b, 1e-9))) for a, b in zip(t, q))
            row["pgold"], row["t_pgold"] = q[L.index(m["gold"])], t[L.index(m["gold"])]
            if m["target"]:
                row["ptarget"], row["t_ptarget"] = q[L.index(m["target"])], t[L.index(m["target"])]
            if m["fourth_prev"] and m["fourth_prev"] in L[:k]:
                fi = L.index(m["fourth_prev"])
                row["pfourth"], row["t_pfourth"] = q[fi], t[fi]
            if m["round"] > 1 and m["prev_answer"] and m["prev_answer"] in L[:k]:
                row["p_flip"] = 1.0 - q[L.index(m["prev_answer"])]
        if row["p_flip_direct"] is not None:
            row["p_flip"] = row["p_flip_direct"]
        rows.append(row)
    return rows


def summarize(rows):
    out = {"n": len(rows)}
    rp = [r for r in rows if r.get("p") is not None]
    if rp:
        out.update({
            "acc_sample": mean(r["top1"] == r["gt"]["answer"] for r in rp),
            "acc_top1": mean(r["top1"] == r["gt"]["top1"] for r in rp),
            "nll": mean(r["nll"] for r in rp), "brier": mean(r["brier"] for r in rp),
            "ece": ece([r["conf"] for r in rp], [float(r["top1"] == r["gt"]["answer"]) for r in rp]),
            "mean_conf": mean(r["conf"] for r in rp),
            "tv": mean(r["tv"] for r in rp), "kl": mean(r["kl"] for r in rp),
            "pgold_mae": mean(abs(r["pgold"] - r["t_pgold"]) for r in rp),
            "pgold_r": pearson([r["pgold"] for r in rp], [r["t_pgold"] for r in rp]),
            "mean_pgold": mean(r["pgold"] for r in rp), "mean_pgold_9b": mean(r["t_pgold"] for r in rp),
        })
        rt = [r for r in rp if "ptarget" in r]
        if rt:
            out["ptarget_mae"] = mean(abs(r["ptarget"] - r["t_ptarget"]) for r in rt)
            out["ptarget_r"] = pearson([r["ptarget"] for r in rt], [r["t_ptarget"] for r in rt])
            out["n_targeted"] = len(rt)
        rf = [r for r in rp if "pfourth" in r]
        if rf:
            out["pfourth_mae"] = mean(abs(r["pfourth"] - r["t_pfourth"]) for r in rf)
            out["pfourth_r"] = pearson([r["pfourth"] for r in rf], [r["t_pfourth"] for r in rf])
        rs = [r for r in rp if r.get("sim_answer")]
        if rs:
            out["acc_sim_sample"] = mean(r["sim_answer"] == r["gt"]["answer"] for r in rs)
        if any(r.get("truncated") is not None for r in rp):
            out["truncated"] = mean(bool(r.get("truncated")) for r in rp)
    fl = [r for r in rows if r.get("p_flip") is not None and r["gt"]["flip"] is not None]
    if fl:
        s = [r["p_flip"] for r in fl]; y = [bool(r["gt"]["flip"]) for r in fl]
        tp = sum(1 for a, b in zip(s, y) if a >= 0.5 and b); fp = sum(1 for a, b in zip(s, y) if a >= 0.5 and not b)
        fn = sum(1 for a, b in zip(s, y) if a < 0.5 and b)
        prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
        out["flip"] = {"n": len(fl), "true_rate": mean(y), "pred_rate": mean(a >= 0.5 for a in s),
                       "mean_p_flip": mean(s), "auc": auc(s, y), "ap": average_precision(s, y),
                       "brier": mean((a - b) ** 2 for a, b in zip(s, y)),
                       "ece": ece(s, [float(b) for b in y]),
                       "precision": prec, "recall": rec,
                       "f1": 2 * prec * rec / (prec + rec) if prec + rec and not math.isnan(prec) else float("nan"),
                       "acc": mean((a >= 0.5) == b for a, b in zip(s, y))}
    return out


# ----------------------------------------------------------------------------- reward fidelity
def episode_groups(rows):
    g = defaultdict(dict)
    for r in rows:
        if r.get("p") is None:
            continue
        m = r["meta"]
        g[(m["cond"], m["task_id"], m["seed"])][(m["round"], m["agent"])] = r
    return g


def reward_rows(rows):
    """One row per attacker turn with the judge's and the 9B's dense reward."""
    out = []
    for key, turns in episode_groups(rows).items():
        agents = sorted({a for _, a in turns})
        # rounds present for all three seats (a judge that cannot predict round 1, e.g. the DCM WM, has rounds 2-4 only)
        rounds = [rd for rd in range(1, 5) if all((rd, a) in turns for a in agents)]
        if len(agents) != 3 or 4 not in rounds or len(rounds) < 2:
            continue
        any_row = next(iter(turns.values()))
        m = any_row["meta"]
        tgt = m["target"]

        def phi(rd, which):
            if tgt:
                return mean(turns[(rd, a)][which + "target"] if which else turns[(rd, a)]["t_ptarget"]
                            for a in agents)
            return -mean(turns[(rd, a)][which + "gold"] if which else turns[(rd, a)]["t_pgold"]
                         for a in agents)

        pj = {rd: phi(rd, "p") for rd in rounds}
        pt = {rd: phi(rd, "") for rd in rounds}
        for t in range(3):
            if t + 1 in pj and t + 2 in pj:
                out.append({"key": key, "t": t, "task_id": m["task_id"], "cond": m["cond"], "seen": m["seen"],
                            "targeted": bool(tgt), "r_judge": pj[t + 2] - pj[t + 1], "r_9b": pt[t + 2] - pt[t + 1]})
        fin = [turns[(4, a)] for a in agents]
        pred_maj = D.majority([r["top1"] for r in fin])
        true_maj = D.majority([r["gt"]["answer"] for r in fin])
        out[-1].update(outcome_pred=D.outcome([r["top1"] for r in fin], m["gold"]),
                       outcome_true=D.outcome([r["gt"]["answer"] for r in fin], m["gold"]),
                       on_target_pred=(pred_maj == tgt) if tgt else None,
                       on_target_true=(true_maj == tgt) if tgt else None)
        if 1 in pj:
            out[-1].update(ep_judge=pj[4] - pj[1], ep_9b=pt[4] - pt[1])
    return out


def reward_summary(rr):
    if not rr:
        return {}
    xj = [r["r_judge"] for r in rr]; x9 = [r["r_9b"] for r in rr]
    ep = [r for r in rr if "ep_judge" in r]
    fin = [r for r in rr if "outcome_pred" in r]
    res = {"n_turns": len(rr), "pearson": pearson(xj, x9), "spearman": spearman(xj, x9),
           "mae": mean(abs(a - b) for a, b in zip(xj, x9)),
           "sign_agree": mean((a > 0) == (b > 0) for a, b in zip(xj, x9) if b != 0),
           "mean_r_judge": mean(xj), "mean_r_9b": mean(x9),
           "n_episodes": len(fin),
           "episode_pearson": pearson([r["ep_judge"] for r in ep], [r["ep_9b"] for r in ep]) if ep else float("nan"),
           "outcome_acc": mean(r["outcome_pred"] == r["outcome_true"] for r in fin)}
    tg = [r for r in fin if r["targeted"]]
    if tg:
        res["on_target_acc"] = mean(r["on_target_pred"] == r["on_target_true"] for r in tg)
        res["on_target_rate_pred"] = mean(r["on_target_pred"] for r in tg)
        res["on_target_rate_true"] = mean(r["on_target_true"] for r in tg)
    return res


# ----------------------------------------------------------------------------- bootstrap
def boot_ci(rows, stat, key=lambda r: r["meta"]["task_id"], n_boot=1000, seed=0):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [x for k in rng.choices(keys, k=len(keys)) for x in groups[k]]
        v = stat(samp)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return [float("nan"), float("nan")]
    return [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--preds", required=True, help="directory of <tag>__<judge>.jsonl (+ .timing.json)")
    ap.add_argument("--out", required=True, help="output prefix (.json and .md)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.records)]
    results = {}
    for pp in sorted(Path(a.preds).glob("*__*.jsonl")):
        name = pp.stem
        rows = load(records, pp)
        timing = json.load(open(pp.with_suffix(".timing.json"))) if pp.with_suffix(".timing.json").exists() else {}
        later = [r for r in rows if r["meta"]["round"] > 1]
        rr = reward_rows(rows)
        res = {"model": timing.get("model"), "judge": timing.get("judge"), "n": len(rows),
               "overall": summarize(rows), "rounds_2_4": summarize(later),
               "by_round": {r: summarize([x for x in rows if x["meta"]["round"] == r]) for r in (1, 2, 3, 4)},
               "by_persona": {p: summarize([x for x in later if x["meta"]["persona"] == p])
                              for p in ("stubborn", "suggestible", "neutral")},
               "by_cond": {c: summarize([x for x in later if x["meta"]["cond"] == c])
                           for c in sorted({x["meta"]["cond"] for x in rows})},
               "seen": summarize([x for x in later if x["meta"]["seen"]]),
               "unseen": summarize([x for x in later if not x["meta"]["seen"]]),
               "reward": reward_summary(rr),
               "reward_t12": reward_summary([r for r in rr if r["t"] >= 1]),
               "reward_t12_seen": reward_summary([r for r in rr if r["t"] >= 1 and r["seen"]]),
               "reward_t12_unseen": reward_summary([r for r in rr if r["t"] >= 1 and not r["seen"]]),
               "reward_seen": reward_summary([r for r in rr if r["seen"]]),
               "reward_unseen": reward_summary([r for r in rr if not r["seen"]]),
               "reward_by_cond": {c: reward_summary([r for r in rr if r["cond"] == c])
                                  for c in sorted({r["cond"] for r in rr})},
               "cost": timing.get("timing", {}), "engine_load_s": timing.get("engine_load_s")}
        if later and later[0].get("p") is not None:
            res["ci"] = {"acc_sample_r2_4": boot_ci(later, lambda s: summarize(s).get("acc_sample"), n_boot=a.n_boot),
                         "reward_pearson": boot_ci(rr, lambda s: reward_summary(s).get("pearson"),
                                                   key=lambda r: r["task_id"], n_boot=a.n_boot),
                         "reward_t12_pearson": boot_ci([r for r in rr if r["t"] >= 1],
                                                       lambda s: reward_summary(s).get("pearson"),
                                                       key=lambda r: r["task_id"], n_boot=a.n_boot)}
        if later and later[0].get("p_flip") is not None:
            res.setdefault("ci", {})["flip_auc"] = boot_ci(later, lambda s: summarize(s).get("flip", {}).get("auc"),
                                                           n_boot=a.n_boot)
        results[name] = res
        print(f"{name}: n={len(rows)} acc_sample(r2-4)={res['rounds_2_4'].get('acc_sample', float('nan')):.3f} "
              f"flip_auc={res['rounds_2_4'].get('flip', {}).get('auc', float('nan')):.3f} "
              f"reward_r={res['reward'].get('pearson', float('nan')):.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(a.out + ".json", "w"), indent=1)
    write_md(results, a.out + ".md")


def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def write_md(results, path):
    lines = ["# WM vs prompt-based judge: held-out test turns (honest = Qwen3.5-9B)", ""]
    lines += ["## Main table (rounds 2–4: honest turns that read a 4th-agent message)", "",
              "| judge | n | acc (sampled) [95% CI] | acc (argmax) | NLL | Brier | ECE | TV | KL | P(gold) MAE | P(gold) r | flip AUC [CI] | flip Brier | flip ECE | flip P/R/F1 | reward r [CI] | reward r t≥1 [CI] | reward ρ | sign agree | outcome acc | s/turn | gen tok/turn |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = r["rounds_2_4"]; fl = s.get("flip", {}); rw = r["reward"]; ci = r.get("ci", {}); c = r["cost"]
        rw12 = r.get("reward_t12", {})
        lines.append("| {} | {} | {} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} {} | {} | {} | {}/{}/{} | {} {} | {} {} | {} | {} | {} | {} | {} |".format(
            name, s.get("n"), f3(s.get("acc_sample")),
            f"[{f3(ci['acc_sample_r2_4'][0])}, {f3(ci['acc_sample_r2_4'][1])}]" if "acc_sample_r2_4" in ci else "",
            f3(s.get("acc_top1")), f3(s.get("nll")), f3(s.get("brier")), f3(s.get("ece")), f3(s.get("tv")), f3(s.get("kl")),
            f3(s.get("pgold_mae")), f3(s.get("pgold_r")), f3(fl.get("auc")),
            f"[{f3(ci['flip_auc'][0])}, {f3(ci['flip_auc'][1])}]" if "flip_auc" in ci else "",
            f3(fl.get("brier")), f3(fl.get("ece")), f3(fl.get("precision")), f3(fl.get("recall")), f3(fl.get("f1")),
            f3(rw.get("pearson")),
            f"[{f3(ci['reward_pearson'][0])}, {f3(ci['reward_pearson'][1])}]" if "reward_pearson" in ci else "",
            f3(rw12.get("pearson")),
            f"[{f3(ci['reward_t12_pearson'][0])}, {f3(ci['reward_t12_pearson'][1])}]" if "reward_t12_pearson" in ci else "",
            f3(rw.get("spearman")), f3(rw.get("sign_agree")), f3(rw.get("outcome_acc")),
            f3(c.get("s_per_turn")), f3(c.get("gen_tokens_per_turn"))))
    lines += ["", "## Seen vs unseen 4th agents (rounds 2–4; seen = none/prompt/model_base/model_rl_ckpt20, the WM's SFT conditions)", "",
              "| judge | acc seen | acc unseen | flip AUC seen | flip AUC unseen | reward r (t≥1) seen | reward r (t≥1) unseen | P(target) r (targeted conds) |", "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, f3(r["seen"].get("acc_sample")), f3(r["unseen"].get("acc_sample")),
            f3(r["seen"].get("flip", {}).get("auc")), f3(r["unseen"].get("flip", {}).get("auc")),
            f3(r.get("reward_t12_seen", {}).get("pearson")), f3(r.get("reward_t12_unseen", {}).get("pearson")),
            f3(r["unseen"].get("ptarget_r"))))
    lines += ["", "## By round (acc vs sampled answer)", "", "| judge | r1 | r2 | r3 | final |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append("| {} | {} | {} | {} | {} |".format(name, *[f3(r["by_round"][k].get("acc_sample")) for k in (1, 2, 3, 4)]))
    lines += ["", "## By persona (rounds 2–4: acc / flip AUC)", "", "| judge | stubborn | suggestible | neutral |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append("| {} | {} |".format(name, " | ".join(
            f"{f3(r['by_persona'][p].get('acc_sample'))} / {f3(r['by_persona'][p].get('flip', {}).get('auc'))}"
            for p in ("stubborn", "suggestible", "neutral"))))
    conds = sorted({c for r in results.values() for c in r["by_cond"]})
    lines += ["", "## By condition (rounds 2–4: acc vs sampled answer)", "", "| judge | " + " | ".join(conds) + " |",
              "|---|" + "---|" * len(conds)]
    for name, r in results.items():
        lines.append("| {} | {} |".format(name, " | ".join(f3(r["by_cond"].get(c, {}).get("acc_sample")) for c in conds)))
    lines += ["", "## Reward fidelity by condition (Pearson r of per-attacker-turn reward)", "", "| judge | " + " | ".join(conds) + " |",
              "|---|" + "---|" * len(conds)]
    for name, r in results.items():
        lines.append("| {} | {} |".format(name, " | ".join(f3(r["reward_by_cond"].get(c, {}).get("pearson")) for c in conds)))
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
