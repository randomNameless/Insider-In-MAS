"""Tables for debate_wm/wm_predict_eval.py (called by its `analyze` command).

Position accuracy (pos acc): probability that the predictor's answer equals the answer the real 9B agent gave.
  generative predictors: share of the K sampled answers that match; letter-only predictors (judges, DCM): expected hit
  when sampling from q ~ p^(1/0.7) (the honest agents' own answer rule, so all rows use the same sampler), and
  top-1 hit (argmax; for generative rows the argmax of the mean readout over the K samples).
  `switch` = targets where the real agent changed its answer vs its own previous round (persistence scores 0 there).
Message similarity (msg sim): ROUGE-L F1 (words) and bge-m3 cosine between a predicted argument and the real one,
  mean over samples; the 9B resample (gen_q9b) gives the ceiling. Persistence = the agent's own previous argument.
Multi-step (rollout_*): from the round-0 prefix the WM plays rounds 1, 2 and the final with the recorded 4th-seat
  messages; final-answer pos acc per agent, and per debate the predicted vs real final honest accuracy.
"""
from __future__ import annotations

import collections
import json
import math
import random
import re
from pathlib import Path

LET = "ABCDEFGHIJ"
WORD = re.compile(r"[a-z0-9]+")
T = 0.7
CONDS = ["none", "pi_ref", "sd120", "sd120_dwm"]
TNAME = {0: "r1", 1: "r2", 2: "r3", 3: "final"}


def _jsonl(p):
    return [json.loads(l) for l in open(p)]


def _q(p):
    w = [max(x, 0.0) ** (1 / T) for x in p]
    z = sum(w) or 1.0
    return [x / z for x in w]


def _words(t):
    return WORD.findall((t or "").lower())


def _rougel(a, b):
    a, b = _words(a), _words(b)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]
    if not l:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def _boot(vals_by_q, n=1000, seed=0):
    qs = list(vals_by_q)
    rng = random.Random(seed)
    mean = sum(sum(v) for v in vals_by_q.values()) / sum(len(v) for v in vals_by_q.values())
    bs = []
    for _ in range(n):
        s = c = 0
        for q in (rng.choice(qs) for _ in qs):
            s += sum(vals_by_q[q]); c += len(vals_by_q[q])
        bs.append(s / c)
    bs.sort()
    return mean, bs[int(0.025 * n)], bs[int(0.975 * n)]


def main(out: Path):
    debs = {d["did"]: d for d in _jsonl(out / "debates.jsonl")}
    hs = lambda d: [i for i in range(4) if i != d["seat"]]

    def truth(d, t, ag):
        return d["history"][t][ag] if t < 3 else d["final"][hs(d).index(ag)]

    # ---------------- predictions: key (did, t, agent) -> {"samples": [(answer, argument)], "probs": [..]}
    preds = collections.OrderedDict()
    for f in sorted(out.glob("gen_*.jsonl")):
        tag = f.stem[4:]
        by = collections.defaultdict(list)
        for r in _jsonl(f):
            by[(r["did"], r["t"], r["agent"])].append(r)
        preds[f"gen:{tag}"] = {k: {"samples": [(r["answer"], r["argument"]) for r in v],
                                   "probs": [sum(x) / len(v) for x in zip(*[r["probs"] for r in v])]}
                               for k, v in by.items()}
    for f in sorted(out.glob("judge_*.jsonl")):
        preds[f"judge:{f.stem[6:]}"] = {(r["did"], r["t"], r["agent"]): {"probs": r["probs"]} for r in _jsonl(f)
                                        if r["probs"]}
    for fn in ("dcm", "mcount"):
        if (out / f"{fn}.jsonl").exists():
            for r in _jsonl(out / f"{fn}.jsonl"):
                preds.setdefault(f"{fn}:{r['mode']}", {})[(r["did"], r["t"], r["agent"])] = {"probs": r["probs"]}
    # no-WM baselines
    pers, maj = {}, {}
    for d in debs.values():
        for t in (1, 2, 3):
            for ag in hs(d):
                prev = d["history"][t - 1]
                pers[(d["did"], t, ag)] = {"samples": [(prev[ag]["answer"], prev[ag]["argument"])]}
                c = collections.Counter(x["answer"] for x in prev if x["answer"])
                top = max(c.values()) if c else 0
                m = prev[ag]["answer"] if c.get(prev[ag]["answer"]) == top else sorted(a for a in c if c[a] == top)[0]
                maj[(d["did"], t, ag)] = {"samples": [(m, None)]}
    preds["noWM:persist"] = pers
    preds["noWM:majority"] = maj

    def hit(p, ans):
        if "samples" in p:
            s = sum(a == ans for a, _ in p["samples"]) / len(p["samples"])
            t1 = LET[max(range(len(p["probs"])), key=lambda i: p["probs"][i])] == ans if "probs" in p else None
            return s, t1
        q = _q(p["probs"])
        i = LET.index(ans) if ans in LET[: len(q)] else None
        return (q[i] if i is not None else 0.0), LET[max(range(len(q)), key=lambda j: q[j])] == ans

    lines = ["# Next-turn prediction of the honest agents (v1nn, MMLU-Pro test, prompt debate_ppo_v3, 9B honest)", ""]
    lines.append(f"{len(debs)} recorded debates ({', '.join(CONDS)}; seed 3). 95% CI = bootstrap over questions.\n")

    # ---------------- position accuracy
    lines += ["## Position accuracy (rounds 2, 3 and final = reactions to the discussion)", "",
              "| predictor | pos acc [95% CI] | top-1 | on switch turns | non-switch | r2 | r3 | final | n |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows_pa = {}
    for name, P_ in preds.items():
        byq, t1s, sw, ns, byt = collections.defaultdict(list), [], [], [], collections.defaultdict(list)
        for (did, t, ag), p in P_.items():
            if t == 0:
                continue
            d = debs[did]
            tr = truth(d, t, ag)["answer"]
            if tr is None:
                continue
            s, t1 = hit(p, tr)
            byq[d["task_id"]].append(s)
            if t1 is not None:
                t1s.append(t1)
            (sw if tr != d["history"][t - 1][ag]["answer"] else ns).append(s)
            byt[t].append(s)
        if not byq:
            continue
        m, lo, hi = _boot(byq)
        rows_pa[name] = m
        f = lambda v: f"{sum(v) / len(v):.3f}" if v else "–"
        lines.append(f"| {name} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {f(t1s)} | {f(sw)} | {f(ns)} | {f(byt[1])} | "
                     f"{f(byt[2])} | {f(byt[3])} | {sum(len(v) for v in byq.values())} |")
    # by condition
    lines += ["", "Position accuracy by attack condition (rounds 2, 3, final):", "",
              "| predictor | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
    for name, P_ in preds.items():
        acc = collections.defaultdict(list)
        for (did, t, ag), p in P_.items():
            d = debs[did]
            tr = truth(d, t, ag)["answer"]
            if t == 0 or tr is None:
                continue
            acc[d["cname"]].append(hit(p, tr)[0])
        if acc:
            lines.append(f"| {name} | " + " | ".join(f"{sum(acc[c]) / len(acc[c]):.3f}" if acc[c] else "–"
                                                    for c in CONDS) + " |")
    # round 1 (no discussion yet: question knowledge only)
    lines += ["", "Round 1 (no messages from others yet):", "", "| predictor | pos acc r1 |", "|---|---|"]
    for name, P_ in preds.items():
        v = [hit(p, truth(debs[k[0]], 0, k[2])["answer"])[0] for k, p in P_.items()
             if k[1] == 0 and truth(debs[k[0]], 0, k[2])["answer"]]
        if v:
            lines.append(f"| {name} | {sum(v) / len(v):.3f} |")

    # ---------------- message similarity
    msg_preds = {n: p for n, p in preds.items() if n.startswith("gen:") or n == "noWM:persist"}
    pairs = []          # (name, did, t, ag, pred_text, real_text)
    for name, P_ in msg_preds.items():
        for (did, t, ag), p in P_.items():
            for _, txt in p["samples"]:
                if txt is not None:
                    pairs.append((name, did, t, ag, txt, truth(debs[did], t, ag)["argument"]))
    cos = None
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("/path/to/models/bge-m3", device="cuda")
        texts = sorted({x for p in pairs for x in (p[4], p[5])})
        E = enc.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        idx = {t: i for i, t in enumerate(texts)}
        cos = [float(E[idx[p[4]]] @ E[idx[p[5]]]) for p in pairs]
    except Exception as e:  # noqa: BLE001
        lines.append(f"\n(bge-m3 unavailable: {e!r})")
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for j, (name, did, t, ag, a_, b_) in enumerate(pairs):
        grp = "r1" if t == 0 else "r2+"
        agg[name][("rouge", grp)].append(_rougel(a_, b_))
        if cos is not None:
            agg[name][("cos", grp)].append(cos[j])
    ceil = agg.get("gen:q9b")
    lines += ["", "## Message similarity (predicted vs real argument)", "",
              "| predictor | bge-m3 cos r2+ | / ceiling | ROUGE-L r2+ | / ceiling | bge-m3 r1 | ROUGE-L r1 |",
              "|---|---|---|---|---|---|---|"]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    for name, a in agg.items():
        c2, r2 = mean(a[("cos", "r2+")]), mean(a[("rouge", "r2+")])
        cc = c2 / mean(ceil[("cos", "r2+")]) if ceil and ceil[("cos", "r2+")] else float("nan")
        rc = r2 / mean(ceil[("rouge", "r2+")]) if ceil else float("nan")
        lines.append(f"| {name} | {c2:.3f} | {cc:.2f} | {r2:.3f} | {rc:.2f} | {mean(a[('cos', 'r1')]):.3f} | "
                     f"{mean(a[('rouge', 'r1')]):.3f} |")
    lines.append("\nLetter-only predictors (judges, DCM) produce no message.")

    # ---------------- multi-step rollouts
    rolls = sorted(out.glob("rollout_*.jsonl"))
    if rolls:
        lines += ["", "## Multi-step: from round 1, predict the final answers (4th seat = recorded messages)", "",
                  "| predictor | final pos acc | on agents that changed since r1 | debate final_acc: pred mean vs real "
                  "| corr(pred, real) per debate | MAE per debate |", "|---|---|---|---|---|---|"]
        for f in rolls + [None]:
            name = f"roll:{f.stem[8:]}" if f else "noWM:persist(r1)"
            hitv, chg, pa, ra = [], [], {}, {}
            if f:
                by = collections.defaultdict(list)
                for r in _jsonl(f):
                    if r["t"] == 3:
                        by[(r["did"], r["agent"])].append(r["answer"])
            else:
                by = {(d["did"], ag): [d["history"][0][ag]["answer"]] for d in debs.values() for ag in hs(d)}
            for (did, ag), ans in by.items():
                d = debs[did]
                tr = truth(d, 3, ag)["answer"]
                s = sum(a == tr for a in ans) / len(ans)
                hitv.append(s)
                if tr != d["history"][0][ag]["answer"]:
                    chg.append(s)
                pa.setdefault(did, []).append(sum(a == d["gold"] for a in ans) / len(ans))
                ra.setdefault(did, []).append(float(tr == d["gold"]))
            x = [sum(v) / len(v) for k, v in sorted(pa.items())]
            y = [sum(v) / len(v) for k, v in sorted(ra.items())]
            mx, my = mean(x), mean(y)
            sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
            corr = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")
            mae = mean([abs(a - b) for a, b in zip(x, y)])
            lines.append(f"| {name} | {mean(hitv):.3f} | {mean(chg):.3f} | {mx:.3f} vs {my:.3f} | {corr:.3f} | {mae:.3f} |")
        lines.append("\nDCM and the letter-only judges cannot roll forward: the next round's input is the agents' "
                     "messages, which they do not produce.")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
