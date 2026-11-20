#!/usr/bin/env python
"""DPO preference pairs for adversarial training of the honest debate agents (2026-11-09).

For every attacked honest prompt (debate_wm/build_at_data.py) with candidate replies from debate_wm/at_sample.py
(one or more --cands files, same prompts, different seed tags):
    chosen   = a valid, untruncated candidate whose letter == the clean-run answer of that agent/round (consistency)
    rejected = a valid, untruncated candidate whose letter != the clean answer (sampled under the same attacked prompt)
One pair per prompt (first chosen / first rejected in candidate order). Prompts with no chosen or no rejected are skipped.
Records: {"prompt": messages, "chosen": "<argument>\\nAnswer: X", "rejected": "...", "n_options", "meta"}.
Split by task id with the same hash rule as at_chain_lambda.sh (val = 10% of fit tasks).

    python -m debate_wm.build_dpo_pairs --attacked ~/data/at/fit_attacked.jsonl --cands ~/data/at/fit_cons_cands.jsonl ~/data/at/fit_cons_cands_x8.jsonl \
        --train ~/data/at/train_dpo.jsonl --val ~/data/at/val_dpo.jsonl
    python -m debate_wm.build_dpo_pairs --attacked ... --cands ~/data/at/fit_cons_cands.jsonl --missing-out ~/data/at/fit_attacked_nochosen.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

from debate_wm.build_sft import target_text


def key(m):
    return (m["task_id"], m["seed"], m["round"], m["agent"], m["run"])


def reply_text(c):
    raw = c["argument_text"] + ("" if c["argument_text"].endswith("\n") else "\n") + "Answer: " + c["answer"]
    return target_text(raw, c["answer"])


def is_val(task_id):
    return int(hashlib.sha256(("at_val|" + task_id).encode()).hexdigest()[:8], 16) % 10 == 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacked", required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--missing-out", default=None, help="write the attacked prompts that still have no chosen candidate")
    a = ap.parse_args()
    att = {}
    for l in open(a.attacked):
        r = json.loads(l)
        att[key(r["meta"])] = r
    cands = defaultdict(list)
    for path in a.cands:
        for l in open(path):
            r = json.loads(l)
            cands[key(r["meta"])] += [c for c in r["cands"] if c["valid"] and not c["truncated"] and c["answer"]]
    st = Counter()
    pairs, missing = [], []
    for k, r in att.items():
        tgt = r["clean_answer"]
        if not tgt:
            st["no_clean_answer"] += 1
            continue
        cs = cands.get(k, [])
        ok = [c for c in cs if c["answer"] == tgt]
        bad = [c for c in cs if c["answer"] != tgt]
        if not ok:
            st["no_chosen"] += 1
            missing.append(r)
            continue
        if not bad:
            st["no_rejected"] += 1
            continue
        rej, src = reply_text(bad[0]), "cand"
        pairs.append({"prompt": r["messages"], "chosen": reply_text(ok[0]), "rejected": rej, "n_options": r["n_options"],
                      "meta": {**r["meta"], "rejected_source": src, "clean_answer": tgt, "rejected_answer": bad[0]["answer"],
                               "n_chosen": len(ok), "n_rejected": len(bad)}})
        st["pairs"] += 1
        st["rejected_" + src] += 1
        st["pairs_" + r["meta"]["run"]] += 1
    print(json.dumps(st, indent=1))
    if a.missing_out:
        with open(a.missing_out, "w") as f:
            for r in missing:
                f.write(json.dumps(r) + "\n")
        print(f"{len(missing)} prompts without chosen -> {a.missing_out}")
    if a.train:
        tr, va = open(a.train, "w"), open(a.val, "w")
        n = Counter()
        for p in pairs:
            v = is_val(p["meta"]["task_id"])
            (va if v else tr).write(json.dumps(p) + "\n")
            n["val" if v else "train"] += 1
        print(json.dumps(n))
        json.dump({"stats": st, "split": n, "cands": a.cands}, open(a.train + ".stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
