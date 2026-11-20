#!/usr/bin/env python
"""Text similarity between model-generated arguments and the teacher's argument on the same turn
(files written by eval_wm.py --save-text). Metrics: ROUGE-L F1 and unigram F1 (word level), word-trigram
Jaccard, length ratio, and rate of persona-typical phrases, by persona and round.

    python -m debate_wm.text_similarity A.jsonl B.jsonl ...
"""
import json
import re
import sys
from collections import defaultdict

WORD = re.compile(r"[a-z0-9]+")
PHRASES = {  # rough persona style markers
    "stubborn": re.compile(r"(?i)\b(stick(ing)? with|maintain|refuse|not (going to|changing)|stand by|firm)\b"),
    "suggestible": re.compile(r"(?i)\b(agree with|convinced|follow|persuaded|adopt|switch(ing)? to)\b"),
    "neutral": re.compile(r"(?i)\b(let'?s|we should|together|consensus|balance)\b"),
}


def words(t):
    t = t.replace("Argument:", " ")
    t = t[: t.rfind("Answer:")] if "Answer:" in t else t
    return WORD.findall(t.lower())


def lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def sims(m, t):
    a, b = words(m), words(t)
    L = lcs(a, b)
    rl = f1(L / len(a), L / len(b)) if a and b else 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    ov = sum((ca & cb).values())
    u1 = f1(ov / len(a), ov / len(b)) if a and b else 0.0
    ta = {" ".join(a[i:i + 3]) for i in range(len(a) - 2)}
    tb = {" ".join(b[i:i + 3]) for i in range(len(b) - 2)}
    j3 = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return {"rougeL": rl, "unigramF1": u1, "trigramJ": j3, "len_ratio": len(a) / max(1, len(b)),
            "model_words": len(a)}


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path)]
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            s = sims(r["model_text"], r["teacher_target"])
            p = r["meta"]["persona"]
            s["persona_marker"] = float(bool(PHRASES[p].search(r["model_text"]))) if p in PHRASES else 0.0
            s["teacher_marker"] = float(bool(PHRASES[p].search(r["teacher_target"]))) if p in PHRASES else 0.0
            s["same_answer"] = float(r["model_top1"] == r["teacher_answer"])
            for key in ("all", f"persona={p}", f"round={r['meta']['round']}"):
                for k, v in s.items():
                    agg[key][k].append(v)
        print(f"== {path}  (n={len(rows)})")
        for key in sorted(agg, key=lambda k: (k != "all", k)):
            g = agg[key]
            print(f"  {key:20s} " + "  ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in g.items()))


if __name__ == "__main__":
    main()
