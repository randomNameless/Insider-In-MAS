"""GPQA-Diamond / LogiQA 2.0 / MedQA (USMLE 4-option) -> the MMLU-Pro tasks.jsonl format, for the zero-shot OOD transfer
eval of the debate insider (2026-11-04), as prepare_ood_truthfulqa.py. All 4 options; option order as in the source
(GPQA mirror and MedQA already shuffle; LogiQA gold letter checked for balance below). n <= 400 sampled with SEED.

  ~/venvs/ju-dpo/bin/python debate_wm/prepare_ood_mcq.py --ds gpqa_diamond --out /path/to/data/ood_gpqa_diamond/tasks.jsonl
"""
import argparse
import ast
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
SEED = 20261104
SOURCES = {"gpqa_diamond": "fingertap/GPQA-Diamond:test (ungated mirror of Idavidrein/gpqa diamond, options pre-shuffled)",
           "logiqa2": "csitfun/LogiQA2.0 GitHub logiqa/DATA/LOGIQA/test.txt (1,572; ~/data/raw_logiqa2/test.txt)", "medqa": "GBaker/MedQA-USMLE-4-options:test"}
OPT = re.compile(r"\n\nA\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*)\s*$", re.S)


def load(ds):
    out = []
    if ds == "gpqa_diamond":
        for r in load_dataset("fingertap/GPQA-Diamond", split="test"):
            q = r["question"]
            i = q.rfind("\n\nA. ")
            m = OPT.match(q[i:])
            assert m, q[-300:]
            out.append((q[:i].strip(), [x.strip() for x in m.groups()], r["answer"]))
    elif ds == "logiqa2":
        for x in open("/path/to/data/raw_logiqa2/test.txt"):
            r = json.loads(x)
            out.append((r["text"].strip() + "\n\n" + r["question"].strip(), [o.strip() for o in r["options"]], LETTERS[int(r["answer"])]))
    elif ds == "medqa":
        for r in load_dataset("GBaker/MedQA-USMLE-4-options", split="test"):
            opts = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
            out.append((r["question"].strip(), [opts[k].strip() for k in "ABCD"], r["answer_idx"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=sorted(SOURCES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    rows = []
    for i, (stem, opts, g) in enumerate(load(a.ds)):
        assert len(opts) == 4 and g in "ABCD", (i, opts, g)
        obs = stem + "\n\nChoices (answer with the option letter):\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        rows.append({"task_id": f"{a.ds}-{i:05d}", "scenario": a.ds, "observation": obs, "answer": g, "split": "test",
                     "role": "ood_test", "metadata": {"source": SOURCES[a.ds], "question_id": i,
                                                      "choices": list(LETTERS[:4]), "choice_texts": opts}})
    n_all = len(rows)
    if len(rows) > a.n:
        rows = random.Random(SEED).sample(rows, a.n)
    rows.sort(key=lambda t: t["task_id"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in rows:
            f.write(json.dumps(t) + "\n")
    from collections import Counter
    words = sum(len(t["observation"].split()) for t in rows) / len(rows)
    print(f"{a.ds}: {len(rows)}/{n_all} tasks -> {a.out}; gold {sorted(Counter(t['answer'] for t in rows).items())}; "
          f"mean words {words:.0f}")


if __name__ == "__main__":
    main()
