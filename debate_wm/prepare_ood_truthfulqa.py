"""TruthfulQA MC1 (truthfulqa/truthful_qa, multiple_choice, validation, 817 Qs) -> the MMLU-Pro tasks.jsonl format, for
the zero-shot OOD transfer eval of the debate insider (2026-11-02). Options: the correct answer + up to 9 incorrect
(source order), shuffled with a per-question seed so the gold letter is uniform. 400 questions sampled with SEED.
All tasks get split="test"; debate_ppo.ddata.splits returns them unchanged for non-MMLU-Pro files.

  python debate_wm/prepare_ood_truthfulqa.py --out /path/to/data/ood_truthfulqa_mc1/tasks.jsonl
"""
import argparse
import json
import random

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
SEED = 20261102


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rows = []
    for i, r in enumerate(ds):
        ch, lab = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        gold = ch[lab.index(1)]
        opts = [gold] + [c for c, l in zip(ch, lab) if l == 0][: len(LETTERS) - 1]
        random.Random(SEED + i).shuffle(opts)
        k = len(opts)
        g = LETTERS[opts.index(gold)]
        obs = r["question"].strip() + "\n\nChoices (answer with the option letter):\n" + "\n".join(
            f"{LETTERS[j]}. {o}" for j, o in enumerate(opts))
        rows.append({"task_id": f"tqa-{i:04d}", "scenario": "truthfulqa_mc1", "observation": obs, "answer": g,
                     "split": "test", "role": "ood_test",
                     "metadata": {"source": "truthfulqa/truthful_qa:multiple_choice:validation", "question_id": i,
                                  "n_options_orig": len(ch), "choices": list(LETTERS[:k]), "choice_texts": opts}})
    rows = sorted(random.Random(SEED).sample(rows, a.n), key=lambda t: t["task_id"])
    with open(a.out, "w") as f:
        for t in rows:
            f.write(json.dumps(t) + "\n")
    ks = [len(t["metadata"]["choices"]) for t in rows]
    print(f"{len(rows)} tasks -> {a.out}; options min/mean/max {min(ks)}/{sum(ks)/len(ks):.2f}/{max(ks)}")


if __name__ == "__main__":
    main()
