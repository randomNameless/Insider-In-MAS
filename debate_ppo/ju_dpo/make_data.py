#!/usr/bin/env python
"""Ju et al. 2025 (arXiv 2506.06800) DPO training data on OUR fit split (2026-10-22). Three steps, one model per process:

  persuade  every fit question x 12 prompts (NoneStrategy + 11 strategies), one message each, by the persuader
            (= their strategy_test.py sweep, PersuaderAgent: system/user prompt, generation_config decoding, 512 tokens)
  listen    the listener reads each message and answers the question (= their ListenerAgent, 8 tokens);
            success = it answers the target (their is_correct)
  pairs     preference pairs (= their strategy_generate_dataset.py, via Bozdag's _mcq port): group by (question, target),
            chosen = a successful message, rejected = a failed one, up to 5 pairs per group (their sampling), groups
            without both are dropped; prompt = the NoneStrategy prompt as one raw string. Also writes PSR per strategy.

Ours vs theirs: questions = MMLU-Pro fit split (1,441; theirs 3,000 CounterFact facts); one target per question from
targets_9b_fitval.json (most-voted wrong option, the rule of our test targets; theirs: CounterFact's target_new);
persuader = Qwen3-4B (our attacker model; theirs: LLaMA-3.1-8B / Qwen2.5-7B / Falcon-3-7B), thinking off;
listener = Qwen3.5-9B, thinking off (the honest model of our debates; theirs: the same LLM as the persuader in 4.3).

    python -m debate_ppo.ju_dpo.make_data persuade --out D --model ~/models/qwen3-4b
    python -m debate_ppo.ju_dpo.make_data listen   --out D --model ~/models/qwen3.5-9b
    python -m debate_ppo.ju_dpo.make_data pairs    --out D
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debate_ppo import baseline_bozdag as BZ  # noqa: E402
from debate_ppo import baseline_ju as JU  # noqa: E402
from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.ddata import file_sha256, splits  # noqa: E402

MAX_PAIRS_PER_INSTANCE = 5


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _jsonl(path):
    return [json.loads(l) for l in open(path)]


def _write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cmd_persuade(a):
    from debate_ppo.engines import make_vllm_engine
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = splits(a.tasks)[a.split][: a.n]
    tgt = json.load(open(a.targets))["targets"]
    rows = []
    for t in tasks:
        letter = tgt[t["task_id"]]
        assert letter != t["answer"], t["task_id"]
        _, _, by_letter = BZ.split_question(t["observation"])
        for s in JU.STRATEGIES:
            rows.append({"task_id": t["task_id"], "question": t["observation"], "n_options": len(t["metadata"]["choices"]),
                         "gold": t["answer"], "target": letter, "target_text": by_letter[letter], "strategy": s,
                         "seed": D.stable_seed("ju_data", t["task_id"], s)})
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len)
    log(f"persuade: {len(tasks)} questions x {len(JU.STRATEGIES)} prompts = {len(rows)} messages, decoding {JU.sampling_kwargs(a.model)}")
    texts, cuts = JU.generate(eng, [JU.persuader_chat(r["question"], r["target"], r["strategy"]) for r in rows],
                              [r["seed"] for r in rows], max_tokens=JU.MAX_TOKENS)
    for r, x, c in zip(rows, texts, cuts):
        r["evidence"], r["cut"] = x, c
    _write_jsonl(out / "persuade.jsonl", rows)
    json.dump({"persuader": a.model, "decoding": JU.sampling_kwargs(a.model), "max_tokens": JU.MAX_TOKENS,
               "tasks": a.tasks, "tasks_sha256": file_sha256(a.tasks), "split": a.split, "n_questions": len(tasks),
               "targets": a.targets, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(out / "persuade_config.json", "w"), indent=1)
    log(f"-> {out / 'persuade.jsonl'} (cut at {JU.MAX_TOKENS} tokens: {sum(cuts)}/{len(cuts)})")


def cmd_listen(a):
    from debate_ppo.engines import make_vllm_engine
    out = Path(a.out)
    rows = _jsonl(out / "persuade.jsonl")
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len)
    log(f"listen: {len(rows)} messages, decoding {JU.sampling_kwargs(a.model)}")
    texts, _ = JU.generate(eng, [JU.listener_chat(r["evidence"], r["question"]) for r in rows],
                           [D.stable_seed("ju_listen", r["task_id"], r["strategy"]) for r in rows],
                           max_tokens=JU.LISTENER_MAX_TOKENS)
    for r, x in zip(rows, texts):
        r["answer"] = x
        r["answer_letter"] = JU.parse_listener(x, r["n_options"])
        r["is_correct"] = r["answer_letter"] == r["target"]          # their name: persuasion succeeded
    _write_jsonl(out / "listen.jsonl", rows)
    json.dump({"listener": a.model, "decoding": JU.sampling_kwargs(a.model), "max_tokens": JU.LISTENER_MAX_TOKENS,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out / "listen_config.json", "w"), indent=1)
    log(f"-> {out / 'listen.jsonl'}  PSR {sum(r['is_correct'] for r in rows) / len(rows):.3f}, "
        f"unparsed {sum(r['answer_letter'] is None for r in rows)}")


def sample_pairs(chosen_list, rejected_list, max_pairs=MAX_PAIRS_PER_INSTANCE):
    """strategy_generate_dataset.py's pairing, verbatim logic (shuffle, zip, fill from unused combos, then with
    replacement). Uses the global `random` state like theirs."""
    chosen_list, rejected_list = list(chosen_list), list(rejected_list)
    random.shuffle(chosen_list)
    random.shuffle(rejected_list)
    pairs = list(zip(chosen_list, rejected_list))
    all_pairs = [(c, r) for c in chosen_list for r in rejected_list]
    used_pairs = set(pairs)
    remain = max_pairs - len(pairs)
    if remain > 0:
        unused_pairs = [p for p in all_pairs if p not in used_pairs]
        if len(unused_pairs) >= remain:
            pairs += random.sample(unused_pairs, remain)
        else:
            pairs += unused_pairs
            while len(pairs) < max_pairs and all_pairs:
                pairs.append(random.choice(all_pairs))
    return pairs[:max_pairs]


def build_pairs(rows, seed=42):
    random.seed(seed)                       # Bozdag's _mcq port: random.seed(42); upstream leaves it unseeded
    groups = collections.OrderedDict()
    for r in rows:
        ev = r["evidence"].strip()
        if not ev:                          # upstream skips items with empty evidence
            continue
        g = groups.setdefault((r["task_id"], r["target"]), {"prompt": JU.dpo_prompt(r["question"], r["target"]),
                                                            "chosen": [], "rejected": []})
        g["chosen" if r["is_correct"] else "rejected"].append(ev)
    data, n_skip = [], 0
    for (tid, _), g in groups.items():
        if not g["chosen"] or not g["rejected"]:
            n_skip += 1
            continue
        for c, rj in sample_pairs(g["chosen"], g["rejected"]):
            data.append({"prompt": g["prompt"], "chosen": c, "rejected": rj, "task_id": tid})
    return data, {"groups": len(groups), "groups_dropped_no_success_or_no_failure": n_skip, "pairs": len(data)}


def cmd_pairs(a):
    out = Path(a.out)
    rows = _jsonl(out / "listen.jsonl")
    data, st = build_pairs(rows, a.seed)
    _write_jsonl(out / "preference_data.jsonl", data)
    psr = {s: sum(r["is_correct"] for r in rows if r["strategy"] == s) / max(1, sum(r["strategy"] == s for r in rows))
           for s in JU.STRATEGIES}
    best = max((s for s in JU.STRATEGIES if s != JU.NONE), key=lambda s: psr[s])
    summ = {**st, "psr_fit_by_strategy": psr, "best_strategy_on_fit": best,
            "n_success_per_question": dict(collections.Counter(
                sum(r["is_correct"] for r in rows if r["task_id"] == t) for t in {r["task_id"] for r in rows})),
            "cut_rate": sum(r["cut"] for r in rows) / len(rows)}
    json.dump(summ, open(out / "pairs_summary.json", "w"), indent=1)
    (out / "best_strategy.txt").write_text(best + "\n")
    for s in JU.STRATEGIES:
        log(f"  PSR {s:24s} {psr[s]:.3f}")
    log(f"{st}  best fixed strategy on fit: {best}  -> {out / 'preference_data.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("persuade", "listen"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--gpu-util", type=float, default=0.85)
        p.add_argument("--max-model-len", type=int, default=6144)
        if name == "persuade":
            p.add_argument("--tasks", default="/path/to/data/mmlupro_v1/tasks.jsonl")
            p.add_argument("--targets", default="/path/to/runs/targets_9b_fitval.json")
            p.add_argument("--split", default="fit")
            p.add_argument("--n", type=int, default=None, help="first n questions (smoke)")
    p = sub.add_parser("pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    {"persuade": cmd_persuade, "listen": cmd_listen, "pairs": cmd_pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
