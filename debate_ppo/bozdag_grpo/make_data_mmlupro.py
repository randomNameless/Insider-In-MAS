#!/usr/bin/env python
"""Bozdag et al. 2026 GRPO reproduction, setup B (2026-10-20): build their GRPO training parquet from OUR
MMLU-Pro fit split instead of TruthfulQA, with OUR victim (Qwen3.5-9B, thinking off) as the persuadee.

Everything per (question, target) pair is THEIR code: process_task() from
src/data_generation/tqa_claim_generation.py (persuadee system / starter prompts, T=0, max_tokens=16384, 3 retries,
pair dropped when the persuadee's initial letter == target letter, persuader system prompt
persuader_prompt_mcq(question, target_text, options_str), user turn = <other_agent_message> persuadee reasoning,
data_source / reward_model / interaction_kwargs layout). Only the task list differs (their precompute_tasks() is
TruthfulQA-specific):
  * question = our MMLU-Pro stem; options_str = our "A. text" lines, all 10 options in our order (their TQA pool is
    gold + target + 2 random wrong, shuffled);
  * targets = EVERY wrong option of the question (their rule: every incorrect answer is a target);
  * questions = the fit split only (val / test untouched; test is the evaluation set).
Usage (persuadee served by vLLM on :8000, see run_lambda.sh):
  python make_data_mmlupro.py --out ~/runs/bozdag_grpo_B/data/mmlupro_fit_training_data
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AP = Path(os.environ.get("ADVERSARIAL_PERSUASION_ROOT", Path.home() / "adversarial-persuasion"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AP / "src"))
sys.path.insert(0, str(AP / "src" / "data_generation"))

import datasets  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

import tqa_claim_generation as G  # noqa: E402  (their module, unmodified)
from prompts.persuadee_prompt import persuadee_prompt_mcq  # noqa: E402

from debate_ppo import baseline_bozdag as BZ  # noqa: E402
from debate_ppo.ddata import splits  # noqa: E402


def build_tasks(tasks_path: str, split: str):
    out = []
    for t in splits(tasks_path)[split]:
        stem, opts, by_letter = BZ.split_question(t["observation"])
        gold = t["answer"]
        letters = sorted(by_letter)
        for tl in letters:
            if tl == gold:
                continue
            out.append({
                "id": f"{t['task_id']}_{tl}",
                "question": stem,
                "correct_answer": by_letter[gold],
                "target_answer": by_letter[tl],
                "options_pool": [by_letter[l] for l in letters],
                "remaining_incorrect_answers": [by_letter[l] for l in letters if l not in (gold, tl)],
                "options_str": opts,
                "target_answer_letter": tl,
                "correct_answer_letter": gold,
            })
    return out


async def run(a):
    tasks = build_tasks(a.tasks, a.split)
    client = AsyncOpenAI(api_key="EMPTY", base_url=a.base_url)
    sem = asyncio.Semaphore(a.max_concurrent)
    sysp = persuadee_prompt_mcq()
    # one persuadee call per question is enough at T=0, but their code calls once per (question, target); kept.
    coros = [G.process_task(t, client, a.model, sysp, sem) for t in tasks]
    rows = []
    for fut in tqdm(asyncio.as_completed(coros), total=len(coros)):
        r = await fut
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda r: r["extra_info"]["id"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    datasets.Dataset.from_list(rows).to_parquet(a.out + ".parquet")
    n_q = len({r["extra_info"]["id"].rsplit("_", 1)[0] for r in rows})
    print(json.dumps({"pairs_total": len(tasks), "rows_kept": len(rows), "questions": n_q,
                      "dropped_target_eq_persuadee_or_failed": len(tasks) - len(rows)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/path/to/data/mmlupro_v1/tasks.jsonl")
    ap.add_argument("--split", default="fit")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-concurrent", type=int, default=48)          # their run_data_gen.sh
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
