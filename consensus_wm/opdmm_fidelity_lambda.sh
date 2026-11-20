#!/usr/bin/env bash
# Paper thread 2026-11-11: sampled next-position hit rate of the PPO-training consensus WM (qwen35-wm = opd-mm, Appendix C)
# on the same 6,603 held-out fixed-seat turns as arm A (0.699; 9B resample ceiling 0.717). qwen35-wm was trained on the
# 9B's own prompt format, so it gets --prompt ref (exact 9B prompt), same sampling as ref9b_fixed. Waits for wm3k chain.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
until grep -q WM3K_CHAIN_DONE ~/runs/wm3k_chain.log; do sleep 120; done
O=~/runs/cwm_eval_opdmm; mkdir -p $O
python -m consensus_wm.eval_wm openloop --model ~/models/qwen35-wm --prompt ref --inp ~/data/cwm/fixed_test.jsonl --out $O/opdmm_fixed.jsonl
python -m consensus_wm.analyze openloop --ref ~/runs/cwm_eval/ref9b_fixed.jsonl A=~/runs/cwm_eval/A_fixed.jsonl opdmm=$O/opdmm_fixed.jsonl > $O/openloop.md 2>&1
echo OPDMM_DONE
