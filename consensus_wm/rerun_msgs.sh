#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."; export PATH=$PWD/.venv/bin:$PATH
D=~/data/cwm; E=~/runs/cwm_eval_msg; mkdir -p $E
python -m consensus_wm.eval_wm openloop --model ~/models/qwen3.5-9b --prompt ref --inp $D/shuffle_test.jsonl --out $E/ref9b_shuffle.jsonl
python -m consensus_wm.eval_wm openloop --model ~/runs/cwm_sft_A/best --prompt persona --inp $D/shuffle_test.jsonl --out $E/A_shuffle.jsonl
python -m consensus_wm.eval_wm openloop --model ~/runs/cwm_sft_C/best --prompt id --inp $D/shuffle_test.jsonl --out $E/C_shuffle.jsonl
echo MSG_DONE
