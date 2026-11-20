#!/bin/bash
# Q2 offline next-turn prediction chain (2026-11-05). Sequential GPU jobs, each well under 90 min.
set -x
cd ~/consensus-attacker; export PATH=$PWD/.venv/bin:$PATH
O=~/runs/wmpred_v1nn; M=~/models
python debate_wm/wm_predict_eval.py gen --out $O --tag dwm   --model $M/debate-wm-v1 --k 4
python debate_wm/wm_predict_eval.py gen --out $O --tag q3_4b --model $M/qwen3-4b     --k 4
python debate_wm/wm_predict_eval.py gen --out $O --tag q9b   --model $M/qwen3.5-9b   --k 4 --gpu-util 0.8
python debate_wm/wm_predict_eval.py judge --out $O --tag q4b --model $M/qwen3.5-4b --judges observer_v2
python debate_wm/wm_predict_eval.py dcm --out $O
python debate_wm/wm_predict_eval.py rollout --out $O --tag dwm   --model $M/debate-wm-v1 --k 4
python debate_wm/wm_predict_eval.py rollout --out $O --tag q3_4b --model $M/qwen3-4b     --k 4
python debate_wm/wm_predict_eval.py rollout --out $O --tag q9b   --model $M/qwen3.5-9b   --k 4 --gpu-util 0.8
echo CHAIN_DONE
