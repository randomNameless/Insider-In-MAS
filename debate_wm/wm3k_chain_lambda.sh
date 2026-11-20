#!/usr/bin/env bash
# 2026-11-11 (thread "cheap WM"): debate WMs trained on only 3,000 real debates (36,000 honest turns, stratified
# subsample of ~/data/wm/v1_fit.jsonl by debate_wm/subsample_games.py), same recipe as wm_sft_v1 (1 epoch, lr 1e-5,
# kl_coef 1.0, max_tokens 16384, accum 4, same val). Students: Qwen3-1.7B (closest to 1.5B) and Qwen3.5-4B (v1's base).
# Then: next-turn fidelity (wm_predict_eval gen + rollout, K=4, same 1200 recorded v1nn debates as wmpred_v1nn) and
# sd120 + WM-BoN8 (debate_ppo/eval_debate_dwm.py, PPO protocol v1nn, 9B honest, test 400 x seeds 3/4/5).
# Every step is a separate job < 90 min.  setsid nohup bash debate_wm/wm3k_chain_lambda.sh > ~/runs/wm3k_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; D=/path/to/data
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
SFT="python -m debate_wm.sft_train --train $D/wm3k/fit3k.jsonl --val $D/wm/v1_val.jsonl --epochs 1.0 --lr 1e-5 --warmup 0.03 --max-tokens 16384 --accum 4 --kl-coef 1.0 --eval-every 100 --val-max 4000 --seed 0"
# eval-every 100: 3k games give ~1/4 of v1's optimizer steps; v1 used 400 of ~1700.

if [ ! -f $R/wm3k_q3_1p7b/last/config.json ]; then ts "train 1.7B"; $SFT --model $M/qwen3-1.7b --out $R/wm3k_q3_1p7b > $R/wm3k_q3_1p7b.log 2>&1; fi
[ -d $M/debate-wm-3k-q3-1.7b ] || cp -r $R/wm3k_q3_1p7b/best $M/debate-wm-3k-q3-1.7b
if [ ! -f $R/wm3k_q35_4b/last/config.json ]; then ts "train 4B"; $SFT --model $M/qwen3.5-4b --out $R/wm3k_q35_4b > $R/wm3k_q35_4b.log 2>&1; fi
[ -d $M/debate-wm-3k-q35-4b ] || python -m debate_wm.export_vllm --trained $R/wm3k_q35_4b/best --base $M/qwen3.5-4b --out $M/debate-wm-3k-q35-4b

# fidelity: new dir, recorded debates + v1/q3_4b/q9b outputs linked read-only from wmpred_v1nn for the same tables
O=$R/wmpred_3k; mkdir -p $O
for f in debates.jsonl dcm.jsonl gen_dwm.jsonl gen_q3_4b.jsonl gen_q9b.jsonl rollout_dwm.jsonl rollout_q3_4b.jsonl rollout_q9b.jsonl judge_q4b_observer_v2.jsonl; do
  [ -e $O/$f ] || { [ -e $R/wmpred_v1nn/$f ] && ln -s $R/wmpred_v1nn/$f $O/$f || true; }
done
for X in dwm3k_1p7b:$M/debate-wm-3k-q3-1.7b dwm3k_4b:$M/debate-wm-3k-q35-4b; do
  T=${X%%:*}; P=${X#*:}
  [ -f $O/gen_$T.jsonl ] || { ts "gen $T"; python debate_wm/wm_predict_eval.py gen --out $O --tag $T --model $P --k 4; }
done

# attack: sd120 + WM-BoN8 with each small WM as the selector (only the selector changes vs exp10_nn_dwmbon)
TAG=exp10_nn; RSD=$R/exp9_debate_ppo_v6_stratdpo_base
for X in 1p7b:$M/debate-wm-3k-q3-1.7b 4b:$M/debate-wm-3k-q35-4b; do
  T=${X%%:*}; P=${X#*:}; RUN=$R/${TAG}_dwmbon3k_$T
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_dwmbon3k_${T}_honest_suggestible_neutral_neutral" \
      --victim $M/qwen3.5-9b --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\"}" \
      --conds targeted_model_rl_sd120__wmbon8dw --seeds 3,4,5 --n 400 \
      --answer-first --profile-set v1nn --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
      --arg-tokens 200 --gpu-util 0.6 --targets $R/targets_none9b_${TAG}.json \
      --rl-checkpoint "sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)" --dwm-model $P
  fi
  ts "play $RUN"
  python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
done

for X in dwm3k_1p7b:$M/debate-wm-3k-q3-1.7b dwm3k_4b:$M/debate-wm-3k-q35-4b; do
  T=${X%%:*}; P=${X#*:}
  [ -f $O/rollout_$T.jsonl ] || { ts "rollout $T"; python debate_wm/wm_predict_eval.py rollout --out $O --tag $T --model $P --k 4; }
done
ts "WM3K_CHAIN_DONE"
