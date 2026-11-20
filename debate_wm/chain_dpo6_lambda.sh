#!/usr/bin/env bash
# Full v6-format strat-DPO chain (2026-10-28): debate_ppo/strat_dpo/run_v6.sh all (collect -> data -> DPO -> merge)
# -> debate_wm/eval_nn_dpo6_lambda.sh (exp10_nn harness) -> debate_wm/nn_paired_dpo6.py tables. Stops at the first error.
# Lambda:  setsid nohup bash ~/consensus-attacker/debate_wm/chain_dpo6_lambda.sh > ~/runs/strat_dpo_v6_chain.log 2>&1 &
set -euo pipefail
CA=$HOME/consensus-attacker; R=$HOME/runs
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
ts "chain start"
bash $CA/debate_ppo/strat_dpo/run_v6.sh all
[ -s $R/strat_dpo_v6_q3_4b/merged_sha256.txt ] || { ts "merge missing; eval NOT started"; exit 1; }
ts "eval start"
bash $CA/debate_wm/eval_nn_dpo6_lambda.sh
ts "paired"
cd $CA && source .venv/bin/activate
CUDA_VISIBLE_DEVICES= python debate_wm/nn_paired_dpo6.py exp10_nn > $R/exp10_nn_dpo6_paired.tsv
cat $R/exp10_nn_dpo6_paired.tsv
ts "CHAIN_DPO6_DONE"
