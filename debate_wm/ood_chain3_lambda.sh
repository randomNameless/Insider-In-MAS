#!/usr/bin/env bash
# 2026-11-04: three more OOD datasets, one after another, each = ood_chain_lambda.sh (none -> ours -> dwmbon) + ood_paired.py.
# setsid nohup bash ~/consensus-attacker/debate_wm/ood_chain3_lambda.sh > ~/runs/ood_chain3.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
for DS in gpqa_diamond logiqa2 medqa; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DATASET $DS"
  DS=$DS bash debate_wm/ood_chain_lambda.sh > /path/to/runs/ood_${DS}_chain.log 2>&1 || echo "FAILED $DS"
  .venv/bin/python debate_wm/ood_paired.py ood_$DS > /path/to/runs/ood_${DS}_paired.md 2> /path/to/runs/ood_${DS}_paired.err
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PAIRED $DS"
done
echo OOD_CHAIN3_DONE
