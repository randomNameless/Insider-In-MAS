#!/usr/bin/env bash
# Wait for the strat-DPO-base PPO run (tmux `train`, debate_v6_stratdpo_base.yaml) to finish, then start its 9B test eval
# (debate_wm/eval_v6sd_ours_lambda.sh) only if training ended cleanly ("[done]" in the log, last/ and checkpoint-0120 saved).
# Tolerates watchdog restarts (debate_ppo/watchdog.sh kills and recreates the `train` session, renaming the log).
# Lambda:  setsid nohup bash ~/consensus-attacker/debate_wm/chain_v6sd_lambda.sh > ~/runs/v6sd_chain.log 2>&1 &
set -u
RUN=$HOME/runs/exp9_debate_ppo_v6_stratdpo_base
LOG=$RUN.log
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(ts)] waiting for training ($LOG)"
while true; do
  grep -qs "\[done\]" "$LOG" && break
  if ! tmux has-session -t train 2>/dev/null; then
    sleep 300                                     # watchdog restart window
    grep -qs "\[done\]" "$LOG" && break
    tmux has-session -t train 2>/dev/null || { echo "[$(ts)] no train session and no [done]; eval NOT started"; exit 1; }
  fi
  sleep 60
done
echo "[$(ts)] training logged [done]"
[ -d $RUN/last ] && [ -d $RUN/checkpoint-0120 ] || { echo "[$(ts)] last/ or checkpoint-0120 missing; eval NOT started"; exit 1; }
for i in $(seq 1 60); do                          # let the trainer and its vLLM EngineCore release the GPU
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 30
done
echo "[$(ts)] GPU free, starting merge + eval"
bash $HOME/consensus-attacker/debate_wm/eval_v6sd_ours_lambda.sh
echo "[$(ts)] chain finished (exit $?)"
