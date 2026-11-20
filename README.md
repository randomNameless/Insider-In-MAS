# Persuasion from Within: Model-Based Insider Attacks on Multi-Agent LLM Debate and Consensus

Anonymous code release accompanying the submission. This repository is a work in progress; see the TODOs below.

## Repository layout

| Path | Contents |
|---|---|
| `attacker/` | Core library for the numerical consensus game: game state and consensus predicate, prompts and action parsing, reward and potential shaping, turn-level GAE and PPO, batched generation, actor/critic/world-model wrappers, rollout and trainer, W-MSR and CP-WBFT defenses. |
| `configs/` | Training configs for the consensus-game attacker. |
| `scripts/` | Entry points: training, adapter merging, transfer evaluation, preflight checks. |
| `tests/` | Unit tests for the consensus-game library (`pytest`). |
| `consensus_wm/` | Consensus-game world model: data collection, SFT, world-model evaluation, attack evaluation. |
| `Experiment/` | Consensus-game experiments: benign baseline, main attack, defenses (W-MSR, CP-WBFT, SAC), agent scaling, topology, defense adaptation, channel ablation. |
| `debate_ppo/` | Multi-agent debate: protocol, engines, PPO trainer and rollout, evaluation, defenses, world-model Best-of-N, and literature baselines (`baseline_*.py`, `ju_dpo/`, `bozdag_grpo/`, `strat_dpo/`). |
| `debate_wm/` | Debate world model: SFT, evaluation, next-turn prediction, OOD transfer, ablations, stealth detection, adversarial-training defense, and paired-analysis scripts. |

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Model and data paths in configs and shell scripts use the placeholder `/path/to/`; replace it with your own locations.

## Reproducing the main results

TODO: commands for (1) consensus game, (2) debate world model, (3) attacker training, (4) evaluation and defenses.

## TODO before release

- [ ] Add data preparation instructions (MMLU-Pro and OOD datasets).
- [ ] Document the main entry point per paper table/figure.
- [ ] Add license.
