#!/usr/bin/env python
"""Experiment 4 PPO: V4 attacker trained against DecentLLMs benign agents."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP1_MAIN = HERE.parent / "Experiment1_Main"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP1_MAIN))

from attacker.config import Namespace, load_config  # noqa: E402
from attacker.trainer import Trainer  # noqa: E402
import exp1_nodefense as exp1_runner  # noqa: E402
from decentllms_defense import DecentLLMsBenignEngine, self_check  # noqa: E402


DEFAULT_CONFIG = HERE / "exp4_decentllms_v1.yaml"


def _merge(dst: Namespace, src: Namespace, skip=()) -> None:
    for key, value in vars(src).items():
        if key in skip:
            continue
        current = getattr(dst, key, None)
        if isinstance(value, Namespace) and isinstance(current, Namespace):
            _merge(current, value, skip=skip)
        else:
            setattr(dst, key, value)


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (owner.parent / path).resolve()


def load_exp4_config(path: str | Path) -> Namespace:
    exp4_path = Path(path).resolve()
    delta = load_config(exp4_path)
    base = load_config(_resolve(exp4_path, delta.base_config))
    v4 = load_config(_resolve(exp4_path, delta.v4_config))
    _merge(base, v4, skip=("base_config",))
    _merge(base, delta, skip=("base_config", "v4_config"))
    return base


class Experiment4Trainer(Trainer):
    def _build_defense(self, cfg):
        dcfg = getattr(cfg, "defense", None)
        if getattr(dcfg, "type", None) != "decentllms_game_adaptation":
            raise ValueError("Experiment4 requires decentllms_game_adaptation")
        # Marker only. The evaluator requires the loaded WM and is installed as
        # a complete BenignEngine immediately after Trainer model construction.
        return None

    def _decent_engine(self, old_engine, max_new_tokens):
        return DecentLLMsBenignEngine(
            old_engine.model, old_engine.tok,
            max_pos=self.cfg.game.max_pos,
            max_new_tokens=max_new_tokens,
            evaluator_max_new_tokens=self.cfg.defense.evaluator_max_new_tokens,
            msg_order=self.cfg.game.msg_order,
            temperature=old_engine.temperature)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.wm = self._decent_engine(self.wm, cfg.world_model.max_new_tokens)

    def _get_eval_engine(self):
        engine = super()._get_eval_engine()
        if isinstance(engine, DecentLLMsBenignEngine):
            return engine
        self._eval_engine = self._decent_engine(
            engine, self.cfg.real_eval.max_new_tokens)
        return self._eval_engine

    def sample_batch(self, base_seed: int, sample: bool = True):
        self.wm.reset_diagnostics()
        episodes = super().sample_batch(base_seed, sample)
        self._last_decent_metrics = self.wm.diagnostic_metrics()
        return episodes

    def train_iteration(self, it: int):
        episodes = super().train_iteration(it)
        metrics = getattr(self, "_last_decent_metrics", None)
        if metrics:
            self.logger.log(it, metrics)
        return episodes


def check_invariants(cfg) -> None:
    self_check()
    assert sorted(cfg.game.benign_personalities) == [
        "neutral", "stubborn", "suggestible"]
    assert cfg.game.n_benign == 3
    assert cfg.defense.type == "decentllms_game_adaptation"
    assert int(cfg.defense.evaluator_count) == 3
    assert int(cfg.defense.criteria_count) == 5
    assert bool(cfg.defense.selected_proposal_as_peer_input)
    assert cfg.actor.model_name == (
        "/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft")
    assert getattr(cfg.actor, "sft_adapter", None) is None
    assert float(cfg.lr.actor) == 2.0e-5
    assert int(cfg.ppo.batch_episodes_N) == 16
    assert float(cfg.ppo.w_next_position) == 0.5
    assert float(cfg.ppo.w_message) == 0.5
    assert cfg.train.out_dir not in {
        "runs/exp1_nodefense_qwen3_ppo_dominant_v4",
        "runs/exp2_paper_wmsr_qwen3_ppo_v4_lr2e5_v1",
        "runs/exp3_cpwbft_qwen3_ppo_v4_lr2e5_v1",
    }


def run_preflight(cfg, episodes: int) -> None:
    original = exp1_runner.Trainer
    exp1_runner.Trainer = Experiment4Trainer
    try:
        exp1_runner.run_preflight(cfg, episodes)
    finally:
        exp1_runner.Trainer = original


def banner(cfg) -> None:
    print("=" * 76)
    print("Experiment 4 - DecentLLMs game adaptation")
    print(f"actor      : {cfg.actor.model_name}")
    print(f"critic     : {cfg.critic.model_name}")
    print(f"WM         : {cfg.world_model.model_name}")
    print("workers    : 4 proposals (3 benign + 1 attacker)")
    print("evaluators : 3 benign evaluators x 5 criteria, score 0--20")
    print("aggregate  : per-worker geometric median; highest total selected")
    print("benign use : selected proposal as sole peer input")
    print(f"PPO        : actor_lr={cfg.lr.actor} rollout_N={cfg.ppo.batch_episodes_N}")
    print(f"out        : {cfg.train.out_dir}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4 DecentLLMs PPO")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    cfg = load_exp4_config(args.config)
    check_invariants(cfg)
    banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
    else:
        Experiment4Trainer(cfg).fit()


if __name__ == "__main__":
    main()
