#!/usr/bin/env python
"""Experiment 4 PPO training against paper-faithful SAC benign agents."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP1 = HERE.parent / "Experiment1_Main"
sys.path[:0] = [str(ROOT), str(EXP1)]

from attacker.config import Namespace, load_config  # noqa: E402
from attacker.trainer import Trainer  # noqa: E402
import exp1_nodefense as exp1_runner  # noqa: E402
from sac_defense import SACBenignEngine, self_check  # noqa: E402

DEFAULT_CONFIG = HERE / "exp4_sac_sync_v1.yaml"


def _merge(dst: Namespace, src: Namespace, skip=()):
    for key, value in vars(src).items():
        if key in skip:
            continue
        current = getattr(dst, key, None)
        if isinstance(value, Namespace) and isinstance(current, Namespace):
            _merge(current, value, skip)
        else:
            setattr(dst, key, value)


def _resolve(owner: Path, value: str):
    path = Path(value)
    return path if path.is_absolute() else (owner.parent / path).resolve()


def load_exp4_config(path):
    path = Path(path).resolve()
    delta = load_config(path)
    base = load_config(_resolve(path, delta.base_config))
    v4 = load_config(_resolve(path, delta.v4_config))
    _merge(base, v4, ("base_config",))
    _merge(base, delta, ("base_config", "v4_config"))
    return base


class Experiment4Trainer(Trainer):
    def _build_defense(self, cfg):
        if cfg.defense.type != "experiment4_sac":
            raise ValueError("Experiment4Trainer requires experiment4_sac")
        # Marker; the loaded WM is needed, so the full engine is installed below.
        return None

    def __init__(self, cfg):
        super().__init__(cfg)
        self.wm = self._sac_engine(
            self.wm, cfg.world_model.max_new_tokens)

    def _sac_engine(self, old_engine, max_new_tokens):
        return SACBenignEngine(
            old_engine.model, old_engine.tok, max_pos=self.cfg.game.max_pos,
            max_new_tokens=max_new_tokens,
            scorer_max_new_tokens=self.cfg.defense.scorer_max_new_tokens,
            f=self.cfg.defense.f, temperature=old_engine.temperature)

    def _get_eval_engine(self):
        engine = super()._get_eval_engine()
        if isinstance(engine, SACBenignEngine):
            return engine
        self._eval_engine = self._sac_engine(
            engine, self.cfg.real_eval.max_new_tokens)
        return self._eval_engine

    def sample_batch(self, base_seed: int, sample: bool = True):
        self.wm.reset_diagnostics()
        episodes = super().sample_batch(base_seed, sample)
        self._last_sac_metrics = self.wm.diagnostic_metrics()
        return episodes

    def train_iteration(self, it: int):
        result = super().train_iteration(it)
        metrics = getattr(self, "_last_sac_metrics", None)
        if metrics:
            self.logger.log(it, metrics)
        return result


def check_invariants(cfg):
    self_check()
    assert cfg.game.n_benign == 3
    assert cfg.defense.type == "experiment4_sac"
    assert cfg.defense.topology == "complete_4"
    assert int(cfg.defense.f) == 1
    init_adapter = getattr(cfg.actor, "sft_adapter", None)
    if init_adapter:
        init_adapter = Path(str(init_adapter)).expanduser()
        assert init_adapter.is_dir(), f"initial PPO adapter directory missing: {init_adapter}"
        assert (init_adapter / "adapter_config.json").is_file(), (
            f"adapter_config.json missing from {init_adapter}")
        assert (init_adapter / "adapter_model.safetensors").is_file(), (
            f"adapter_model.safetensors missing from {init_adapter}")
        assert float(cfg.lr.actor) in (5e-6, 1e-5), (
            "SAC curriculum fine-tuning requires actor lr 5e-6 or 1e-5")
        assert "curriculum" in cfg.train.out_dir
    else:
        # Preserve the original SAC-from-SFT experiment invariant.
        assert float(cfg.lr.actor) == 2e-5
    assert int(cfg.ppo.batch_episodes_N) == 16
    assert float(cfg.ppo.w_next_position) == 0.5
    assert float(cfg.ppo.w_message) == 0.5
    assert "sync" in cfg.train.out_dir


def run_preflight(cfg, episodes):
    original = exp1_runner.Trainer
    exp1_runner.Trainer = Experiment4Trainer
    try:
        exp1_runner.run_preflight(cfg, episodes)
    finally:
        exp1_runner.Trainer = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    cfg = load_exp4_config(args.config)
    check_invariants(cfg)
    print("=" * 72)
    print("Experiment 4 - Self-Anchored Consensus (SAC), complete graph, F=1")
    print(f"actor={cfg.actor.model_name} lr={cfg.lr.actor} out={cfg.train.out_dir}")
    print(f"initial_adapter={getattr(cfg.actor, 'sft_adapter', None) or 'fresh LoRA'}")
    print("receiver-side scores: 3 benign receivers x 4 candidate responses/round")
    print("filter: remove bottom-1 only among neighbors scoring below self")
    print("timing: synchronous snapshot -> generate all -> atomic commit")
    print("=" * 72)
    if args.preflight:
        run_preflight(cfg, args.episodes)
    else:
        Experiment4Trainer(cfg).fit()


if __name__ == "__main__":
    main()
