#!/usr/bin/env python
"""Experiment 3 PPO: V4 policy training against CP-WBFT benign agents."""
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
from cpwbft_defense import Experiment3CPWBFTDefense, self_check  # noqa: E402


DEFAULT_CONFIG = HERE / "exp3_cpwbft_v1.yaml"


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


def load_exp3_config(path: str | Path) -> Namespace:
    exp3_path = Path(path).resolve()
    delta = load_config(exp3_path)
    base = load_config(_resolve(exp3_path, delta.base_config))
    v4 = load_config(_resolve(exp3_path, delta.v4_config))
    _merge(base, v4, skip=("base_config",))
    _merge(base, delta, skip=("base_config", "v4_config"))
    return base


class Experiment3Trainer(Trainer):
    """Inject CP-WBFT without changing the shared Trainer or older experiments."""

    def _build_defense(self, cfg):
        dcfg = getattr(cfg, "defense", None)
        dtype = getattr(dcfg, "type", None) if dcfg is not None else None
        if dtype != "experiment3_cpwbft":
            raise ValueError(
                "Experiment3Trainer requires defense.type=experiment3_cpwbft")
        margin = float(getattr(dcfg, "margin", 0.0))
        keep_min_one = bool(getattr(dcfg, "keep_min_one", True))
        print("[defense] CP-WBFT confidence input filter: "
              f"source=Position-token decoder peakedness margin={margin} "
              f"keep_min_one={keep_min_one}")
        return Experiment3CPWBFTDefense(margin, keep_min_one)


def check_invariants(cfg) -> None:
    self_check()
    assert sorted(cfg.game.benign_personalities) == [
        "neutral", "stubborn", "suggestible"]
    assert cfg.game.n_benign == 3
    assert cfg.defense.type == "experiment3_cpwbft"
    assert cfg.defense.confidence_source == "position_token_decoder_peakedness"
    assert float(cfg.defense.margin) == 0.0
    assert bool(cfg.defense.keep_min_one)
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
    }


def run_preflight(cfg, episodes: int) -> None:
    original = exp1_runner.Trainer
    exp1_runner.Trainer = Experiment3Trainer
    try:
        exp1_runner.run_preflight(cfg, episodes)
    finally:
        exp1_runner.Trainer = original


def banner(cfg) -> None:
    print("=" * 74)
    print("Experiment 3 - CP-WBFT, mixed benign personalities")
    print(f"actor      : {cfg.actor.model_name}")
    print(f"critic     : {cfg.critic.model_name}")
    print(f"WM         : {cfg.world_model.model_name}")
    print("confidence : Position-token decoder peakedness")
    print(f"filter     : C_j >= C_i - {cfg.defense.margin}; "
          f"keep_min_one={cfg.defense.keep_min_one}")
    print(f"PPO        : actor_lr={cfg.lr.actor} "
          f"rollout_N={cfg.ppo.batch_episodes_N} weights=0.5/0.5")
    print(f"out        : {cfg.train.out_dir}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 3 PPO with CP-WBFT")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    cfg = load_exp3_config(args.config)
    check_invariants(cfg)
    banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
    else:
        Experiment3Trainer(cfg).fit()


if __name__ == "__main__":
    main()
