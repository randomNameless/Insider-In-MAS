#!/usr/bin/env python
"""Experiment 2 PPO: V4 policy training against paper-faithful global MSR."""
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
from paper_wmsr import PaperGlobalMSRDefense, self_check  # noqa: E402


DEFAULT_CONFIG = HERE / "exp2_wmsr_v1.yaml"


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


def load_exp2_config(path: str | Path) -> Namespace:
    exp2_path = Path(path).resolve()
    delta = load_config(exp2_path)
    base = load_config(_resolve(exp2_path, delta.base_config))
    v4 = load_config(_resolve(exp2_path, delta.v4_config))
    _merge(base, v4, skip=("base_config",))
    _merge(base, delta, skip=("base_config", "v4_config"))
    return base


class Experiment2Trainer(Trainer):
    """Inject the isolated Exp2 filter without changing shared Trainer code."""

    def _build_defense(self, cfg):
        dcfg = getattr(cfg, "defense", None)
        dtype = getattr(dcfg, "type", None) if dcfg is not None else None
        if dtype != "paper_global_wmsr":
            raise ValueError(
                "Experiment2Trainer requires defense.type=paper_global_wmsr"
            )
        f = int(getattr(dcfg, "f", 1))
        print(f"[defense] paper global MSR input filter, f={f}")
        return PaperGlobalMSRDefense(f=f)


def check_invariants(cfg) -> None:
    self_check()
    assert sorted(cfg.game.benign_personalities) == [
        "neutral", "stubborn", "suggestible"
    ]
    assert cfg.game.n_benign == 3
    assert cfg.defense.type == "paper_global_wmsr"
    assert int(cfg.defense.f) == 1
    assert cfg.actor.model_name == (
        "/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"
    )
    assert getattr(cfg.actor, "sft_adapter", None) is None
    assert float(cfg.lr.actor) == 2.0e-5, "Experiment 2 must keep V4 actor LR"
    assert int(cfg.ppo.batch_episodes_N) == 16
    assert float(cfg.ppo.w_next_position) == 0.5
    assert float(cfg.ppo.w_message) == 0.5
    assert cfg.train.out_dir != "runs/exp1_nodefense_qwen3_ppo_dominant_v4"


def run_preflight(cfg, episodes: int) -> None:
    # Reuse the complete, current [A]-[F] suite without editing it. Its Trainer
    # global is replaced only for this call so sampled rollouts use Exp2's filter.
    original = exp1_runner.Trainer
    exp1_runner.Trainer = Experiment2Trainer
    try:
        exp1_runner.run_preflight(cfg, episodes)
    finally:
        exp1_runner.Trainer = original


def banner(cfg) -> None:
    print("=" * 72)
    print("Experiment 2 - PAPER GLOBAL MSR, mixed benign personalities")
    print(f"actor    : {cfg.actor.model_name}")
    print(f"critic   : {cfg.critic.model_name}")
    print(f"WM       : {cfg.world_model.model_name}")
    print(f"defense  : sort received neighbor proposals; trim f={cfg.defense.f} each end")
    print(f"PPO      : actor_lr={cfg.lr.actor} rollout_N={cfg.ppo.batch_episodes_N} ")
    print(f"out      : {cfg.train.out_dir}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2 PPO with paper global MSR")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    cfg = load_exp2_config(args.config)
    check_invariants(cfg)
    banner(cfg)
    if args.preflight:
        run_preflight(cfg, args.episodes)
    else:
        Experiment2Trainer(cfg).fit()


if __name__ == "__main__":
    main()

