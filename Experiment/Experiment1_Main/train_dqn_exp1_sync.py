#!/usr/bin/env python
"""Synchronous Experiment-1 DQN wrapper with isolated artifacts."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "Experiment1_Eval"
sys.path.insert(0, str(EVAL))
import train_attacker_maxentropy_qwen_mixed as impl  # noqa: E402

if __name__ == "__main__":
    if "--best_path" not in sys.argv:
        sys.argv.extend(["--best_path", str(HERE / "runs/dqn_exp1_sync/best")])
    if "--save_path" not in sys.argv:
        sys.argv.extend(["--save_path", str(HERE / "runs/dqn_exp1_sync/attacker_softq_sync.pt")])
    if "--eval_curve_path" not in sys.argv:
        sys.argv.extend(["--eval_curve_path", str(HERE / "runs/dqn_exp1_sync/eval_curve.csv")])
    impl.main()
