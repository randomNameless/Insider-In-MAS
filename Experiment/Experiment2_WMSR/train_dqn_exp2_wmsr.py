#!/usr/bin/env python
"""Experiment 2 DQN wrapper using the paper-faithful global MSR filter."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "Experiment1_Eval"
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

import train_attacker_maxentropy_qwen_mixed_wmsr as dqn_impl  # noqa: E402
from paper_wmsr import retained_neighbor_indices, self_check  # noqa: E402


def _paper_visible_others(self, i):
    return retained_neighbor_indices(self.pos, i, int(self.wmsr_F))


def main() -> None:
    self_check()
    # Runtime-local replacement only; the original DQN implementation is not edited.
    dqn_impl.AttackEnvQwenMixedWMSR._visible_others = _paper_visible_others
    if "--out_dir" not in sys.argv:
        sys.argv.extend(["--out_dir", str(HERE / "runs" / "dqn_sync_paper_wmsr_v1")])
    if "--save_path" not in sys.argv:
        sys.argv.extend(["--save_path", str(HERE / "attacker_softq_sync_wmsr.pt")])
    dqn_impl.main()


if __name__ == "__main__":
    main()
