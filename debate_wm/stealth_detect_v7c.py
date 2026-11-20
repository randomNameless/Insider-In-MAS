"""Stealth detector on the v7c PPO checkpoints (exp10_nn_v7c), same judge/prompt/decoding as stealth_detect.py.
Writes to a separate out dir so ~/runs/stealth_detect is untouched; sd120 is re-judged as a reproducibility check.

  python debate_wm/stealth_detect_v7c.py --out ~/runs/stealth_detect_v7c
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stealth_detect as S

S.CONDS = [
    ("v7c200", "exp10_nn_v7c", "targeted_model_rl_v7c200", "ours: v7c-200 single sample"),
    ("v7c240", "exp10_nn_v7c", "targeted_model_rl_v7c240", "ours: v7c-240 single sample"),
    ("sd120_recheck", "exp10_nn_ours", "targeted_model_rl_sd120", "ours: sd120 single sample (re-judged)"),
]

FULL = [
    ("v7c200_wm_peek", "exp10_nn_v7c_dwmbon", "targeted_model_rl_v7c200__wmbon8dw", "ours: v7c-200 + WM-BoN8 (peek)"),
    ("v7c200_wm_np", "exp10_nn_np_v7c", "targeted_model_rl_v7c200__wmbon8dw", "ours: v7c-200 + WM-BoN8 (no-peek)"),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="also judge the v7c-200 + WM-BoN8 runs (existing rows are skipped)")
    ap.add_argument("--prl-np", action="store_true", help="also judge P-RL no-think 200 + WM-BoN8 (no-peek)")
    a = ap.parse_args()
    if a.full:
        S.CONDS = S.CONDS + FULL
    if a.prl_np:
        S.CONDS = S.CONDS + [("prl_nothink200_wm_np", "exp10_nn_np_bozprl", "bozdag_prl_3b_nothink_len200_wmbon8dw",
                              "P-RL no-think 200 tok + WM-BoN8 (no-peek)")]
    S.cmd_run(a)
