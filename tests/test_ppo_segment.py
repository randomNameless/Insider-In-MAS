import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attacker.ppo import W, ppo_turn_loss, segment_weighted_mean, turn_log_ratio


def _masks(n_pos, n_msg):
    V = n_pos + n_msg
    pos = torch.zeros(V, dtype=torch.bool)
    msg = torch.zeros(V, dtype=torch.bool)
    pos[:n_pos] = True
    msg[n_pos:] = True
    return {"next_position": pos, "message": msg}


def test_turn_log_ratio_weighting():
    logp = torch.tensor([0.0, 0.0, 0.0])       # pos, msg, msg
    old = torch.tensor([-1.0, -0.5, -0.5])     # d = [1.0, 0.5, 0.5]
    seg = _masks(1, 2)
    weights = {"next_position": 0.1, "message": 0.9}
    D = float(turn_log_ratio(logp, old, seg, weights=weights))
    expected = 0.1 * 1.0 + 0.9 * 0.5           # w_pos*mean_pos + w_msg*mean_msg
    assert abs(D - expected) < 1e-6


def test_equal_segment_weighting():
    logp = torch.tensor([0.0, 0.0, 0.0])
    old = torch.tensor([-1.0, -0.5, -0.5])
    D = float(turn_log_ratio(logp, old, _masks(1, 2)))
    assert abs(D - 0.75) < 1e-6


def test_equal_weighting_also_applies_to_auxiliary_losses():
    values = torch.tensor([1.0, 0.5, 0.5])
    got = float(segment_weighted_mean(values, _masks(1, 2)))
    assert abs(got - 0.75) < 1e-6


def test_missing_segment_renormalizes():
    # message missing -> only next_position present -> weight renormalized to 1.0
    logp = torch.tensor([0.0])
    old = torch.tensor([-2.0])                 # d = 2.0
    seg = {"next_position": torch.tensor([True]),
           "message": torch.tensor([False])}
    D = float(turn_log_ratio(logp, old, seg))
    assert abs(D - 2.0) < 1e-6, "missing segment must renormalize, not shrink"
    # the WRONG behavior (0.1 weight) would give 0.2
    assert abs(D - 0.2) > 1e-3


def test_effective_advantage_conserved():
    # sum_j A_t * w_k/n_k over all value tokens == A_t
    n_pos, n_msg, A = 2, 5, 1.7
    total = n_pos * (W["next_position"] / n_pos) + n_msg * (W["message"] / n_msg)
    assert abs(total - 1.0) < 1e-12
    assert abs(A * total - A) < 1e-12


def test_clip_all_or_nothing():
    # positive advantage, ratio above 1+eps -> clipped to (1+eps)*A
    D = torch.tensor(1.0)                       # ratio = e ~ 2.718
    A, eps = 2.0, 0.2
    loss = float(ppo_turn_loss(D, A, eps))
    assert abs(loss - (1.0 + eps) * A) < 1e-5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_ppo_segment OK")
