import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker.potential import d_pair, d_range, phi


def test_phi_bounds():
    assert phi([10, 10, 10]) == 0.0
    assert abs(phi([0, 0, 20]) - (0.7 * 1.0 + 0.3 * (40 / 40))) < 1e-9
    assert 0.0 <= phi([3, 7, 15]) <= 1.0


def test_d_pair_max():
    # 3 agents: two at extremes, one anywhere -> pairwise sum in [0,40]
    assert abs(d_pair([0, 0, 20]) - 40 / 40) < 1e-9
    assert abs(d_pair([0, 20, 20]) - 40 / 40) < 1e-9


def test_d_range():
    assert d_range([5, 5, 5]) == 0.0
    assert d_range([0, 10, 20]) == 1.0


def test_attacker_excluded_is_caller_responsibility():
    # phi only ever sees benign positions; identical benign -> identical phi
    assert phi([4, 8, 12]) == phi([4, 8, 12])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("test_potential OK")
