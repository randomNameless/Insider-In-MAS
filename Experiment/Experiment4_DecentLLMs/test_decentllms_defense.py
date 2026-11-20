import numpy as np

from decentllms_defense import (
    fallback_scores, geometric_median, parse_evaluation, select_worker,
    self_check,
)


def test_self_check():
    self_check()


def test_geometric_median_resists_one_extreme_evaluator():
    result = geometric_median([[10, 10], [10, 10], [100, -100]])
    assert np.linalg.norm(result - np.array([10, 10])) < 1e-4


def test_invalid_json_uses_bounded_fallback():
    result = parse_evaluation("not-json", [0, 4, 10, 20])
    assert result.shape == (4, 5)
    assert np.all((0 <= result) & (result <= 20))


def test_honest_majority_selects_consistently_high_worker():
    honest = np.full((4, 5), 5.0)
    honest[2] = 18.0
    byzantine = np.full((4, 5), 20.0)
    byzantine[2] = 0.0
    winner, _, _ = select_worker(
        [honest, honest.copy(), byzantine],
        [(0, "a"), (1, "b"), (2, "c"), (3, "d")], 0)
    assert winner == 2
