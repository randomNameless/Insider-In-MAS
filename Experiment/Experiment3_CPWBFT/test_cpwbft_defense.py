from types import SimpleNamespace

import pytest

from cpwbft_defense import (
    Experiment3CPWBFTDefense, retained_confident_indices, self_check)


def test_self_check():
    self_check()


def test_threshold_and_prompt_order():
    assert retained_confident_indices([0.8, 0.6, 0.9, 0.4], 0) == [2]
    assert retained_confident_indices([0.8, 0.6, 0.9, 0.4], 0, 0.2) == [1, 2]


def test_empty_filter_keeps_one_most_confident_neighbor():
    assert retained_confident_indices([0.9, 0.1, 0.2, 0.3], 0) == [3]


def test_defense_reads_state_confidences():
    state = SimpleNamespace(confidences=[0.4, 0.7, 0.5, 0.8])
    assert Experiment3CPWBFTDefense().visible_others(1, state) == [3]


def test_invalid_confidence_or_margin_is_rejected():
    with pytest.raises(ValueError):
        retained_confident_indices([0.5, 1.1], 0)
    with pytest.raises(ValueError):
        retained_confident_indices([0.5, 0.4], 0, margin=-0.1)
