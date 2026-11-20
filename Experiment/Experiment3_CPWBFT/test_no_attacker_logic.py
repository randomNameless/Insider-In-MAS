from eval_exp3_no_attacker import consensus_reached, visible_others


def test_consensus_scores_first_three_only():
    assert consensus_reached([7, 7, 7, 20])
    assert not consensus_reached([7, 8, 7, 7])


def test_only_first_three_agents_filter():
    confidence = [0.8, 0.6, 0.9, 0.4]
    assert visible_others(0, confidence, 0.0) == [2]
    assert visible_others(3, confidence, 0.0) == [0, 1, 2]
