from eval_four_benign import consensus_reached, paired_initial_positions


def test_consensus_is_over_first_three_agents_only():
    assert consensus_reached([7, 7, 7, 7])
    assert consensus_reached([7, 7, 7, 20])
    assert not consensus_reached([7, 7, 8, 7])
    assert consensus_reached([7, 7, 8, 20], tolerance=1)


def test_initial_positions_are_deterministic_and_bounded():
    a = paired_initial_positions(9000, 17, 10, 20)
    b = paired_initial_positions(9000, 17, 10, 20)
    assert a == b
    assert len(a) == 4
    assert all(0 <= value <= 20 for value in a)
