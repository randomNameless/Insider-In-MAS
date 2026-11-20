from paper_wmsr import retained_neighbor_indices, self_check


def test_self_check():
    self_check()


def test_each_agent_gets_median_received_neighbor_on_complete_n4_f1():
    positions = [0, 4, 13, 20]
    assert retained_neighbor_indices(positions, 0, 1) == [2]
    assert retained_neighbor_indices(positions, 1, 1) == [2]
    assert retained_neighbor_indices(positions, 2, 1) == [1]
    assert retained_neighbor_indices(positions, 3, 1) == [1]


def test_f0_keeps_all_neighbors():
    assert retained_neighbor_indices([0, 4, 13, 20], 2, 0) == [0, 1, 3]

