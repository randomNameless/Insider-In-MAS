from topology_logic import (TOPOLOGIES, adjacency, build_topology, graph_metrics,
                            is_connected, topology_benign_personalities)


def test_all_topologies_are_connected_and_simple():
    for name in TOPOLOGIES:
        edges = build_topology(name, 9000)
        assert len(edges) == len(set(edges))
        assert all(a < b for a, b in edges)
        assert is_connected(6, edges)


def test_expected_degrees():
    complete = adjacency(6, build_topology("complete", 1))
    ring = adjacency(6, build_topology("ring", 1))
    assert {len(row) for row in complete} == {5}
    assert {len(row) for row in ring} == {2}


def test_star_placement_and_benign_connectivity():
    center = graph_metrics(6, build_topology("star_attacker_center", 1), 5)
    leaf = graph_metrics(6, build_topology("star_attacker_leaf", 1), 5)
    assert center["attacker_degree"] == 5
    assert not center["benign_subgraph_connected"]
    assert leaf["attacker_degree"] == 1
    assert leaf["benign_subgraph_connected"]


def test_path_attacker_placements():
    endpoint = graph_metrics(6, build_topology("path_attacker_endpoint", 1), 5)
    middle = graph_metrics(6, build_topology("path_attacker_middle", 1), 5)
    assert endpoint["n_edges"] == middle["n_edges"] == 5
    assert endpoint["attacker_degree"] == 1
    assert endpoint["benign_subgraph_connected"]
    assert middle["attacker_degree"] == 2
    assert not middle["benign_subgraph_connected"]


def test_roster_and_balanced_roles():
    from collections import Counter
    roster = Counter(stubborn=1, suggestible=2, neutral=2)
    assignments = [topology_benign_personalities(9000, ep) for ep in range(30)]
    assert all(Counter(row) == roster for row in assignments)
    assert len({tuple(row) for row in assignments}) == 30
    for slot in range(5):
        assert Counter(row[slot] for row in assignments) == Counter(
            stubborn=6, suggestible=12, neutral=12)
    assert assignments[0] == topology_benign_personalities(9000, 0)
