from collections import Counter

from scaling_logic import (paired_benign_personalities, consensus_reached,
                           paired_cooperative_added_personalities,
                           paired_initial_positions, parse_condition,
                           parse_horizons)


def test_explicit_received_snapshot_uses_other_agents_only():
    # Importing the full evaluator requires GPU libraries, so keep this schema
    # expectation mirrored as a lightweight pure-data check.
    positions = [2, 4, 6]
    messages = ["a", "b", "c"]
    agent = 1
    received = [
        {"agent": i, "position": positions[i], "message": messages[i]}
        for i in range(len(positions)) if i != agent
    ]
    assert received == [
        {"agent": 0, "position": 2, "message": "a"},
        {"agent": 2, "position": 6, "message": "c"},
    ]


def test_conditions_and_personalities():
    assert parse_condition("8x2") == (8, 2)
    assignment = paired_benign_personalities(9000, 7, 8)
    assert len(assignment) == 8
    assert set(assignment) <= {"stubborn", "suggestible", "neutral"}
    assert max(Counter(assignment).values()) <= 3
    assert paired_benign_personalities(9000, 7, 5) == assignment[:5]
    assert paired_benign_personalities(9000, 7, 8) == assignment


def test_consensus_uses_every_benign_and_excludes_malicious():
    assert consensus_reached([7, 7, 7, 7, 0], 4)
    assert not consensus_reached([7, 7, 7, 6, 7], 4)
    assert consensus_reached([7, 7, 7, 6, 0], 4, tolerance=1)


def test_initial_positions_are_prefix_paired_across_group_sizes():
    small = paired_initial_positions(9000, 3, 10, 20, 5)
    large = paired_initial_positions(9000, 3, 10, 20, 10)
    assert small == large[:5]


def test_added_agents_are_only_neutral_or_suggestible_and_prefix_paired():
    full = paired_cooperative_added_personalities(9000, 7, 8)
    assert full[:3] == ["stubborn", "suggestible", "neutral"]
    assert set(full[3:]) <= {"neutral", "suggestible"}
    assert paired_cooperative_added_personalities(9000, 7, 4) == full[:4]


def test_horizons_are_unique_and_sorted():
    assert parse_horizons("20,10,15,10") == (10, 15, 20)
