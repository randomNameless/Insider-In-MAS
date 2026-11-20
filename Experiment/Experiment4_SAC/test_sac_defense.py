from sac_defense import parse_score, retained_neighbors


def test_score_parser_matches_appendix_fallback():
    assert parse_score("0.8") == (0.8, False)
    assert parse_score("Score: 4.2")[0] == 1.0
    assert parse_score("invalid") == (0.5, True)


def test_filter_only_below_self_and_at_most_f():
    # self=0.8; candidates below self are agent 1=0.1 and agent 3=0.2.
    # F=1 removes only the lowest (agent 1), retaining agents 2 and 3.
    assert retained_neighbors([0.8, 0.1, 0.9, 0.2], 0, 1) == [2, 3]
    assert retained_neighbors([0.5, 0.5, 0.5, 0.9], 0, 1) == [1, 2, 3]
    assert retained_neighbors([0.8, 0.1, 0.9, 0.2], 0, 2) == [2]
