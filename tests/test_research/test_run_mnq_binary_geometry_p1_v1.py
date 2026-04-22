from research.run_mnq_binary_geometry_p1_v1 import benjamini_hochberg


def test_bh_two_values_is_monotone() -> None:
    adj = benjamini_hochberg([0.01, 0.04])
    assert len(adj) == 2
    assert 0.0 <= adj[0] <= adj[1] <= 1.0
