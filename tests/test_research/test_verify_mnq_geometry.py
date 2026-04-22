from research.verify_mnq_geometry import assign_bucket


def test_assign_bucket_boundaries() -> None:
    assert assign_bucket(-0.01) == "co_located_break"
    assert assign_bucket(0.0) == "co_located_break"
    assert assign_bucket(0.5) == "choked"
    assert assign_bucket(1.0) == "choked"
    assert assign_bucket(1.5) == "mid_clearance"
    assert assign_bucket(2.0) == "mid_clearance"
    assert assign_bucket(2.01) == "open_air"
