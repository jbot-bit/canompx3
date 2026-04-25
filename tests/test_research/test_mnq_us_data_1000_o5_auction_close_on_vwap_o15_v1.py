from research.mnq_us_data_1000_o5_auction_close_on_vwap_o15_v1 import (
    _compute_clv,
    _feature_fire_for_row,
)


def test_compute_clv_returns_none_on_flat_range() -> None:
    assert _compute_clv(100.0, 100.0, 100.0, 100.0) is None


def test_feature_fire_long_requires_close_near_high() -> None:
    assert _feature_fire_for_row("long", 0.80) == 1
    assert _feature_fire_for_row("long", 0.74) == 0


def test_feature_fire_short_requires_close_near_low() -> None:
    assert _feature_fire_for_row("short", 0.20) == 1
    assert _feature_fire_for_row("short", 0.26) == 0
