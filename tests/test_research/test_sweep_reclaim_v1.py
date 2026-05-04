"""Tests for research_sweep_reclaim_v1 helper semantics."""

from research.research_sweep_reclaim_v1 import reclaim_direction, signed_response


def test_reclaim_direction_from_below_is_short_reversal():
    assert reclaim_direction("below") == -1


def test_reclaim_direction_from_above_is_long_reversal():
    assert reclaim_direction("above") == 1


def test_signed_response_normalizes_by_atr():
    val = signed_response(reclaim_close=100.0, future_close=99.0, atr_20=2.0, direction=-1)
    assert val == 0.5


def test_signed_response_fail_closed_on_bad_atr():
    assert signed_response(reclaim_close=100.0, future_close=101.0, atr_20=0.0, direction=1) is None
