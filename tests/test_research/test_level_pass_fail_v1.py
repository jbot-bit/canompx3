"""Tests for research_level_pass_fail_v1 helper semantics."""

from research.research_level_pass_fail_v1 import signed_direction, signed_response


def test_signed_direction_close_through_from_below_is_continuation_up():
    assert signed_direction("below", "close_through") == 1


def test_signed_direction_wick_fail_from_below_is_reversal_down():
    assert signed_direction("below", "wick_fail") == -1


def test_signed_direction_close_through_from_above_is_continuation_down():
    assert signed_direction("above", "close_through") == -1


def test_signed_response_normalizes_by_atr():
    val = signed_response(event_close=100.0, future_close=101.0, atr_20=2.0, direction=1)
    assert val == 0.5


def test_signed_response_inverts_for_short_direction():
    val = signed_response(event_close=100.0, future_close=99.0, atr_20=2.0, direction=-1)
    assert val == 0.5


def test_signed_response_fail_closed_on_bad_atr():
    assert signed_response(event_close=100.0, future_close=101.0, atr_20=0.0, direction=1) is None
