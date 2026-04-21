"""Locks the truth table for has_missing_input per
docs/runtime/stages/phase-6e-detectors-input-validation-refactor.md.

Contract: True iff any value is None or NaN. Inf passes through
(extreme-value alerts MUST fire, not hide).
"""

import math

from trading_app.live.detectors._validation import has_missing_input


def test_none_is_missing():
    assert has_missing_input(None) is True


def test_nan_is_missing():
    assert has_missing_input(float("nan")) is True


def test_zero_is_not_missing():
    assert has_missing_input(0.0) is False


def test_negative_float_is_not_missing():
    assert has_missing_input(-3.42) is False


def test_positive_float_is_not_missing():
    assert has_missing_input(0.30) is False


def test_positive_infinity_is_not_missing():
    assert has_missing_input(math.inf) is False


def test_negative_infinity_is_not_missing():
    assert has_missing_input(-math.inf) is False


def test_multiple_all_finite_is_not_missing():
    assert has_missing_input(0.60, 0.40, 50.0) is False


def test_multiple_with_one_none_is_missing():
    assert has_missing_input(0.60, None, 50.0) is True


def test_multiple_with_one_nan_is_missing():
    assert has_missing_input(0.60, 0.40, float("nan")) is True


def test_empty_call_is_not_missing():
    # Zero args -> no missing values by definition. Defensive edge.
    assert has_missing_input() is False
