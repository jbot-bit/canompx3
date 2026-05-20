"""Tests for FAST_LANE pre-flight OOS-power gate.

Covers ``_compute_expected_oos_power`` (the pure-math gate) and the
``classify()`` integration path that emits ``REJECTED_OOS_UNPOWERED``.

Literature grounding:
    - research/oos_power.py::POWER_TIERS (0.50 DIRECTIONAL_ONLY floor)
    - backtesting-methodology.md RULE 3.3 (canonical OOS-power doctrine)
"""

from __future__ import annotations

import math

import pytest

from scripts.research.fast_lane_promote_queue import (
    OOS_COHEN_D_TARGET,
    OOS_POWER_FLOOR,
    PromoteEntry,
    _compute_expected_oos_power,
    classify,
)


def _entry(*, fire: float = 0.50) -> PromoteEntry:
    """Build a minimal QUEUE-eligible PromoteEntry for classify() tests.

    Only fields touched by the OOS-power gate carry meaningful values;
    the rest are filled with safe defaults.
    """
    return PromoteEntry(
        result_md="docs/audit/results/test.md",
        strategy_id="TEST_STRAT",
        direction="pooled",
        pooled_t=4.0,
        pooled_expr=0.10,
        pooled_n=200,
        pooled_fire=fire,
        long_n=100,
        long_expr=0.10,
        long_t=3.0,
        short_n=100,
        short_expr=0.10,
        short_t=3.0,
        pooled_universe_n=400,
        long_fire=0.25,
        short_fire=0.25,
        long_side_verdict="PROMOTE_AS_STANDALONE",
        short_side_verdict="PROMOTE_AS_STANDALONE",
        pooling_artifact=False,
        revocation_sidecar=None,
        heavyweight_prereg=None,
        park_entry=None,
        status="ERROR",
        error_reason=None,
    )


class TestComputeExpectedOosPower:
    """Pure-math gate behaviour."""

    def test_high_fire_high_window_passes_floor(self):
        # 50% fire * 365 days = 183 trades at d=0.3 -> ~0.94 power
        power, n, why = _compute_expected_oos_power(0.50, 365)
        assert why is None
        assert n == pytest.approx(183, abs=1)
        assert power > OOS_POWER_FLOOR

    def test_low_fire_short_window_fails_floor(self):
        # MNQ O30 case: 14.68% fire * 136 days -> 20 trades -> ~0.25 power
        power, n, why = _compute_expected_oos_power(0.1468, 136)
        assert why is None
        assert n == 20
        assert power < OOS_POWER_FLOOR

    def test_fire_rate_nan_fails_closed(self):
        power, n, why = _compute_expected_oos_power(float("nan"), 136)
        assert why == "fire_rate unavailable or non-positive"
        assert power == 0.0
        assert n == 0

    def test_fire_rate_zero_fails_closed(self):
        power, n, why = _compute_expected_oos_power(0.0, 136)
        assert why == "fire_rate unavailable or non-positive"
        assert power == 0.0

    def test_oos_window_zero_fails_closed(self):
        power, n, why = _compute_expected_oos_power(0.50, 0)
        assert why == "OOS window not yet opened (oos_window_days <= 0)"
        assert power == 0.0

    def test_oos_window_negative_fails_closed(self):
        power, n, why = _compute_expected_oos_power(0.50, -5)
        assert why == "OOS window not yet opened (oos_window_days <= 0)"

    def test_expected_n_below_two_fails_closed(self):
        # 1% fire * 100 days -> 1 trade -> one_sample_power undefined
        power, n, why = _compute_expected_oos_power(0.01, 100)
        assert why == "expected_n_oos < 2 (one_sample_power undefined)"
        assert power == 0.0
        assert n == 1

    def test_cohen_d_target_constant_is_positive(self):
        assert OOS_COHEN_D_TARGET > 0

    def test_power_floor_in_valid_range(self):
        assert 0.0 < OOS_POWER_FLOOR <= 1.0


class TestClassifyIntegration:
    """End-to-end classify() emission of REJECTED_OOS_UNPOWERED."""

    def test_passing_cell_returns_queued(self):
        entry = _entry(fire=0.50)
        status, reason = classify(entry, oos_window_days=365)
        assert status == "QUEUED"
        assert reason is None

    def test_failing_cell_returns_rejected_oos_unpowered(self):
        entry = _entry(fire=0.10)
        status, reason = classify(entry, oos_window_days=100)
        assert status == "REJECTED_OOS_UNPOWERED"
        assert reason is not None
        assert "RULE 3.3" in reason
        assert "expected_power" in reason

    def test_missing_oos_window_fails_closed(self):
        entry = _entry(fire=0.50)
        status, reason = classify(entry, oos_window_days=None)
        assert status == "REJECTED_OOS_UNPOWERED"
        assert "OOS pre-flight unresolved" in (reason or "")

    def test_oos_window_error_propagated(self):
        entry = _entry(fire=0.50)
        status, reason = classify(entry, oos_window_days=None, oos_window_error="DB unreadable")
        assert status == "REJECTED_OOS_UNPOWERED"
        assert "DB unreadable" in (reason or "")

    def test_revoked_short_circuits_before_oos_gate(self):
        entry = _entry(fire=0.10)
        entry.revocation_sidecar = "docs/audit/results/test.revocation.md"
        # Even with a low-power fire, REVOKED takes precedence.
        status, _ = classify(entry, oos_window_days=100)
        assert status == "REVOKED"

    def test_escalated_short_circuits_before_oos_gate(self):
        entry = _entry(fire=0.10)
        entry.heavyweight_prereg = "docs/audit/hypotheses/heavy.yaml"
        status, _ = classify(entry, oos_window_days=100)
        assert status == "ESCALATED"

    def test_pooling_artifact_short_circuits_before_oos_gate(self):
        entry = _entry(fire=0.50)
        entry.pooling_artifact = True
        status, _ = classify(entry, oos_window_days=365)
        assert status == "ERROR"
