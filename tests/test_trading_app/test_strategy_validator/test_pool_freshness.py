"""Tests for ``check_validator_pool_freshness`` (Stage 3 advisory drift check).

The check reports drift between a row's frozen ``discovery_k`` and the live
per-session pool size in ``experimental_strategies``. Frozen K is the
audit-trail anchor (set on first promotion only; see strategy_validator.py
L2216-2219); live K shifts as peer instruments rerun discovery.

These tests inject a fake DuckDB-like connection so they don't depend on real
gold.db state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pipeline.check_drift import check_validator_pool_freshness


class _FakeCursor:
    def __init__(self, rows: list[Any]):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeCon:
    """Returns canned rows depending on which SELECT was issued.

    The check issues two distinct SELECTs:
      1. validated_setups (filtered by promoted_at >= now() - INTERVAL 7 DAY)
      2. experimental_strategies grouped by (instrument, orb_label)
    We discriminate by checking for "validated_setups" / "experimental_strategies"
    in the SQL string -- crude but stable for the check's current shape.
    """

    def __init__(self, validated_rows: list, experimental_rows: list):
        self._validated = validated_rows
        self._experimental = experimental_rows

    def execute(self, sql: str, _params=None):  # noqa: D401
        if "validated_setups" in sql:
            return _FakeCursor(self._validated)
        if "experimental_strategies" in sql:
            return _FakeCursor(self._experimental)
        return _FakeCursor([])

    def close(self):
        pass


class TestPoolFreshness:
    def test_no_drift_yields_no_violations(self):
        """frozen_k == live_k -> no violation."""
        con = _FakeCon(
            validated_rows=[
                ("MGC_TEST_E2_RR1.0_CB1_NO_FILTER", "MGC", "CME_REOPEN", 5300),
            ],
            experimental_rows=[("MGC", "CME_REOPEN", 5300)],
        )
        violations = check_validator_pool_freshness(con=con)
        assert violations == [], f"unexpected: {violations}"

    def test_drift_above_threshold_yields_violation(self):
        """frozen_k=5300, live_k=2580 -> 51.3% drift > 10% default -> reports."""
        con = _FakeCon(
            validated_rows=[
                ("MGC_TEST_E2_RR1.0_CB1_NO_FILTER", "MGC", "CME_REOPEN", 5300),
            ],
            experimental_rows=[("MGC", "CME_REOPEN", 2580)],
        )
        violations = check_validator_pool_freshness(con=con)
        assert any("POOL-FRESHNESS" in v and "MGC_TEST" in v and "5300" in v for v in violations), (
            f"expected drift violation, got: {violations}"
        )

    def test_drift_below_threshold_silent(self):
        """5% drift below 10% threshold -> no violation."""
        con = _FakeCon(
            validated_rows=[("MGC_T_E2_RR1.0_CB1_NF", "MGC", "CME_REOPEN", 1000)],
            experimental_rows=[("MGC", "CME_REOPEN", 950)],
        )
        violations = check_validator_pool_freshness(con=con)
        assert violations == [], f"unexpected: {violations}"

    def test_custom_threshold_respected(self):
        """drift_threshold=0.01 trips on 5% drift."""
        con = _FakeCon(
            validated_rows=[("MGC_T_E2_RR1.0_CB1_NF", "MGC", "CME_REOPEN", 1000)],
            experimental_rows=[("MGC", "CME_REOPEN", 950)],
        )
        violations = check_validator_pool_freshness(con=con, drift_threshold=0.01)
        assert any("POOL-FRESHNESS" in v for v in violations)

    def test_missing_live_pool_skipped(self):
        """If a row's (instrument, session) has no live pool entry, skip silently
        (don't raise -- that would be a different kind of bug to catch)."""
        con = _FakeCon(
            validated_rows=[
                ("MGC_T_E2_RR1.0_CB1_NF", "MGC", "RETIRED_SESSION", 5300),
            ],
            experimental_rows=[("MGC", "CME_REOPEN", 100)],  # different session
        )
        violations = check_validator_pool_freshness(con=con)
        assert violations == [], f"unexpected: {violations}"

    def test_zero_frozen_k_skipped(self):
        """Defensive: frozen_k=0 must not raise ZeroDivisionError."""
        con = _FakeCon(
            validated_rows=[("X_TEST_E2_RR1.0_CB1_NF", "MGC", "CME_REOPEN", 0)],
            experimental_rows=[("MGC", "CME_REOPEN", 100)],
        )
        violations = check_validator_pool_freshness(con=con)
        assert violations == [], f"unexpected: {violations}"


class TestPoolFreshnessIdempotency:
    """Calling the check twice in succession yields identical results --
    the check itself is read-only, so nothing should mutate.

    This is the plan's "validator idempotency" test in spirit -- the validator
    itself can't be invoked safely from a unit test (it requires a populated
    DB and writes), so we assert the property on the drift check that
    surfaces validator-induced drift.
    """

    def test_consecutive_runs_identical(self):
        con = _FakeCon(
            validated_rows=[
                ("MGC_T_E2_RR1.0_CB1_NF", "MGC", "CME_REOPEN", 5300),
                ("MES_T_E2_RR1.0_CB1_NF", "MES", "NYSE_OPEN", 1000),
            ],
            experimental_rows=[
                ("MGC", "CME_REOPEN", 2580),
                ("MES", "NYSE_OPEN", 950),
            ],
        )
        v1 = check_validator_pool_freshness(con=con)
        v2 = check_validator_pool_freshness(con=con)
        assert v1 == v2, "drift check is non-idempotent"
