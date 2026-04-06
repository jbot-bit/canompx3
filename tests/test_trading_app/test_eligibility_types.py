"""Tests for trading_app.eligibility.types — immutable data structures.

These tests prove:
- All nine ConditionStatus enum values exist and are string-valued
- ConditionRecord is frozen (immutable)
- is_blocking / is_resolved derived properties behave correctly
- EligibilityReport.summary counts every condition exactly once
- EligibilityReport.conditions_by_category filters correctly
"""

from __future__ import annotations

from datetime import date

import pytest

from trading_app.eligibility.types import (
    ConditionCategory,
    ConditionRecord,
    ConditionStatus,
    ConfidenceTier,
    EligibilityReport,
    FreshnessStatus,
    OverallStatus,
    ResolvesAt,
)


class TestConditionStatus:
    """The nine explicit statuses must exist and be distinct."""

    def test_all_nine_statuses_present(self):
        expected = {
            "PASS",
            "FAIL",
            "PENDING",
            "DATA_MISSING",
            "NOT_APPLICABLE_INSTRUMENT",
            "NOT_APPLICABLE_SESSION",
            "NOT_APPLICABLE_DIRECTION",
            "RULES_NOT_LOADED",
            "STALE_VALIDATION",
        }
        actual = {s.value for s in ConditionStatus}
        assert actual == expected

    def test_statuses_are_string_valued(self):
        for s in ConditionStatus:
            assert isinstance(s.value, str)


class TestConditionRecord:
    """ConditionRecord is immutable and derives blocking/resolved state."""

    def _make_record(self, status: ConditionStatus, category: ConditionCategory = ConditionCategory.PRE_SESSION):
        return ConditionRecord(
            name="test",
            category=category,
            status=status,
            resolves_at=ResolvesAt.STARTUP,
            source_filter="TEST",
        )

    def test_is_frozen(self):
        rec = self._make_record(ConditionStatus.PASS)
        with pytest.raises((AttributeError, TypeError)):
            rec.name = "changed"  # type: ignore[misc]

    def test_pass_is_not_blocking(self):
        assert self._make_record(ConditionStatus.PASS).is_blocking is False

    def test_fail_is_blocking(self):
        assert self._make_record(ConditionStatus.FAIL).is_blocking is True

    def test_data_missing_is_blocking(self):
        assert self._make_record(ConditionStatus.DATA_MISSING).is_blocking is True

    def test_pending_is_not_blocking(self):
        assert self._make_record(ConditionStatus.PENDING).is_blocking is False

    def test_not_applicable_variants_are_not_blocking(self):
        for status in (
            ConditionStatus.NOT_APPLICABLE_INSTRUMENT,
            ConditionStatus.NOT_APPLICABLE_SESSION,
            ConditionStatus.NOT_APPLICABLE_DIRECTION,
        ):
            assert self._make_record(status).is_blocking is False

    def test_rules_not_loaded_is_not_blocking(self):
        # Treat as warning, not hard block — trader must decide
        assert self._make_record(ConditionStatus.RULES_NOT_LOADED).is_blocking is False

    def test_stale_validation_is_not_blocking(self):
        assert self._make_record(ConditionStatus.STALE_VALIDATION).is_blocking is False

    def test_pending_is_not_resolved(self):
        assert self._make_record(ConditionStatus.PENDING).is_resolved is False

    def test_not_applicable_direction_is_not_resolved(self):
        # Will resolve at break detection
        assert self._make_record(ConditionStatus.NOT_APPLICABLE_DIRECTION).is_resolved is False

    def test_pass_is_resolved(self):
        assert self._make_record(ConditionStatus.PASS).is_resolved is True


class TestEligibilityReport:
    """EligibilityReport summary and filtering work correctly."""

    def _make_report(self, conditions: list[ConditionRecord]) -> EligibilityReport:
        return EligibilityReport(
            strategy_id="MNQ_TEST_E2_RR2.0_CB1_NO_FILTER",
            instrument="MNQ",
            session="TEST",
            entry_model="E2",
            trading_day=date(2026, 4, 7),
            as_of_timestamp=None,
            freshness_status=FreshnessStatus.FRESH,
            conditions=tuple(conditions),
            overall_status=OverallStatus.ELIGIBLE,
        )

    def test_summary_counts_all_conditions(self):
        conds = [
            ConditionRecord(
                name="a",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="A",
            ),
            ConditionRecord(
                name="b",
                category=ConditionCategory.INTRA_SESSION,
                status=ConditionStatus.PENDING,
                resolves_at=ResolvesAt.ORB_FORMATION,
                source_filter="B",
            ),
            ConditionRecord(
                name="c",
                category=ConditionCategory.OVERLAY,
                status=ConditionStatus.FAIL,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="C",
            ),
        ]
        report = self._make_report(conds)
        summary = report.summary
        assert summary[ConditionStatus.PASS] == 1
        assert summary[ConditionStatus.PENDING] == 1
        assert summary[ConditionStatus.FAIL] == 1
        # Sum should equal total conditions — nothing is silently dropped
        assert sum(summary.values()) == 3

    def test_summary_includes_all_nine_statuses(self):
        report = self._make_report([])
        summary = report.summary
        assert set(summary.keys()) == set(ConditionStatus)

    def test_blocking_conditions_filters_correctly(self):
        conds = [
            ConditionRecord(
                name="pass",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="A",
            ),
            ConditionRecord(
                name="fail",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.FAIL,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="B",
            ),
            ConditionRecord(
                name="data_missing",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.DATA_MISSING,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="C",
            ),
        ]
        report = self._make_report(conds)
        blocking = report.blocking_conditions
        assert len(blocking) == 2
        names = {c.name for c in blocking}
        assert names == {"fail", "data_missing"}

    def test_conditions_by_category_filters_correctly(self):
        conds = [
            ConditionRecord(
                name="pre",
                category=ConditionCategory.PRE_SESSION,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="A",
            ),
            ConditionRecord(
                name="intra",
                category=ConditionCategory.INTRA_SESSION,
                status=ConditionStatus.PENDING,
                resolves_at=ResolvesAt.ORB_FORMATION,
                source_filter="B",
            ),
            ConditionRecord(
                name="overlay",
                category=ConditionCategory.OVERLAY,
                status=ConditionStatus.PASS,
                resolves_at=ResolvesAt.STARTUP,
                source_filter="C",
            ),
        ]
        report = self._make_report(conds)
        assert len(report.conditions_by_category(ConditionCategory.PRE_SESSION)) == 1
        assert len(report.conditions_by_category(ConditionCategory.INTRA_SESSION)) == 1
        assert len(report.conditions_by_category(ConditionCategory.OVERLAY)) == 1
        assert len(report.conditions_by_category(ConditionCategory.DIRECTIONAL)) == 0


class TestEnumsExist:
    """Load-bearing enums must exist with expected values."""

    def test_all_categories(self):
        assert ConditionCategory.PRE_SESSION
        assert ConditionCategory.INTRA_SESSION
        assert ConditionCategory.OVERLAY
        assert ConditionCategory.DIRECTIONAL

    def test_all_resolves_at(self):
        assert ResolvesAt.STARTUP
        assert ResolvesAt.ORB_FORMATION
        assert ResolvesAt.BREAK_DETECTED
        assert ResolvesAt.CONFIRM_COMPLETE
        assert ResolvesAt.TRADE_ENTERED

    def test_all_freshness(self):
        assert FreshnessStatus.FRESH
        assert FreshnessStatus.PRIOR_DAY
        assert FreshnessStatus.STALE
        assert FreshnessStatus.NO_DATA

    def test_all_overall_status(self):
        assert OverallStatus.ELIGIBLE
        assert OverallStatus.INELIGIBLE
        assert OverallStatus.DATA_MISSING
        assert OverallStatus.NEEDS_LIVE_DATA

    def test_all_confidence_tiers(self):
        assert ConfidenceTier.PROVEN
        assert ConfidenceTier.PLAUSIBLE
        assert ConfidenceTier.LEGACY
        assert ConfidenceTier.UNKNOWN
