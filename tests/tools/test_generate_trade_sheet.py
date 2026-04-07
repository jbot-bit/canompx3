"""Tests for scripts.tools.generate_trade_sheet."""

from datetime import date
from pathlib import Path

import pytest

from scripts.tools.generate_trade_sheet import (
    FitnessCheckResult,
    _check_fitness,
    _enrich_trades_with_eligibility,
    _fitness_badge,
    _prefetch_feature_rows,
    _status_badge_from_eligibility,
)


def test_check_fitness_caches_success(monkeypatch):
    calls = {"count": 0}

    class DummyFitness:
        fitness_status = "FIT"

    def fake_compute_fitness(strategy_id, db_path):
        calls["count"] += 1
        return DummyFitness()

    monkeypatch.setattr("scripts.tools.generate_trade_sheet.compute_fitness", fake_compute_fitness)

    cache = {}
    first = _check_fitness("SID_1", Path("gold.db"), cache)
    second = _check_fitness("SID_1", Path("gold.db"), cache)

    assert first == FitnessCheckResult(status="FIT", error=None)
    assert second == first
    assert calls["count"] == 1


def test_check_fitness_returns_unknown_with_error(monkeypatch):
    def fake_compute_fitness(strategy_id, db_path):
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.tools.generate_trade_sheet.compute_fitness", fake_compute_fitness)

    result = _check_fitness("SID_2", Path("gold.db"), {})

    assert result.status == "UNKNOWN"
    assert result.error == "RuntimeError: boom"


def test_fitness_badge_unknown_is_not_decay():
    badge = _fitness_badge("UNKNOWN")

    assert "badge-unknown" in badge
    assert "badge-decay" not in badge


# ──────────────────────────────────────────────────────────────────────
# Trade-book canonicalization tests (Phase 2 of eligibility-context plan)
# Verifies that the trade sheet is a thin consumer of
# trading_app.eligibility.builder.build_eligibility_report and that the
# parallel-model classifier has been deleted.
# Design: docs/plans/2026-04-07-trade-book-canonicalization-design.md
# ──────────────────────────────────────────────────────────────────────


def _trade(overall, **overrides):
    """Build a minimal trade dict with the six elig_* keys for badge tests."""
    base = {
        "strategy_id": "MNQ_NYSE_CLOSE_E2_RR2.0_CB1_NO_FILTER",
        "instrument": "MNQ",
        "elig_overall": overall,
        "elig_blocking": (),
        "elig_pending": (),
        "elig_stale": False,
        "elig_size_mult": 1.0,
        "elig_freshness": "FRESH",
    }
    base.update(overrides)
    return base


class TestStatusBadgeFromEligibility:
    """Pure-function display mapping from elig_* keys to HTML fragment dict.

    No DB, no canonical builder calls — exercises the badge adapter in
    isolation across all OverallStatus values plus the UNKNOWN fallback.
    """

    def test_eligible_produces_green_check_no_row_class(self):
        view = _status_badge_from_eligibility(_trade("ELIGIBLE"))
        assert "badge-filter-active" in view["badge_html"]
        assert view["row_class_suffix"] == ""
        assert view["pills_html"] == ""
        assert view["tooltip_parts"] == []

    def test_ineligible_produces_inactive_badge_and_row_inactive(self):
        view = _status_badge_from_eligibility(
            _trade(
                "INELIGIBLE",
                elig_blocking=("PDR_ATR < 0.20", "ORB size >= 5 pts"),
            )
        )
        assert "badge-filter-check" in view["badge_html"]
        assert "INACTIVE" in view["badge_html"]
        assert view["row_class_suffix"] == " row-inactive"
        # Tooltip lists the blocking conditions
        assert any("blocked by:" in p for p in view["tooltip_parts"])
        assert any("PDR_ATR" in p for p in view["tooltip_parts"])

    def test_needs_live_data_produces_verify_badge_no_row_class(self):
        view = _status_badge_from_eligibility(
            _trade(
                "NEEDS_LIVE_DATA",
                elig_pending=("ORB size threshold", "cost ratio < 10%"),
            )
        )
        assert "badge-filter-check" in view["badge_html"]
        assert "VERIFY" in view["badge_html"]
        assert view["row_class_suffix"] == ""
        assert any("waiting on:" in p for p in view["tooltip_parts"])

    def test_data_missing_produces_data_badge_and_row_inactive(self):
        view = _status_badge_from_eligibility(
            _trade("DATA_MISSING", elig_freshness="NO_DATA")
        )
        assert "badge-filter-missing" in view["badge_html"]
        assert "DATA" in view["badge_html"]
        assert view["row_class_suffix"] == " row-inactive"
        assert any("feature data missing" in p for p in view["tooltip_parts"])

    def test_unknown_falls_back_to_verify_with_error_in_tooltip(self):
        view = _status_badge_from_eligibility(
            _trade(
                "UNKNOWN",
                elig_error="ValueError: Unknown filter_type 'XYZ_S075'",
            )
        )
        assert "badge-filter-check" in view["badge_html"]
        assert "VERIFY" in view["badge_html"]
        # Error message surfaces in the tooltip
        assert any("eligibility error:" in p for p in view["tooltip_parts"])
        assert any("Unknown filter_type" in p for p in view["tooltip_parts"])

    def test_stale_validation_pill_appears_alongside_main_badge(self):
        view = _status_badge_from_eligibility(_trade("ELIGIBLE", elig_stale=True))
        # Main badge unchanged
        assert "badge-filter-active" in view["badge_html"]
        # STALE pill is additive
        assert "pill-stale" in view["pills_html"]
        assert "STALE" in view["pills_html"]

    def test_half_size_pill_appears_when_size_mult_below_one(self):
        view = _status_badge_from_eligibility(
            _trade("ELIGIBLE", elig_size_mult=0.5)
        )
        assert "badge-filter-active" in view["badge_html"]
        assert "pill-half" in view["pills_html"]
        assert "HALF" in view["pills_html"]

    def test_both_pills_can_appear_independently_of_main_badge(self):
        view = _status_badge_from_eligibility(
            _trade("ELIGIBLE", elig_stale=True, elig_size_mult=0.5)
        )
        assert "pill-stale" in view["pills_html"]
        assert "pill-half" in view["pills_html"]

    def test_freshness_prior_day_appears_in_tooltip(self):
        view = _status_badge_from_eligibility(
            _trade("ELIGIBLE", elig_freshness="PRIOR_DAY")
        )
        assert any("yesterday" in p for p in view["tooltip_parts"])

    def test_freshness_stale_appears_in_tooltip(self):
        view = _status_badge_from_eligibility(
            _trade("ELIGIBLE", elig_freshness="STALE")
        )
        assert any("STALE" in p for p in view["tooltip_parts"])

    def test_pure_function_does_not_mutate_input(self):
        trade = _trade("INELIGIBLE", elig_blocking=("X",))
        before = dict(trade)
        _status_badge_from_eligibility(trade)
        assert trade == before, "badge helper must not mutate the input trade dict"


class TestEnrichTradesWithEligibility:
    """Verifies the canonical-builder consumer attaches expected keys
    and falls back loudly on exception."""

    def test_fallback_on_exception_attaches_unknown_and_error_key(self, monkeypatch, capsys):
        """When build_eligibility_report raises, every trade still gets
        a complete set of elig_* keys with overall=UNKNOWN and an
        elig_error string. Mirrors the existing _check_fitness fallback."""

        def boom(*args, **kwargs):
            raise RuntimeError("simulated canonical failure")

        monkeypatch.setattr(
            "scripts.tools.generate_trade_sheet.build_eligibility_report",
            boom,
        )

        trades = [
            {
                "strategy_id": "MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
                "instrument": "MGC",
                "aperture": 5,
            }
        ]
        _enrich_trades_with_eligibility(trades, date(2026, 4, 7), {})

        t = trades[0]
        assert t["elig_overall"] == "UNKNOWN"
        assert "RuntimeError: simulated canonical failure" in t["elig_error"]
        # Defaults are attached so the badge helper doesn't KeyError
        assert t["elig_blocking"] == ()
        assert t["elig_pending"] == ()
        assert t["elig_stale"] is False
        assert t["elig_size_mult"] == 1.0
        assert t["elig_freshness"] == ""

        # Loud failure: WARNING printed to stdout (matches _check_fitness)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "MGC_CME_REOPEN" in captured.out

    def test_success_path_attaches_six_keys(self, monkeypatch):
        """When build_eligibility_report succeeds, six elig_* keys are
        attached with values derived from the report's properties."""
        from trading_app.eligibility.types import (
            ConditionRecord,
            ConditionStatus,
            ConditionCategory,
            EligibilityReport,
            FreshnessStatus,
            OverallStatus,
            ResolvesAt,
        )

        fake_report = EligibilityReport(
            strategy_id="MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
            instrument="MGC",
            session="CME_REOPEN",
            entry_model="E2",
            trading_day=date(2026, 4, 7),
            as_of_timestamp=None,
            freshness_status=FreshnessStatus.PRIOR_DAY,
            conditions=(
                ConditionRecord(
                    name="ORB size >= 6 pts",
                    category=ConditionCategory.INTRA_SESSION,
                    status=ConditionStatus.PENDING,
                    resolves_at=ResolvesAt.ORB_FORMATION,
                ),
            ),
            overall_status=OverallStatus.NEEDS_LIVE_DATA,
        )

        def fake_build(*args, **kwargs):
            return fake_report

        monkeypatch.setattr(
            "scripts.tools.generate_trade_sheet.build_eligibility_report",
            fake_build,
        )

        trades = [
            {
                "strategy_id": "MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6",
                "instrument": "MGC",
                "aperture": 5,
            }
        ]
        _enrich_trades_with_eligibility(trades, date(2026, 4, 7), {})

        t = trades[0]
        assert t["elig_overall"] == "NEEDS_LIVE_DATA"
        assert t["elig_pending"] == ("ORB size >= 6 pts",)
        assert t["elig_blocking"] == ()
        assert t["elig_stale"] is False
        assert t["elig_size_mult"] == 1.0
        assert t["elig_freshness"] == "PRIOR_DAY"
        assert "elig_error" not in t  # success path does not attach elig_error


class TestPrefetchFeatureRows:
    """Verifies the prefetch helper opens exactly one DB connection
    and runs exactly one query per unique (instrument, aperture) pair."""

    def test_one_connection_per_call_n_queries_per_unique_triple(self, monkeypatch):
        """5 trade dicts spanning 3 unique (instrument, aperture) pairs
        must produce exactly 1 connect call and exactly 3 execute calls."""
        connect_calls = {"count": 0}
        execute_calls = {"count": 0}

        class FakeRowResult:
            def fetchone(self):
                return ("MGC", 5, 100.0, 0.5)

        class FakeConnection:
            def __init__(self):
                self.description = [
                    ("symbol",),
                    ("orb_minutes",),
                    ("atr_20_pct",),
                    ("overnight_range_pct",),
                ]

            def execute(self, sql, params):
                execute_calls["count"] += 1
                return FakeRowResult()

            def close(self):
                pass

        def fake_connect(path, read_only=False):
            connect_calls["count"] += 1
            return FakeConnection()

        monkeypatch.setattr(
            "scripts.tools.generate_trade_sheet.duckdb.connect",
            fake_connect,
        )

        trades = [
            {"instrument": "MGC", "aperture": 5},
            {"instrument": "MGC", "aperture": 5},   # dup pair
            {"instrument": "MNQ", "aperture": 5},
            {"instrument": "MNQ", "aperture": 5},   # dup pair
            {"instrument": "MES", "aperture": 5},
        ]
        result = _prefetch_feature_rows(trades, Path("dummy.db"))

        assert connect_calls["count"] == 1, "must open exactly one connection"
        assert execute_calls["count"] == 3, "must run exactly one query per unique pair"
        assert len(result) == 3
        assert ("MGC", 5) in result
        assert ("MNQ", 5) in result
        assert ("MES", 5) in result

    def test_empty_trade_list_returns_empty_dict_no_connection(self, monkeypatch):
        connect_calls = {"count": 0}

        def fake_connect(*args, **kwargs):
            connect_calls["count"] += 1
            raise AssertionError("must not connect when there are no trades")

        monkeypatch.setattr(
            "scripts.tools.generate_trade_sheet.duckdb.connect",
            fake_connect,
        )
        result = _prefetch_feature_rows([], Path("dummy.db"))
        assert result == {}
        assert connect_calls["count"] == 0


class TestEnrichTradesIntegration:
    """End-to-end smoke test against the real gold.db. Skipped gracefully
    when the DB is missing (fresh-clone CI). Verifies that the canonical
    builder produces non-UNKNOWN results for every currently deployed lane."""

    def test_deployed_lanes_resolve_through_canonical_builder(self):
        from pipeline.paths import GOLD_DB_PATH
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}; integration test skipped")

        # Build a trades list directly from prop_profiles deployed lanes
        # (avoiding collect_trades's DB roundtrips so this test stays scoped)
        trades = []
        for pid, profile in ACCOUNT_PROFILES.items():
            if not profile.active or not profile.daily_lanes:
                continue
            for lane in profile.daily_lanes:
                trades.append({
                    "strategy_id": lane.strategy_id,
                    "instrument": lane.instrument,
                    "aperture": 5,
                })

        if not trades:
            pytest.skip("no deployed lanes in prop_profiles; nothing to test")

        feature_rows = _prefetch_feature_rows(trades, Path(GOLD_DB_PATH))
        _enrich_trades_with_eligibility(trades, date(2026, 4, 7), feature_rows)

        # Every deployed trade dict carries all six eligibility keys
        for t in trades:
            assert "elig_overall" in t, f"missing elig_overall for {t['strategy_id']}"
            assert "elig_blocking" in t
            assert "elig_pending" in t
            assert "elig_stale" in t
            assert "elig_size_mult" in t
            assert "elig_freshness" in t
            assert t["elig_overall"] in (
                "ELIGIBLE",
                "INELIGIBLE",
                "DATA_MISSING",
                "NEEDS_LIVE_DATA",
                "UNKNOWN",
            ), f"unexpected elig_overall {t['elig_overall']!r} for {t['strategy_id']}"

        # Anti-drift: with the parser fix in place, NO deployed lane should
        # be UNKNOWN (which would mean the canonical builder raised). The
        # 32% UNKNOWN rate that triggered commit 35ae1fd must not return.
        unknowns = [t for t in trades if t["elig_overall"] == "UNKNOWN"]
        assert not unknowns, (
            f"deployed lanes raising in canonical builder: "
            f"{[t['strategy_id'] for t in unknowns]}"
        )
