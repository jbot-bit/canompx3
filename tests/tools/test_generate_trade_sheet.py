"""Tests for scripts.tools.generate_trade_sheet."""

from datetime import date
from pathlib import Path

import pytest

from scripts.tools.generate_trade_sheet import (
    FitnessCheckResult,
    _build_filter_universe_rows,
    _check_fitness,
    _enrich_trades_with_eligibility,
    _fitness_badge,
    _prefetch_feature_rows,
    _render_filter_universe_section,
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
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}; integration test skipped")

        # Build a trades list directly from prop_profiles deployed lanes
        # (avoiding collect_trades's DB roundtrips so this test stays scoped)
        trades = []
        for pid, profile in ACCOUNT_PROFILES.items():
            lanes = effective_daily_lanes(profile)
            if not profile.active or not lanes:
                continue
            for lane in lanes:
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


# ──────────────────────────────────────────────────────────────────────
# View B — Filter Universe Audit tests
# Design: docs/plans/2026-04-07-filter-universe-audit-design.md
# ──────────────────────────────────────────────────────────────────────


def _fu_row(status="LIVE", filter_type="COST_LT10", routed=21, deployed=2, **overrides):
    """Build a minimal filter-universe row dict for renderer tests."""
    base = {
        "filter_type": filter_type,
        "class_name": "CostRatioFilter",
        "description": "Cost < 10% of ORB",
        "confidence_tier": "",
        "validated_for": (),
        "last_revalidated": "",
        "is_stale": False,
        "routed": routed,
        "deployed": deployed,
        "status": status,
    }
    base.update(overrides)
    return base


class TestRenderFilterUniverseSection:
    """Pure renderer: list[row dict] → HTML fragment. No DB, no I/O."""

    def test_empty_rows_produces_empty_summary(self):
        html = _render_filter_universe_section([])
        assert "Filter Universe Audit" in html
        assert "0 total" in html
        assert "<table" in html

    def test_live_row_uses_row_live_class_and_live_badge(self):
        html = _render_filter_universe_section([_fu_row(status="LIVE", deployed=2)])
        assert "row-live" in html
        assert "badge-filter-live" in html
        assert ">LIVE<" in html

    def test_routed_row_uses_row_routed_class_and_check_badge(self):
        html = _render_filter_universe_section(
            [_fu_row(status="ROUTED", routed=10, deployed=0)]
        )
        assert "row-routed" in html
        # ROUTED intentionally reuses badge-filter-check (LOW-2 accepted)
        assert "badge-filter-check" in html
        assert ">ROUTED<" in html

    def test_dead_row_uses_row_dead_class_and_dead_badge(self):
        html = _render_filter_universe_section(
            [_fu_row(status="DEAD", routed=0, deployed=0)]
        )
        assert "row-dead" in html
        assert "badge-filter-dead" in html
        assert ">DEAD<" in html

    def test_empty_metadata_renders_as_em_dash(self):
        html = _render_filter_universe_section([_fu_row()])
        # Empty tier, validated_for, last_revalidated → em dash (&mdash;)
        assert "&mdash;" in html

    def test_stale_pill_appears_when_is_stale_true(self):
        html = _render_filter_universe_section(
            [_fu_row(last_revalidated="2025-01-01", is_stale=True)]
        )
        assert "pill-stale" in html
        assert "STALE" in html

    def test_validated_for_chips_cap_at_three_with_more_indicator(self):
        html = _render_filter_universe_section(
            [
                _fu_row(
                    validated_for=(
                        ("MNQ", "NYSE_CLOSE"),
                        ("MNQ", "NYSE_OPEN"),
                        ("MNQ", "US_DATA_830"),
                        ("MNQ", "CME_PRECLOSE"),
                        ("MNQ", "COMEX_SETTLE"),
                    )
                )
            ]
        )
        # First 3 shown as chips, +2 indicator for the rest
        assert "vf-chip" in html
        assert "vf-more" in html
        assert "+2" in html
        # Full list appears in tooltip
        assert "COMEX_SETTLE" in html

    def test_summary_counts_match_rows(self):
        rows = [
            _fu_row(status="LIVE", filter_type="A", deployed=2),
            _fu_row(status="LIVE", filter_type="B", deployed=1),
            _fu_row(status="ROUTED", filter_type="C", routed=5, deployed=0),
            _fu_row(status="DEAD", filter_type="D", routed=0, deployed=0),
        ]
        html = _render_filter_universe_section(rows)
        assert "4 total" in html
        assert "3 routed" in html  # LIVE + ROUTED have routed > 0? Actually LIVE has routed=21 by default, so all 3 count
        assert "2 deployed" in html  # 2 LIVE rows
        assert "3 live lanes" in html  # deployed 2 + 1 = 3

    def test_pure_function_does_not_mutate_input(self):
        row = _fu_row()
        snapshot = dict(row)
        _render_filter_universe_section([row])
        assert row == snapshot, "renderer must not mutate input row dict"


class TestBuildFilterUniverseRows:
    """Stats helper: ALL_FILTERS + ATR_VELOCITY_OVERLAY + DB + prop_profiles
    → sorted row dicts. Skips gracefully if gold.db is absent."""

    def test_returns_row_per_filter_plus_overlay(self):
        from pipeline.paths import GOLD_DB_PATH
        from trading_app.config import ALL_FILTERS

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}")

        rows = _build_filter_universe_rows(GOLD_DB_PATH, date(2026, 4, 7))
        # One row per filter + one for ATR_VELOCITY_OVERLAY
        assert len(rows) == len(ALL_FILTERS) + 1

    def test_rows_sorted_by_deployed_then_routed_then_filter_type(self):
        from pipeline.paths import GOLD_DB_PATH

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}")

        rows = _build_filter_universe_rows(GOLD_DB_PATH, date(2026, 4, 7))
        # Invariant: sort is (-deployed, -routed, filter_type)
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            key_a = (-a["deployed"], -a["routed"], a["filter_type"])
            key_b = (-b["deployed"], -b["routed"], b["filter_type"])
            assert key_a <= key_b, (
                f"sort order violated at index {i}: "
                f"{a['filter_type']} (d={a['deployed']}, r={a['routed']}) "
                f"vs {b['filter_type']} (d={b['deployed']}, r={b['routed']})"
            )

    def test_live_routed_dead_classification_is_exhaustive_and_exclusive(self):
        from pipeline.paths import GOLD_DB_PATH

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}")

        rows = _build_filter_universe_rows(GOLD_DB_PATH, date(2026, 4, 7))
        # Every row has exactly one of the three statuses, and the status
        # matches the deployed/routed counts.
        for r in rows:
            assert r["status"] in {"LIVE", "ROUTED", "DEAD"}
            if r["status"] == "LIVE":
                assert r["deployed"] > 0, f"LIVE but deployed=0: {r['filter_type']}"
            elif r["status"] == "ROUTED":
                assert r["deployed"] == 0 and r["routed"] > 0, (
                    f"ROUTED but counts wrong: {r['filter_type']}"
                )
            else:  # DEAD
                assert r["deployed"] == 0 and r["routed"] == 0, (
                    f"DEAD but counts nonzero: {r['filter_type']}"
                )

    def test_deployed_count_matches_active_profile_lane_count(self):
        from pipeline.paths import GOLD_DB_PATH
        from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}")

        rows = _build_filter_universe_rows(GOLD_DB_PATH, date(2026, 4, 7))
        sum_deployed = sum(r["deployed"] for r in rows)

        expected = 0
        for _pid, profile in ACCOUNT_PROFILES.items():
            if profile.active:
                expected += len(effective_daily_lanes(profile))

        assert sum_deployed == expected, (
            f"View B deployed count ({sum_deployed}) does not match "
            f"active profile lane total ({expected})"
        )

    def test_routed_count_matches_active_validated_setups_query(self):
        import duckdb

        from pipeline.paths import GOLD_DB_PATH

        if not Path(GOLD_DB_PATH).exists():
            pytest.skip(f"gold.db not present at {GOLD_DB_PATH}")

        rows = _build_filter_universe_rows(GOLD_DB_PATH, date(2026, 4, 7))

        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM validated_setups WHERE LOWER(status)='active'"
            ).fetchone()
            assert row is not None
            expected = row[0]
        finally:
            con.close()

        # Note: multiple View B rows can map to the same filter_type across
        # ALL_FILTERS entries — but every active validated strategy's
        # filter_type is represented exactly once in ALL_FILTERS, so the
        # per-row routed sum should equal the active strategy count.
        sum_routed = sum(r["routed"] for r in rows)
        assert sum_routed == expected

    def test_stale_threshold_uses_canonical_180_day_constant(self):
        """Gate 1 LOW-1 fix: literal 180 replaced with
        VALIDATION_FRESHNESS_DAYS import. This test pins the delegation."""
        from trading_app.eligibility.builder import VALIDATION_FRESHNESS_DAYS

        # The canonical constant is 180. If this ever changes, the View B
        # helper inherits it automatically because of the import-driven
        # delegation.
        assert VALIDATION_FRESHNESS_DAYS == 180
