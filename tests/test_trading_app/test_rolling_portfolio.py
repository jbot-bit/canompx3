"""
Tests for rolling portfolio: window generation, stability scoring,
double-break detection, day-of-week concentration, and portfolio integration.

Uses in-memory DuckDB with synthetic data.
"""

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from pipeline.build_daily_features import detect_double_break
from scripts.infra.rolling_eval import DOUBLE_BREAK_THRESHOLD, generate_rolling_windows
from trading_app.rolling_portfolio import (
    FULL_WEIGHT_SAMPLE,
    STABLE_THRESHOLD,
    TRANSITIONING_THRESHOLD,
    FamilyResult,
    _window_weight,
    aggregate_rolling_performance,
    classify_stability,
    compute_day_of_week_stats,
    load_all_rolling_run_labels,
    load_rolling_degraded_counts,
    load_rolling_results,
    load_rolling_validated_strategies,
    make_family_id,
)

# =========================================================================
# Window generation tests
# =========================================================================


class TestWindowGeneration:
    """Test rolling window date range generation."""

    def test_basic_window_generation(self):
        windows = generate_rolling_windows(
            train_months=12,
            test_start=date(2025, 1, 1),
            test_end=date(2025, 3, 1),
        )
        assert len(windows) == 3  # Jan, Feb, Mar

    def test_window_dates_no_overlap(self):
        windows = generate_rolling_windows(
            train_months=6,
            test_start=date(2025, 1, 1),
            test_end=date(2025, 6, 1),
        )
        # Each test month starts right after previous ends
        for i in range(1, len(windows)):
            prev_test_end = windows[i - 1]["test_end"]
            curr_test_start = windows[i]["test_start"]
            assert curr_test_start > prev_test_end

    def test_no_lookahead(self):
        """Train end must be before test start."""
        windows = generate_rolling_windows(
            train_months=12,
            test_start=date(2025, 1, 1),
            test_end=date(2025, 12, 1),
        )
        for w in windows:
            assert w["train_end"] < w["test_start"], (
                f"Lookahead violation: train_end={w['train_end']} >= test_start={w['test_start']}"
            )

    def test_train_period_correct_length(self):
        windows = generate_rolling_windows(
            train_months=12,
            test_start=date(2025, 6, 1),
            test_end=date(2025, 6, 1),
        )
        assert len(windows) == 1
        w = windows[0]
        assert w["train_start"] == date(2024, 6, 1)
        assert w["train_end"] == date(2025, 5, 31)
        assert w["test_start"] == date(2025, 6, 1)
        assert w["test_end"] == date(2025, 6, 30)

    def test_run_label_format(self):
        windows = generate_rolling_windows(
            train_months=18,
            test_start=date(2025, 1, 1),
            test_end=date(2025, 1, 1),
        )
        assert windows[0]["run_label"] == "rolling_18m_2025_01"

    def test_single_window(self):
        windows = generate_rolling_windows(
            train_months=6,
            test_start=date(2025, 7, 1),
            test_end=date(2025, 7, 1),
        )
        assert len(windows) == 1


# =========================================================================
# Stability scoring tests
# =========================================================================


class TestStabilityScoring:
    """Test sample-size-weighted stability scoring."""

    def test_full_weight_at_50_trades(self):
        assert _window_weight(50) == 1.0

    def test_full_weight_above_50(self):
        assert _window_weight(100) == 1.0

    def test_partial_weight_at_20_trades(self):
        assert abs(_window_weight(20) - 0.4) < 1e-6

    def test_zero_weight_at_zero_trades(self):
        assert _window_weight(0) == 0.0

    def test_linear_scaling(self):
        assert abs(_window_weight(25) - 0.5) < 1e-6

    def test_classify_stable(self):
        assert classify_stability(0.7) == "STABLE"
        assert classify_stability(0.6) == "STABLE"

    def test_classify_transitioning(self):
        assert classify_stability(0.5) == "TRANSITIONING"
        assert classify_stability(0.3) == "TRANSITIONING"

    def test_classify_degraded(self):
        assert classify_stability(0.2) == "DEGRADED"
        assert classify_stability(0.0) == "DEGRADED"


class TestAggregatePerformance:
    """Test family-level aggregation of rolling results."""

    def _make_validated(
        self, run_label, orb_label="CME_REOPEN", em="E1", ft="ORB_G4", rr=2.0, cb=2, sample=50, exp_r=0.15, sharpe=0.1
    ):
        return {
            "run_label": run_label,
            "strategy_id": f"MGC_{orb_label}_{em}_RR{rr}_CB{cb}_{ft}",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "orb_label": orb_label,
            "entry_model": em,
            "filter_type": ft,
            "rr_target": rr,
            "confirm_bars": cb,
            "sample_size": sample,
            "win_rate": 0.45,
            "expectancy_r": exp_r,
            "sharpe_ratio": sharpe,
            "max_drawdown_r": 5.0,
            "yearly_results": "{}",
        }

    def test_all_windows_pass_high_sample(self):
        """Strategy passing all windows with 50+ trades = STABLE."""
        labels = [f"rolling_12m_2025_{m:02d}" for m in range(1, 7)]
        validated = [self._make_validated(label, sample=60) for label in labels]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 1
        assert results[0].classification == "STABLE"
        assert results[0].weighted_stability >= STABLE_THRESHOLD

    def test_low_sample_windows_penalized(self):
        """Windows with 20 trades get 0.4 weight, not 1.0."""
        labels = [f"rolling_12m_2025_{m:02d}" for m in range(1, 7)]
        # Pass 3 of 6 windows, but all with only 20 trades
        validated = [self._make_validated(label, sample=20) for label in labels[:3]]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 1
        # 3 passing windows with weight 0.4 = 1.2
        # 3 failing windows with weight 1.0 = 3.0
        # weighted_stability = 1.2 / (1.2 + 3.0) = 0.286
        r = results[0]
        assert r.classification == "DEGRADED"
        assert r.weighted_stability < TRANSITIONING_THRESHOLD

    def test_no_qualifying_strategies(self):
        """No validated strategies = empty results."""
        labels = ["rolling_12m_2025_01"]
        results = aggregate_rolling_performance([], labels, {})
        assert results == []

    def test_multiple_families(self):
        """Different families aggregated separately."""
        labels = ["rolling_12m_2025_01", "rolling_12m_2025_02"]
        validated = [
            self._make_validated("rolling_12m_2025_01", orb_label="CME_REOPEN"),
            self._make_validated("rolling_12m_2025_02", orb_label="CME_REOPEN"),
            self._make_validated("rolling_12m_2025_01", orb_label="LONDON_METALS"),
        ]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 2
        fam_ids = {r.family_id for r in results}
        assert "CME_REOPEN_E1_ORB_G4" in fam_ids
        assert "LONDON_METALS_E1_ORB_G4" in fam_ids

    def test_empty_windows_list(self):
        results = aggregate_rolling_performance([], [], {})
        assert results == []


# =========================================================================
# Double-break tests
# =========================================================================


class TestDoubleBreak:
    """Test double-break detection logic."""

    def test_both_boundaries_breached(self):
        """Double break when both ORB high and low are hit."""
        # Create bars that breach both sides
        orb_high = 2000.0
        orb_low = 1990.0

        bars = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(
                    [
                        "2025-01-06 23:05:00",  # After CME_REOPEN ORB close
                        "2025-01-06 23:10:00",
                        "2025-01-06 23:15:00",
                    ],
                    utc=True,
                ),
                "open": [1995.0, 2001.0, 1989.0],
                "high": [2001.0, 2003.0, 1991.0],
                "low": [1993.0, 1999.0, 1988.0],
                "close": [2001.0, 2001.0, 1989.0],
                "volume": [100, 100, 100],
            }
        )

        result = detect_double_break(bars, date(2025, 1, 7), "CME_REOPEN", 5, orb_high, orb_low)
        assert result is True

    def test_only_one_boundary(self):
        """No double break when only one side is breached."""
        orb_high = 2000.0
        orb_low = 1990.0

        bars = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(
                    [
                        "2025-01-06 23:05:00",
                        "2025-01-06 23:10:00",
                    ],
                    utc=True,
                ),
                "open": [1995.0, 1996.0],
                "high": [2001.0, 2002.0],  # Breaches high
                "low": [1993.0, 1994.0],  # Never breaches low (1990)
                "close": [1999.0, 2001.0],
                "volume": [100, 100],
            }
        )

        result = detect_double_break(bars, date(2025, 1, 7), "CME_REOPEN", 5, orb_high, orb_low)
        assert result is False

    def test_missing_orb_data(self):
        bars = pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])
        result = detect_double_break(bars, date(2025, 1, 7), "CME_REOPEN", 5, None, None)
        assert result is None

    def test_no_bars_in_window(self):
        """Empty window returns None."""
        bars = pd.DataFrame(
            {
                "ts_utc": pd.to_datetime(["2025-01-06 10:00:00"], utc=True),
                "open": [1995.0],
                "high": [2001.0],
                "low": [1993.0],
                "close": [1999.0],
                "volume": [100],
            }
        )
        result = detect_double_break(bars, date(2025, 1, 7), "CME_REOPEN", 5, 2000.0, 1990.0)
        assert result is None

    def test_threshold_constant(self):
        assert DOUBLE_BREAK_THRESHOLD == 0.67


# =========================================================================
# Day-of-week concentration tests
# =========================================================================


class TestDayOfWeekConcentration:
    """Test day-of-week concentration detection."""

    def test_even_distribution(self):
        """Even distribution across 5 days = 0.2 concentration."""
        day_total_r = {"Mon": 10, "Tue": 10, "Wed": 10, "Thu": 10, "Fri": 10}
        total_abs_r = sum(day_total_r.values())
        concentration = max(day_total_r.values()) / total_abs_r
        assert abs(concentration - 0.2) < 1e-6

    def test_single_day_dominance(self):
        """One day drives >50% of edge = flagged."""
        day_total_r = {"Mon": 50, "Tue": 5, "Wed": 5, "Thu": 5, "Fri": 5}
        total_abs_r = sum(day_total_r.values())
        concentration = max(day_total_r.values()) / total_abs_r
        assert concentration > 0.5


# =========================================================================
# Portfolio integration tests (with in-memory DB)
# =========================================================================


def _setup_rolling_db(tmp_path):
    """Create a temp DB with regime schema and seed rolling data."""
    db_path = tmp_path / "test_rolling.db"
    con = duckdb.connect(str(db_path))

    # Create pipeline tables (needed for imports)
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    # Create trading_app tables (uses db_path, not con)
    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=db_path)

    # Create regime tables
    from trading_app.regime.schema import init_regime_schema

    init_regime_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))

    # Seed regime_strategies with rolling labels
    labels = [f"rolling_12m_2025_{m:02d}" for m in range(1, 4)]
    for label in labels:
        con.execute(
            """
            INSERT INTO regime_strategies
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type,
             sample_size, win_rate, expectancy_r, sharpe_ratio, max_drawdown_r,
             median_risk_points, avg_risk_points,
             validation_status, validation_notes, yearly_results,
             start_date, end_date)
            VALUES (?, 'MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4', 'MGC', 'CME_REOPEN', 5,
                    2.0, 2, 'E1', 'ORB_G4',
                    55, 0.45, 0.15, 0.12, 5.0, 3.5, 3.5,
                    'PASSED', '', '{}',
                    '2024-01-01', '2024-12-31')
        """,
            [label],
        )

    # Seed regime_validated
    for label in labels:
        con.execute(
            """
            INSERT INTO regime_validated
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type,
             sample_size, win_rate, expectancy_r,
             years_tested, all_years_positive, stress_test_passed,
             sharpe_ratio, max_drawdown_r, yearly_results, status,
             start_date, end_date)
            VALUES (?, 'MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4', 'MGC', 'CME_REOPEN', 5,
                    2.0, 2, 'E1', 'ORB_G4',
                    55, 0.45, 0.15,
                    1, TRUE, TRUE,
                    0.12, 5.0, '{}', 'active',
                    '2024-01-01', '2024-12-31')
        """,
            [label],
        )

    con.commit()
    con.close()
    return db_path


class TestPortfolioIntegration:
    """Test that rolling strategies integrate into portfolio system."""

    def test_load_rolling_results(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        results = load_rolling_results(db_path, train_months=12)
        assert len(results) == 3  # 3 windows, each with 1 validated

    def test_load_run_labels(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        labels = load_all_rolling_run_labels(db_path, train_months=12)
        assert len(labels) == 3
        assert all(label.startswith("rolling_12m_") for label in labels)

    def test_load_rolling_validated_for_portfolio(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        strategies = load_rolling_validated_strategies(
            db_path,
            "MGC",
            train_months=12,
            min_weighted_score=0.0,  # Accept anything
            min_expectancy_r=0.0,
        )
        assert len(strategies) >= 1
        assert strategies[0]["source"] == "rolling"

    def test_rolling_source_tag(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        strategies = load_rolling_validated_strategies(
            db_path,
            "MGC",
            train_months=12,
            min_weighted_score=0.0,
            min_expectancy_r=0.0,
        )
        for s in strategies:
            assert s["source"] == "rolling"

    def test_rolling_avg_fields_threaded(self, tmp_path):
        """Returned dicts must contain rolling_avg_expectancy_r and rolling_weighted_stability."""
        db_path = _setup_rolling_db(tmp_path)
        strategies = load_rolling_validated_strategies(
            db_path,
            "MGC",
            train_months=12,
            min_weighted_score=0.0,
            min_expectancy_r=0.0,
        )
        assert len(strategies) >= 1
        for s in strategies:
            assert "rolling_avg_expectancy_r" in s, "missing rolling_avg_expectancy_r key"
            assert "rolling_weighted_stability" in s, "missing rolling_weighted_stability key"
            assert isinstance(s["rolling_avg_expectancy_r"], float)
            assert isinstance(s["rolling_weighted_stability"], float)

    def test_no_data_returns_empty(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        # Use a train_months that has no data
        results = load_rolling_results(db_path, train_months=6)
        assert results == []

    def test_degraded_counts(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        # Add a degraded strategy
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO regime_strategies
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type,
             sample_size, win_rate, expectancy_r, sharpe_ratio, max_drawdown_r,
             median_risk_points, avg_risk_points,
             validation_status, validation_notes, yearly_results,
             start_date, end_date)
            VALUES ('rolling_12m_2025_01', 'MGC_LONDON_METALS_E3_RR2.0_CB4_ORB_G6',
                    'MGC', 'LONDON_METALS', 5, 2.0, 4, 'E3', 'ORB_G6',
                    30, 0.40, 0.10, 0.05, 8.0, 4.0, 4.0,
                    'REJECTED', 'Auto-degraded: double-break >67% in training window',
                    '{}', '2024-01-01', '2024-12-31')
        """)
        con.commit()
        con.close()

        counts = load_rolling_degraded_counts(db_path, train_months=12)
        assert "LONDON_METALS_E3_ORB_G6" in counts
        assert counts["LONDON_METALS_E3_ORB_G6"] == 1


# =========================================================================
# compute_day_of_week_stats: orb_minutes threading (DF-04)
# =========================================================================


class TestComputeDayOfWeekStatsOrbMinutes:
    """Regression coverage for DF-04: orb_minutes must be honored by the
    daily_features eligibility query, not hardcoded to 5."""

    def _seed_db(self, tmp_path, orb_minutes: int):
        """Create minimal daily_features + orb_outcomes at the given aperture."""
        from pipeline.init_db import ORB_LABELS

        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))
        size_col_defs = ", ".join(f"orb_{lbl}_size DOUBLE" for lbl in ORB_LABELS)
        size_cols = ", ".join(f"orb_{lbl}_size" for lbl in ORB_LABELS)
        con.execute(
            f"""
            CREATE TABLE daily_features (
                symbol      TEXT,
                trading_day DATE,
                orb_minutes INTEGER,
                {size_col_defs}
            )
            """
        )
        size_placeholders = ", ".join("?" for _ in ORB_LABELS)
        for td in ("2026-01-05", "2026-01-06", "2026-01-07"):
            params = ["MGC", td, orb_minutes] + [5.0] * len(ORB_LABELS)
            con.execute(
                f"""
                INSERT INTO daily_features (symbol, trading_day, orb_minutes, {size_cols})
                VALUES (?, ?, ?, {size_placeholders})
                """,
                params,
            )
        con.execute(
            """
            CREATE TABLE orb_outcomes (
                symbol      TEXT,
                orb_label   TEXT,
                entry_model TEXT,
                orb_minutes INTEGER,
                trading_day DATE,
                pnl_r       DOUBLE
            )
            """
        )
        con.execute(
            """
            INSERT INTO orb_outcomes VALUES
            ('MGC', 'CME_REOPEN', 'E1', ?, '2026-01-05', 0.5),
            ('MGC', 'CME_REOPEN', 'E1', ?, '2026-01-06', -0.2),
            ('MGC', 'CME_REOPEN', 'E1', ?, '2026-01-07', 0.3)
            """,
            [orb_minutes, orb_minutes, orb_minutes],
        )
        con.close()
        return db_path

    def _family(self) -> FamilyResult:
        return FamilyResult(
            family_id="CME_REOPEN_E1_NO_FILTER",
            orb_label="CME_REOPEN",
            entry_model="E1",
            filter_type="NO_FILTER",
            windows_total=1,
            windows_passed=1,
            weighted_stability=1.0,
            classification="STABLE",
            avg_expectancy_r=0.1,
            avg_sharpe=1.0,
            total_sample_size=3,
            oos_cumulative_r=0.6,
            double_break_degraded_windows=0,
        )

    def test_default_orb_minutes_5_finds_5m_rows(self, tmp_path):
        db_path = self._seed_db(tmp_path, orb_minutes=5)
        results = compute_day_of_week_stats(db_path, [self._family()], instrument="MGC")
        assert results[0].day_of_week_stats is not None, (
            "Default orb_minutes=5 must find the 5m rows"
        )

    def test_explicit_orb_minutes_15_finds_15m_rows(self, tmp_path):
        db_path = self._seed_db(tmp_path, orb_minutes=15)
        results = compute_day_of_week_stats(
            db_path, [self._family()], instrument="MGC", orb_minutes=15
        )
        assert results[0].day_of_week_stats is not None, (
            "Explicit orb_minutes=15 must reach the daily_features query"
        )

    def test_mismatched_orb_minutes_returns_empty(self, tmp_path):
        """Seed 15m data, query with default 5m → no eligible days → no stats."""
        db_path = self._seed_db(tmp_path, orb_minutes=15)
        results = compute_day_of_week_stats(db_path, [self._family()], instrument="MGC")
        assert results[0].day_of_week_stats is None, (
            "Default orb_minutes=5 must NOT match 15m rows"
        )

    def test_outcomes_query_filters_aperture_no_contamination(self, tmp_path):
        """PIPELINE_AUDIT_2026-02-27 F1 sibling bug: orb_outcomes query must
        filter by orb_minutes, otherwise 5m/15m/30m outcomes for the same
        trading_day get mixed into a single avg_pnl_r before DOW stats run.
        """
        from pipeline.init_db import ORB_LABELS

        db_path = tmp_path / "contam.db"
        con = duckdb.connect(str(db_path))
        size_col_defs = ", ".join(f"orb_{lbl}_size DOUBLE" for lbl in ORB_LABELS)
        size_cols = ", ".join(f"orb_{lbl}_size" for lbl in ORB_LABELS)
        con.execute(
            f"""
            CREATE TABLE daily_features (
                symbol TEXT, trading_day DATE, orb_minutes INTEGER,
                {size_col_defs}
            )
            """
        )
        size_placeholders = ", ".join("?" for _ in ORB_LABELS)
        params = ["MGC", "2026-01-05", 5] + [5.0] * len(ORB_LABELS)
        con.execute(
            f"""
            INSERT INTO daily_features (symbol, trading_day, orb_minutes, {size_cols})
            VALUES (?, ?, ?, {size_placeholders})
            """,
            params,
        )
        con.execute(
            """
            CREATE TABLE orb_outcomes (
                symbol TEXT, orb_label TEXT, entry_model TEXT,
                orb_minutes INTEGER, trading_day DATE, pnl_r DOUBLE
            )
            """
        )
        # Same trading_day, different apertures, WILDLY different pnl_r.
        # If the query doesn't filter by orb_minutes, AVG mixes them.
        con.execute(
            """
            INSERT INTO orb_outcomes VALUES
            ('MGC', 'CME_REOPEN', 'E1', 5,  '2026-01-05', 1.00),
            ('MGC', 'CME_REOPEN', 'E1', 15, '2026-01-05', -5.00),
            ('MGC', 'CME_REOPEN', 'E1', 30, '2026-01-05', -5.00)
            """
        )
        con.close()

        results = compute_day_of_week_stats(db_path, [self._family()], instrument="MGC")
        stats = results[0].day_of_week_stats
        assert stats is not None
        # DuckDB DAYOFWEEK: 2026-01-05 is Monday → 1
        mon = stats.get("Mon")
        assert mon is not None, f"expected Mon stat, got {stats}"
        # With the fix, only the 5m row contributes → exp_r == 1.00.
        # Without the fix, AVG(1.00, -5.00, -5.00) == -3.0.
        assert mon["exp_r"] == 1.0, (
            f"outcomes query is mixing apertures: Mon exp_r={mon['exp_r']}, expected 1.0"
        )


# =========================================================================
# Family ID tests
# =========================================================================


class TestFamilyId:
    def test_basic_format(self):
        assert make_family_id("CME_REOPEN", "E1", "ORB_G4") == "CME_REOPEN_E1_ORB_G4"

    def test_no_filter(self):
        assert make_family_id("LONDON_METALS", "E3", "NO_FILTER") == "LONDON_METALS_E3_NO_FILTER"


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_single_window_all_pass(self):
        """Single window with high sample = STABLE."""
        labels = ["rolling_12m_2025_01"]
        validated = [
            {
                "run_label": "rolling_12m_2025_01",
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4",
                "orb_label": "CME_REOPEN",
                "entry_model": "E1",
                "filter_type": "ORB_G4",
                "rr_target": 2.0,
                "confirm_bars": 2,
                "sample_size": 80,
                "win_rate": 0.50,
                "expectancy_r": 0.20,
                "sharpe_ratio": 0.15,
                "max_drawdown_r": 4.0,
            }
        ]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 1
        assert results[0].classification == "STABLE"

    def test_all_windows_fail(self):
        """No validated strategies in any window = empty."""
        labels = ["rolling_12m_2025_01", "rolling_12m_2025_02"]
        results = aggregate_rolling_performance([], labels, {})
        assert results == []
