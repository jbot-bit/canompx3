"""
Tests for rolling portfolio: window generation, stability scoring,
double-break detection, day-of-week concentration, and portfolio integration.

Uses in-memory DuckDB with synthetic data.
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.infra.rolling_eval import generate_rolling_windows, DOUBLE_BREAK_THRESHOLD
from trading_app.rolling_portfolio import (
    make_family_id,
    _window_weight,
    classify_stability,
    aggregate_rolling_performance,
    FamilyResult,
    STABLE_THRESHOLD,
    TRANSITIONING_THRESHOLD,
    FULL_WEIGHT_SAMPLE,
    load_rolling_results,
    load_all_rolling_run_labels,
    load_rolling_degraded_counts,
    load_rolling_validated_strategies,
)
from pipeline.build_daily_features import detect_double_break


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
                f"Lookahead violation: train_end={w['train_end']} >= "
                f"test_start={w['test_start']}"
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

    def _make_validated(self, run_label, orb_label="0900", em="E1",
                         ft="ORB_G4", rr=2.0, cb=2, sample=50,
                         exp_r=0.15, sharpe=0.1):
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
        validated = [self._make_validated(l, sample=60) for l in labels]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 1
        assert results[0].classification == "STABLE"
        assert results[0].weighted_stability >= STABLE_THRESHOLD

    def test_low_sample_windows_penalized(self):
        """Windows with 20 trades get 0.4 weight, not 1.0."""
        labels = [f"rolling_12m_2025_{m:02d}" for m in range(1, 7)]
        # Pass 3 of 6 windows, but all with only 20 trades
        validated = [self._make_validated(l, sample=20) for l in labels[:3]]

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
            self._make_validated("rolling_12m_2025_01", orb_label="0900"),
            self._make_validated("rolling_12m_2025_02", orb_label="0900"),
            self._make_validated("rolling_12m_2025_01", orb_label="1800"),
        ]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 2
        fam_ids = {r.family_id for r in results}
        assert "0900_E1_ORB_G4" in fam_ids
        assert "1800_E1_ORB_G4" in fam_ids

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

        bars = pd.DataFrame({
            "ts_utc": pd.to_datetime([
                "2025-01-06 23:05:00",  # After 0900 ORB close
                "2025-01-06 23:10:00",
                "2025-01-06 23:15:00",
            ], utc=True),
            "open": [1995.0, 2001.0, 1989.0],
            "high": [2001.0, 2003.0, 1991.0],
            "low": [1993.0, 1999.0, 1988.0],
            "close": [2001.0, 2001.0, 1989.0],
            "volume": [100, 100, 100],
        })

        result = detect_double_break(
            bars, date(2025, 1, 7), "0900", 5, orb_high, orb_low
        )
        assert result is True

    def test_only_one_boundary(self):
        """No double break when only one side is breached."""
        orb_high = 2000.0
        orb_low = 1990.0

        bars = pd.DataFrame({
            "ts_utc": pd.to_datetime([
                "2025-01-06 23:05:00",
                "2025-01-06 23:10:00",
            ], utc=True),
            "open": [1995.0, 1996.0],
            "high": [2001.0, 2002.0],  # Breaches high
            "low": [1993.0, 1994.0],   # Never breaches low (1990)
            "close": [1999.0, 2001.0],
            "volume": [100, 100],
        })

        result = detect_double_break(
            bars, date(2025, 1, 7), "0900", 5, orb_high, orb_low
        )
        assert result is False

    def test_missing_orb_data(self):
        bars = pd.DataFrame(columns=["ts_utc", "open", "high", "low", "close", "volume"])
        result = detect_double_break(bars, date(2025, 1, 7), "0900", 5, None, None)
        assert result is None

    def test_no_bars_in_window(self):
        """Empty window returns None."""
        bars = pd.DataFrame({
            "ts_utc": pd.to_datetime(["2025-01-06 10:00:00"], utc=True),
            "open": [1995.0],
            "high": [2001.0],
            "low": [1993.0],
            "close": [1999.0],
            "volume": [100],
        })
        result = detect_double_break(
            bars, date(2025, 1, 7), "0900", 5, 2000.0, 1990.0
        )
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
        con.execute("""
            INSERT INTO regime_strategies
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type,
             sample_size, win_rate, expectancy_r, sharpe_ratio, max_drawdown_r,
             median_risk_points, avg_risk_points,
             validation_status, validation_notes, yearly_results,
             start_date, end_date)
            VALUES (?, 'MGC_0900_E1_RR2.0_CB2_ORB_G4', 'MGC', '0900', 5,
                    2.0, 2, 'E1', 'ORB_G4',
                    55, 0.45, 0.15, 0.12, 5.0, 3.5, 3.5,
                    'PASSED', '', '{}',
                    '2024-01-01', '2024-12-31')
        """, [label])

    # Seed regime_validated
    for label in labels:
        con.execute("""
            INSERT INTO regime_validated
            (run_label, strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type,
             sample_size, win_rate, expectancy_r,
             years_tested, all_years_positive, stress_test_passed,
             sharpe_ratio, max_drawdown_r, yearly_results, status,
             start_date, end_date)
            VALUES (?, 'MGC_0900_E1_RR2.0_CB2_ORB_G4', 'MGC', '0900', 5,
                    2.0, 2, 'E1', 'ORB_G4',
                    55, 0.45, 0.15,
                    1, TRUE, TRUE,
                    0.12, 5.0, '{}', 'active',
                    '2024-01-01', '2024-12-31')
        """, [label])

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
        assert all(l.startswith("rolling_12m_") for l in labels)

    def test_load_rolling_validated_for_portfolio(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        strategies = load_rolling_validated_strategies(
            db_path, "MGC", train_months=12,
            min_weighted_score=0.0,  # Accept anything
            min_expectancy_r=0.0,
        )
        assert len(strategies) >= 1
        assert strategies[0]["source"] == "rolling"

    def test_rolling_source_tag(self, tmp_path):
        db_path = _setup_rolling_db(tmp_path)
        strategies = load_rolling_validated_strategies(
            db_path, "MGC", train_months=12,
            min_weighted_score=0.0,
            min_expectancy_r=0.0,
        )
        for s in strategies:
            assert s["source"] == "rolling"

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
            VALUES ('rolling_12m_2025_01', 'MGC_1800_E3_RR2.0_CB4_ORB_G6',
                    'MGC', '1800', 5, 2.0, 4, 'E3', 'ORB_G6',
                    30, 0.40, 0.10, 0.05, 8.0, 4.0, 4.0,
                    'REJECTED', 'Auto-degraded: double-break >67% in training window',
                    '{}', '2024-01-01', '2024-12-31')
        """)
        con.commit()
        con.close()

        counts = load_rolling_degraded_counts(db_path, train_months=12)
        assert "1800_E3_ORB_G6" in counts
        assert counts["1800_E3_ORB_G6"] == 1


# =========================================================================
# Family ID tests
# =========================================================================


class TestFamilyId:

    def test_basic_format(self):
        assert make_family_id("0900", "E1", "ORB_G4") == "0900_E1_ORB_G4"

    def test_no_filter(self):
        assert make_family_id("1800", "E3", "NO_FILTER") == "1800_E3_NO_FILTER"


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:

    def test_single_window_all_pass(self):
        """Single window with high sample = STABLE."""
        labels = ["rolling_12m_2025_01"]
        validated = [{
            "run_label": "rolling_12m_2025_01",
            "strategy_id": "MGC_0900_E1_RR2.0_CB2_ORB_G4",
            "orb_label": "0900",
            "entry_model": "E1",
            "filter_type": "ORB_G4",
            "rr_target": 2.0,
            "confirm_bars": 2,
            "sample_size": 80,
            "win_rate": 0.50,
            "expectancy_r": 0.20,
            "sharpe_ratio": 0.15,
            "max_drawdown_r": 4.0,
        }]

        results = aggregate_rolling_performance(validated, labels, {})
        assert len(results) == 1
        assert results[0].classification == "STABLE"

    def test_all_windows_fail(self):
        """No validated strategies in any window = empty."""
        labels = ["rolling_12m_2025_01", "rolling_12m_2025_02"]
        results = aggregate_rolling_performance([], labels, {})
        assert results == []
