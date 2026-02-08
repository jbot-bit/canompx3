"""
Tests for trading_app.portfolio module.
"""

import sys
import json
from pathlib import Path

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from trading_app.portfolio import (
    PortfolioStrategy,
    Portfolio,
    compute_position_size,
    compute_position_size_prop,
    load_validated_strategies,
    diversify_strategies,
    build_portfolio,
    estimate_daily_capital,
    build_strategy_daily_series,
    correlation_matrix,
    MIN_OVERLAP_DAYS,
)
from pipeline.cost_model import get_cost_spec


def _cost():
    return get_cost_spec("MGC")


def _make_strategy(**overrides):
    """Build a strategy dict with sane defaults."""
    base = {
        "strategy_id": "MGC_2300_E1_RR2.0_CB5_NO_FILTER",
        "instrument": "MGC",
        "orb_label": "2300",
        "entry_model": "E1",
        "rr_target": 2.0,
        "confirm_bars": 5,
        "filter_type": "NO_FILTER",
        "expectancy_r": 0.30,
        "win_rate": 0.55,
        "sample_size": 300,
        "sharpe_ratio": 0.4,
        "max_drawdown_r": 5.0,
        "median_risk_points": 10.0,
    }
    base.update(overrides)
    return base


def _setup_db(tmp_path, strategies):
    """Create temp DB with schema + validated_setups + experimental_strategies rows."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    from trading_app.db_manager import init_trading_app_schema
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    for s in strategies:
        # Insert into experimental_strategies first (for median_risk_points JOIN)
        con.execute("""
            INSERT INTO experimental_strategies
            (strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type, filter_params,
             sample_size, win_rate, avg_win_r, avg_loss_r, expectancy_r,
             sharpe_ratio, max_drawdown_r, median_risk_points, avg_risk_points,
             yearly_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            s["strategy_id"], s["instrument"], s["orb_label"], 5,
            s["rr_target"], s["confirm_bars"], s["entry_model"], s["filter_type"], "{}",
            s["sample_size"], s["win_rate"], 1.8, 1.0, s["expectancy_r"],
            s.get("sharpe_ratio"), s.get("max_drawdown_r"),
            s.get("median_risk_points"), s.get("median_risk_points"), "{}",
        ])

        # Insert into validated_setups
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, promoted_from, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type, filter_params,
             sample_size, win_rate, expectancy_r, years_tested, all_years_positive,
             stress_test_passed, sharpe_ratio, max_drawdown_r, yearly_results,
             status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            s["strategy_id"], s["strategy_id"], s["instrument"], s["orb_label"], 5,
            s["rr_target"], s["confirm_bars"], s["entry_model"], s["filter_type"], "{}",
            s["sample_size"], s["win_rate"], s["expectancy_r"], 4, True,
            True, s.get("sharpe_ratio"), s.get("max_drawdown_r"), "{}",
            "ACTIVE",
        ])
    con.commit()
    con.close()
    return db_path


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizing:

    def test_basic_sizing(self):
        """$25K equity, 2% risk, 10pt ORB, $10/point = 5 contracts."""
        contracts = compute_position_size(25000.0, 2.0, 10.0, _cost())
        assert contracts == 5

    def test_small_risk(self):
        """Small ORB = more contracts."""
        contracts = compute_position_size(25000.0, 2.0, 3.0, _cost())
        assert contracts == 16

    def test_large_risk(self):
        """Large ORB = fewer contracts."""
        contracts = compute_position_size(25000.0, 2.0, 50.0, _cost())
        assert contracts == 1

    def test_zero_risk(self):
        """Zero risk points = 0 contracts."""
        contracts = compute_position_size(25000.0, 2.0, 0.0, _cost())
        assert contracts == 0

    def test_risk_too_large_returns_zero(self):
        """If risk per contract exceeds budget, return 0."""
        # $1000 account, 1% risk = $10 available. 100pt ORB = $1000 risk. Can't trade.
        contracts = compute_position_size(1000.0, 1.0, 100.0, _cost())
        assert contracts == 0

    def test_prop_firm_sizing(self):
        """Prop firm: risk based on max drawdown, not equity."""
        # $5K drawdown, 2% risk, 10pt ORB
        contracts = compute_position_size_prop(5000.0, 2.0, 10.0, _cost())
        assert contracts == 1

    def test_prop_firm_zero_drawdown(self):
        contracts = compute_position_size_prop(0.0, 2.0, 10.0, _cost())
        assert contracts == 0


# ============================================================================
# Diversification Tests
# ============================================================================

class TestDiversification:

    def test_max_strategies_enforced(self):
        # Use different orb_labels to avoid per-orb limit
        orbs = ["0900", "1000", "1100", "1800", "2300", "0030"]
        candidates = [
            _make_strategy(strategy_id=f"s{i}", orb_label=orbs[i % len(orbs)])
            for i in range(50)
        ]
        selected = diversify_strategies(candidates, max_strategies=10)
        assert len(selected) == 10

    def test_max_per_orb_enforced(self):
        candidates = [
            _make_strategy(strategy_id=f"s{i}", orb_label="2300")
            for i in range(20)
        ]
        selected = diversify_strategies(candidates, max_strategies=20, max_per_orb=3)
        assert len(selected) == 3

    def test_diversifies_across_orbs(self):
        candidates = [
            _make_strategy(strategy_id=f"s_2300_{i}", orb_label="2300", expectancy_r=0.30 - i*0.01)
            for i in range(5)
        ] + [
            _make_strategy(strategy_id=f"s_1800_{i}", orb_label="1800", expectancy_r=0.25 - i*0.01)
            for i in range(5)
        ]
        selected = diversify_strategies(candidates, max_strategies=6, max_per_orb=3)
        orbs = [s["orb_label"] for s in selected]
        assert "2300" in orbs
        assert "1800" in orbs

    def test_empty_candidates(self):
        selected = diversify_strategies([], max_strategies=10)
        assert selected == []

    def test_max_per_entry_model(self):
        candidates = [
            _make_strategy(strategy_id=f"s_e1_{i}", entry_model="E1", expectancy_r=0.30-i*0.01)
            for i in range(10)
        ]
        selected = diversify_strategies(candidates, max_strategies=10, max_per_entry_model=3)
        assert len(selected) == 3


# ============================================================================
# Portfolio Construction Tests
# ============================================================================

class TestBuildPortfolio:

    def test_builds_from_db(self, tmp_path):
        strategies = [
            _make_strategy(strategy_id=f"MGC_2300_E1_RR2.0_CB5_NO_FILTER"),
            _make_strategy(strategy_id=f"MGC_1800_E2_RR1.5_CB4_ORB_G4",
                           orb_label="1800", entry_model="E2", expectancy_r=0.25),
        ]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")
        assert len(portfolio.strategies) == 2

    def test_filters_low_expr(self, tmp_path):
        strategies = [
            _make_strategy(strategy_id="s1", expectancy_r=0.30),
            _make_strategy(strategy_id="s2", expectancy_r=0.05),
        ]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC", min_expectancy_r=0.10)
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].strategy_id == "s1"

    def test_empty_db(self, tmp_path):
        db_path = _setup_db(tmp_path, [])
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")
        assert len(portfolio.strategies) == 0

    def test_portfolio_parameters_stored(self, tmp_path):
        db_path = _setup_db(tmp_path, [_make_strategy()])
        portfolio = build_portfolio(
            db_path=db_path, instrument="MGC",
            account_equity=50000.0, risk_per_trade_pct=1.5,
            max_concurrent_positions=5, max_daily_loss_r=3.0,
        )
        assert portfolio.account_equity == 50000.0
        assert portfolio.risk_per_trade_pct == 1.5
        assert portfolio.max_concurrent_positions == 5
        assert portfolio.max_daily_loss_r == 3.0


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:

    def test_json_roundtrip(self, tmp_path):
        strategies = [_make_strategy()]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")

        json_str = portfolio.to_json()
        loaded = Portfolio.from_json(json_str)

        assert loaded.name == portfolio.name
        assert loaded.instrument == portfolio.instrument
        assert loaded.account_equity == portfolio.account_equity
        assert len(loaded.strategies) == len(portfolio.strategies)
        assert loaded.strategies[0].strategy_id == portfolio.strategies[0].strategy_id

    def test_summary_correct(self, tmp_path):
        strategies = [
            _make_strategy(strategy_id="s1", orb_label="2300", entry_model="E1"),
            _make_strategy(strategy_id="s2", orb_label="1800", entry_model="E2",
                           expectancy_r=0.20, win_rate=0.50),
        ]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")
        summary = portfolio.summary()

        assert summary["strategy_count"] == 2
        assert summary["orb_distribution"]["2300"] == 1
        assert summary["orb_distribution"]["1800"] == 1
        assert summary["entry_model_distribution"]["E1"] == 1
        assert summary["entry_model_distribution"]["E2"] == 1
        assert summary["avg_expectancy_r"] == pytest.approx(0.25, abs=0.01)

    def test_empty_summary(self):
        portfolio = Portfolio(
            name="empty", instrument="MGC", strategies=[],
            account_equity=25000, risk_per_trade_pct=2.0,
            max_concurrent_positions=3, max_daily_loss_r=5.0,
        )
        summary = portfolio.summary()
        assert summary["strategy_count"] == 0


# ============================================================================
# Capital Estimation Tests
# ============================================================================

class TestCapitalEstimation:

    def test_estimates_with_strategies(self, tmp_path):
        strategies = [_make_strategy()]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")
        capital = estimate_daily_capital(portfolio, _cost())

        assert capital["estimated_daily_trades"] > 0
        assert capital["avg_risk_points"] > 0
        assert capital["risk_per_trade_dollars"] > 0
        assert capital["max_concurrent_risk_dollars"] > 0

    def test_estimates_empty_portfolio(self):
        portfolio = Portfolio(
            name="empty", instrument="MGC", strategies=[],
            account_equity=25000, risk_per_trade_pct=2.0,
            max_concurrent_positions=3, max_daily_loss_r=5.0,
        )
        capital = estimate_daily_capital(portfolio, _cost())
        assert capital["estimated_daily_trades"] == 0


# ============================================================================
# CLI Tests
# ============================================================================

# ============================================================================
# Correlation / Shared Calendar Tests
# ============================================================================

def _setup_db_with_outcomes(tmp_path, strategies, daily_features_rows, outcome_rows):
    """Create a test DB with daily_features, orb_outcomes, and validated_setups.

    daily_features_rows: list of dicts with at minimum
        {trading_day, symbol, orb_minutes, orb_0900_size, orb_2300_size, ...}
    outcome_rows: list of dicts with
        {trading_day, symbol, orb_label, orb_minutes, rr_target, confirm_bars,
         entry_model, pnl_r, outcome}
    """
    db_path = tmp_path / "corr_test.db"
    con = duckdb.connect(str(db_path))
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    from trading_app.db_manager import init_trading_app_schema
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))

    # Insert daily_features
    for df_row in daily_features_rows:
        td = df_row["trading_day"]
        sym = df_row.get("symbol", "MGC")
        om = df_row.get("orb_minutes", 5)
        s0900 = df_row.get("orb_0900_size")
        s2300 = df_row.get("orb_2300_size")
        con.execute("""
            INSERT INTO daily_features (trading_day, symbol, orb_minutes,
                bar_count_1m, orb_0900_size, orb_2300_size)
            VALUES (?, ?, ?, 1440, ?, ?)
        """, [td, sym, om, s0900, s2300])

    # Insert strategies
    for s in strategies:
        con.execute("""
            INSERT INTO experimental_strategies
            (strategy_id, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type, filter_params,
             sample_size, win_rate, avg_win_r, avg_loss_r, expectancy_r,
             sharpe_ratio, max_drawdown_r, median_risk_points, avg_risk_points,
             yearly_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            s["strategy_id"], s["instrument"], s["orb_label"], 5,
            s["rr_target"], s["confirm_bars"], s["entry_model"], s["filter_type"], "{}",
            s["sample_size"], s["win_rate"], 1.8, 1.0, s["expectancy_r"],
            s.get("sharpe_ratio"), s.get("max_drawdown_r"),
            s.get("median_risk_points"), s.get("median_risk_points"), "{}",
        ])

        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, promoted_from, instrument, orb_label, orb_minutes,
             rr_target, confirm_bars, entry_model, filter_type, filter_params,
             sample_size, win_rate, expectancy_r, years_tested, all_years_positive,
             stress_test_passed, sharpe_ratio, max_drawdown_r, yearly_results,
             status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            s["strategy_id"], s["strategy_id"], s["instrument"], s["orb_label"], 5,
            s["rr_target"], s["confirm_bars"], s["entry_model"], s["filter_type"], "{}",
            s["sample_size"], s["win_rate"], s["expectancy_r"], 4, True,
            True, s.get("sharpe_ratio"), s.get("max_drawdown_r"), "{}",
            "ACTIVE",
        ])

    # Insert orb_outcomes
    for oc in outcome_rows:
        con.execute("""
            INSERT INTO orb_outcomes
            (trading_day, symbol, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, outcome, pnl_r)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            oc["trading_day"], oc.get("symbol", "MGC"), oc["orb_label"],
            oc.get("orb_minutes", 5), oc["rr_target"],
            oc["confirm_bars"], oc["entry_model"],
            oc.get("outcome", "win"), oc["pnl_r"],
        ])

    con.commit()
    con.close()
    return db_path


import datetime

def _make_daily_features_rows(n_days=300, start="2023-01-02"):
    """Generate n_days of daily_features rows with ORB sizes."""
    rows = []
    d = datetime.date.fromisoformat(start)
    for i in range(n_days):
        rows.append({
            "trading_day": d,
            "symbol": "MGC",
            "orb_minutes": 5,
            # Vary sizes: 0900 always has data; 2300 has size that varies
            "orb_0900_size": 5.0 + (i % 10) * 0.5,  # 5.0 - 9.5
            "orb_2300_size": 3.0 + (i % 8) * 0.5,    # 3.0 - 6.5
        })
        # Skip weekends (rough approximation)
        d += datetime.timedelta(days=1)
        if d.weekday() >= 5:
            d += datetime.timedelta(days=7 - d.weekday())
    return rows


class TestBuildStrategyDailySeries:

    def test_shared_calendar_index(self, tmp_path):
        """All strategies share the same daily index from daily_features."""
        df_rows = _make_daily_features_rows(250)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="2300", filter_type="NO_FILTER",
        )
        # Trades for A on first 100 days, B on last 50
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.5}
            for i in range(100)
        ]
        outcomes_b = [
            {"trading_day": df_rows[200 + i]["trading_day"], "orb_label": "2300",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": -1.0}
            for i in range(50)
        ]
        db_path = _setup_db_with_outcomes(
            tmp_path, [strat_a, strat_b], df_rows, outcomes_a + outcomes_b,
        )
        series_df, stats = build_strategy_daily_series(db_path, [
            strat_a["strategy_id"], strat_b["strategy_id"],
        ])
        # Both strategies must have same index
        assert len(series_df) == 250
        assert list(series_df.columns) == [
            strat_a["strategy_id"], strat_b["strategy_id"],
        ]

    def test_eligible_no_trade_is_zero(self, tmp_path):
        """Eligible day with no trade should be 0.0, not NaN."""
        df_rows = _make_daily_features_rows(50)
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        # Only trade on day 0
        outcomes = [
            {"trading_day": df_rows[0]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 1.5}
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)
        series_df, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        col = series_df[strat["strategy_id"]]

        # Day 0 should have trade pnl_r
        assert col.iloc[0] == pytest.approx(1.5)
        # Days 1-49: eligible (NO_FILTER), no trade -> 0.0
        for i in range(1, 50):
            assert col.iloc[i] == pytest.approx(0.0), f"Day {i} should be 0.0"

    def test_ineligible_day_is_nan(self, tmp_path):
        """Ineligible day (filter fails) should be NaN."""
        df_rows = _make_daily_features_rows(50)
        # G8 filter: only days with orb_0900_size >= 8.0
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="0900", filter_type="ORB_G8",
        )
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, [])
        series_df, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        col = series_df[strat["strategy_id"]]

        # Check that ineligible days are NaN and eligible days are 0.0
        for i in range(min(50, len(col))):
            size = df_rows[i]["orb_0900_size"]
            if size >= 8.0:
                assert col.iloc[i] == pytest.approx(0.0), f"Day {i} size={size} should be 0.0"
            else:
                assert np.isnan(col.iloc[i]), f"Day {i} size={size} should be NaN"

    def test_stats_counts(self, tmp_path):
        """Stats dict should report eligible, traded, and padded days correctly."""
        df_rows = _make_daily_features_rows(100)
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        outcomes = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.3}
            for i in range(30)
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)
        _, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        s = stats[strat["strategy_id"]]
        assert s["eligible_days"] == 100
        assert s["traded_days"] == 30
        assert s["padded_zero_days"] == 70

    def test_ineligible_outcomes_not_overlaid(self, tmp_path):
        """REGRESSION: outcomes on filter-ineligible days must NOT be overlaid.

        This is the core bug from fix4.txt. orb_outcomes has break-days
        regardless of filter, so if a G8 strategy has outcomes on days
        where orb_size < 8, those must remain NaN (not get pnl_r).
        """
        df_rows = _make_daily_features_rows(50)
        # G8 filter: only days with orb_0900_size >= 8.0
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="0900", filter_type="ORB_G8",
        )
        # Create outcomes on ALL days — including ineligible ones
        outcomes = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.5}
            for i in range(50)
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)
        series_df, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        col = series_df[strat["strategy_id"]]
        s = stats[strat["strategy_id"]]

        # Verify ineligible days remain NaN (not overwritten by pnl_r)
        for i in range(min(50, len(col))):
            size = df_rows[i]["orb_0900_size"]
            if size < 8.0:
                assert np.isnan(col.iloc[i]), (
                    f"Day {i} (size={size}) is ineligible for G8 but got {col.iloc[i]}; "
                    f"should be NaN"
                )

        # Verify skipped count is non-zero
        assert s["overlays_skipped_ineligible"] > 0, (
            "Expected some overlays skipped due to ineligibility"
        )
        # traded_days must be much less than total outcomes
        assert s["traded_days"] < 50, "Not all outcomes should be overlaid for G8 filter"

    def test_empty_strategies(self, tmp_path):
        """Empty strategy list returns empty DataFrame."""
        df_rows = _make_daily_features_rows(10)
        db_path = _setup_db_with_outcomes(tmp_path, [], df_rows, [])
        series_df, stats = build_strategy_daily_series(db_path, [])
        assert series_df.empty
        assert stats == {}


class TestCorrelationMatrix:

    def test_diagonal_is_one(self, tmp_path):
        """Diagonal of correlation matrix should be 1.0."""
        df_rows = _make_daily_features_rows(250)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="2300", filter_type="NO_FILTER",
        )
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": 0.5 if i % 2 == 0 else -1.0}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "2300",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": -1.0 if i % 2 == 0 else 0.5}
            for i in range(200)
        ]
        db_path = _setup_db_with_outcomes(
            tmp_path, [strat_a, strat_b], df_rows, outcomes_a + outcomes_b,
        )
        ids = [strat_a["strategy_id"], strat_b["strategy_id"]]
        corr = correlation_matrix(db_path, ids, min_overlap_days=50)
        assert corr.loc[ids[0], ids[0]] == pytest.approx(1.0)
        assert corr.loc[ids[1], ids[1]] == pytest.approx(1.0)

    def test_overlap_guard_nan(self, tmp_path):
        """Pairs with insufficient overlap should get NaN correlation."""
        df_rows = _make_daily_features_rows(300)
        # A trades on days with orb_0900_size >= 8 (sparse ~ 40% of days)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="0900", filter_type="ORB_G8",
        )
        # B trades on days with orb_2300_size >= 8 (also sparse)
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_ORB_G8",
            orb_label="2300", filter_type="ORB_G8",
        )
        db_path = _setup_db_with_outcomes(
            tmp_path, [strat_a, strat_b], df_rows, [],
        )
        ids = [strat_a["strategy_id"], strat_b["strategy_id"]]
        # Set high overlap threshold so it triggers NaN
        corr = correlation_matrix(db_path, ids, min_overlap_days=999)
        assert np.isnan(corr.loc[ids[0], ids[1]])
        assert np.isnan(corr.loc[ids[1], ids[0]])

    def test_sufficient_overlap_computes(self, tmp_path):
        """With sufficient overlap, correlation is computed (not NaN)."""
        df_rows = _make_daily_features_rows(250)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="2300", filter_type="NO_FILTER",
        )
        # Both trade every day -> both NO_FILTER -> all 250 days eligible -> 250 overlap
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.3}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "2300",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": -0.5}
            for i in range(200)
        ]
        db_path = _setup_db_with_outcomes(
            tmp_path, [strat_a, strat_b], df_rows, outcomes_a + outcomes_b,
        )
        ids = [strat_a["strategy_id"], strat_b["strategy_id"]]
        corr = correlation_matrix(db_path, ids, min_overlap_days=50)
        # Should have a real correlation value, not NaN
        val = corr.loc[ids[0], ids[1]]
        assert not np.isnan(val)

    def test_anticorrelated_strategies(self, tmp_path):
        """Opposite-return strategies should have negative correlation."""
        df_rows = _make_daily_features_rows(250)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="0900", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="2300", filter_type="NO_FILTER",
        )
        # A wins when i is even, B wins when i is odd (anticorrelated)
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "0900",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": 1.5 if i % 2 == 0 else -1.0}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "2300",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": -1.0 if i % 2 == 0 else 1.5}
            for i in range(200)
        ]
        db_path = _setup_db_with_outcomes(
            tmp_path, [strat_a, strat_b], df_rows, outcomes_a + outcomes_b,
        )
        ids = [strat_a["strategy_id"], strat_b["strategy_id"]]
        corr = correlation_matrix(db_path, ids, min_overlap_days=50)
        val = corr.loc[ids[0], ids[1]]
        assert val < -0.5, f"Expected negative correlation, got {val}"

    def test_empty_returns_empty(self, tmp_path):
        """Empty strategy list -> empty DataFrame."""
        df_rows = _make_daily_features_rows(10)
        db_path = _setup_db_with_outcomes(tmp_path, [], df_rows, [])
        corr = correlation_matrix(db_path, [])
        assert corr.empty


class TestCLI:
    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/portfolio.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout
