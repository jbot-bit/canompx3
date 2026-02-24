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
    compute_vol_scalar,
    compute_position_size_vol_scaled,
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
        "orb_label": "US_DATA_830",
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


class TestVolScalar:

    def test_median_atr_equals_current(self):
        """When ATR == median, scalar = 1.0."""
        assert compute_vol_scalar(25.0, 25.0) == pytest.approx(1.0)

    def test_high_atr_reduces_size(self):
        """ATR double median -> scalar = 0.5 (half size)."""
        assert compute_vol_scalar(50.0, 25.0) == pytest.approx(0.5)

    def test_low_atr_increases_size(self):
        """ATR half median -> scalar = 1.5 (capped at max)."""
        assert compute_vol_scalar(12.5, 25.0) == pytest.approx(1.5)

    def test_clamped_at_max(self):
        """Very low ATR clamped at max_scalar."""
        assert compute_vol_scalar(1.0, 100.0, max_scalar=2.0) == pytest.approx(2.0)

    def test_clamped_at_min(self):
        """Very high ATR clamped at min_scalar."""
        assert compute_vol_scalar(1000.0, 25.0, min_scalar=0.25) == pytest.approx(0.25)

    def test_zero_atr_returns_one(self):
        assert compute_vol_scalar(0.0, 25.0) == 1.0

    def test_zero_median_returns_one(self):
        assert compute_vol_scalar(25.0, 0.0) == 1.0


class TestVolScaledPositionSizing:

    def test_scalar_one_matches_base(self):
        """vol_scalar=1.0 should give same result as compute_position_size."""
        base = compute_position_size(25000.0, 2.0, 10.0, _cost())
        scaled = compute_position_size_vol_scaled(25000.0, 2.0, 10.0, _cost(), 1.0)
        assert scaled == base

    def test_high_vol_fewer_contracts(self):
        """High ATR -> scalar 0.5 -> fewer contracts."""
        base = compute_position_size(25000.0, 2.0, 10.0, _cost())
        scaled = compute_position_size_vol_scaled(25000.0, 2.0, 10.0, _cost(), 0.5)
        assert scaled < base

    def test_low_vol_more_contracts(self):
        """Low ATR -> scalar 1.5 -> more contracts."""
        base = compute_position_size(25000.0, 2.0, 10.0, _cost())
        scaled = compute_position_size_vol_scaled(25000.0, 2.0, 10.0, _cost(), 1.5)
        assert scaled > base

    def test_zero_scalar(self):
        assert compute_position_size_vol_scaled(25000.0, 2.0, 10.0, _cost(), 0.0) == 0

    def test_zero_risk(self):
        assert compute_position_size_vol_scaled(25000.0, 2.0, 0.0, _cost(), 1.0) == 0


# ============================================================================
# Diversification Tests
# ============================================================================

class TestDiversification:

    def test_max_strategies_enforced(self):
        # Use different orb_labels to avoid per-orb limit
        orbs = ["CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS", "US_DATA_830", "NYSE_OPEN"]
        candidates = [
            _make_strategy(strategy_id=f"s{i}", orb_label=orbs[i % len(orbs)])
            for i in range(50)
        ]
        selected = diversify_strategies(candidates, max_strategies=10)
        assert len(selected) == 10

    def test_max_per_orb_enforced(self):
        candidates = [
            _make_strategy(strategy_id=f"s{i}", orb_label="US_DATA_830")
            for i in range(20)
        ]
        selected = diversify_strategies(candidates, max_strategies=20, max_per_orb=3)
        assert len(selected) == 3

    def test_diversifies_across_orbs(self):
        candidates = [
            _make_strategy(strategy_id=f"s_2300_{i}", orb_label="US_DATA_830", expectancy_r=0.30 - i*0.01)
            for i in range(5)
        ] + [
            _make_strategy(strategy_id=f"s_1800_{i}", orb_label="LONDON_METALS", expectancy_r=0.25 - i*0.01)
            for i in range(5)
        ]
        selected = diversify_strategies(candidates, max_strategies=6, max_per_orb=3)
        orbs = [s["orb_label"] for s in selected]
        assert "US_DATA_830" in orbs
        assert "LONDON_METALS" in orbs

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
            _make_strategy(strategy_id=f"MGC_1800_E3_RR1.5_CB4_ORB_G4",
                           orb_label="LONDON_METALS", entry_model="E3", expectancy_r=0.25),
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
            _make_strategy(strategy_id="s1", orb_label="US_DATA_830", entry_model="E1"),
            _make_strategy(strategy_id="s2", orb_label="LONDON_METALS", entry_model="E3",
                           expectancy_r=0.20, win_rate=0.50),
        ]
        db_path = _setup_db(tmp_path, strategies)
        portfolio = build_portfolio(db_path=db_path, instrument="MGC")
        summary = portfolio.summary()

        assert summary["strategy_count"] == 2
        assert summary["orb_distribution"]["US_DATA_830"] == 1
        assert summary["orb_distribution"]["LONDON_METALS"] == 1
        assert summary["entry_model_distribution"]["E1"] == 1
        assert summary["entry_model_distribution"]["E3"] == 1
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
        {trading_day, symbol, orb_minutes, orb_CME_REOPEN_size, orb_US_DATA_830_size, ...}
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
        s0900 = df_row.get("orb_CME_REOPEN_size")
        s2300 = df_row.get("orb_US_DATA_830_size")
        dow = df_row.get("day_of_week")
        is_nfp = df_row.get("is_nfp_day", False)
        is_opex = df_row.get("is_opex_day", False)
        is_fri = df_row.get("is_friday", False)
        con.execute("""
            INSERT INTO daily_features (trading_day, symbol, orb_minutes,
                bar_count_1m, orb_CME_REOPEN_size, orb_US_DATA_830_size,
                day_of_week, is_nfp_day, is_opex_day, is_friday)
            VALUES (?, ?, ?, 1440, ?, ?, ?, ?, ?, ?)
        """, [td, sym, om, s0900, s2300, dow, is_nfp, is_opex, is_fri])

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

def _make_daily_features_rows(n_days=300, start="2023-01-02", orb_minutes=5):
    """Generate n_days of daily_features rows with ORB sizes.

    orb_minutes controls ORB size multiplier:
      5 -> 1.0x, 15 -> 1.5x, 30 -> 1.8x  (wider ORBs for longer windows)
    """
    multiplier = {5: 1.0, 15: 1.5, 30: 1.8}.get(orb_minutes, 1.0)
    rows = []
    d = datetime.date.fromisoformat(start)
    for i in range(n_days):
        rows.append({
            "trading_day": d,
            "symbol": "MGC",
            "orb_minutes": orb_minutes,
            "day_of_week": d.weekday(),
            # Vary sizes: 0900 always has data; 2300 has size that varies
            "orb_CME_REOPEN_size": (5.0 + (i % 10) * 0.5) * multiplier,  # 5m: 5.0 - 9.5
            "orb_US_DATA_830_size": (3.0 + (i % 8) * 0.5) * multiplier,    # 5m: 3.0 - 6.5
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
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="US_DATA_830", filter_type="NO_FILTER",
        )
        # Trades for A on first 100 days, B on last 50
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.5}
            for i in range(100)
        ]
        outcomes_b = [
            {"trading_day": df_rows[200 + i]["trading_day"], "orb_label": "US_DATA_830",
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
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        # Only trade on day 0
        outcomes = [
            {"trading_day": df_rows[0]["trading_day"], "orb_label": "CME_REOPEN",
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
        # G8 filter: only days with orb_CME_REOPEN_size >= 8.0
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="CME_REOPEN", filter_type="ORB_G8",
        )
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, [])
        series_df, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        col = series_df[strat["strategy_id"]]

        # Check that ineligible days are NaN and eligible days are 0.0
        for i in range(min(50, len(col))):
            size = df_rows[i]["orb_CME_REOPEN_size"]
            if size >= 8.0:
                assert col.iloc[i] == pytest.approx(0.0), f"Day {i} size={size} should be 0.0"
            else:
                assert np.isnan(col.iloc[i]), f"Day {i} size={size} should be NaN"

    def test_stats_counts(self, tmp_path):
        """Stats dict should report eligible, traded, and padded days correctly."""
        df_rows = _make_daily_features_rows(100)
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        outcomes = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
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
        # G8 filter: only days with orb_CME_REOPEN_size >= 8.0
        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="CME_REOPEN", filter_type="ORB_G8",
        )
        # Create outcomes on ALL days — including ineligible ones
        outcomes = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.5}
            for i in range(50)
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)
        series_df, stats = build_strategy_daily_series(db_path, [strat["strategy_id"]])
        col = series_df[strat["strategy_id"]]
        s = stats[strat["strategy_id"]]

        # Verify ineligible days remain NaN (not overwritten by pnl_r)
        for i in range(min(50, len(col))):
            size = df_rows[i]["orb_CME_REOPEN_size"]
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


class TestCalendarOverlay:
    """Calendar overlay (NFP/OPEX skip) in R-series builder."""

    def test_nfp_day_nan_with_overlay(self, tmp_path):
        """NFP day should be NaN when calendar_overlay is provided."""
        from trading_app.config import CALENDAR_SKIP_NFP_OPEX
        df_rows = _make_daily_features_rows(10)
        # Mark day 3 as NFP
        df_rows[3]["is_nfp_day"] = True

        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        outcomes = [
            {"trading_day": df_rows[3]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 1.5}
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)

        # With overlay: NFP day should be NaN (skipped)
        series_df, stats = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        col = series_df[strat["strategy_id"]]
        assert np.isnan(col.iloc[3]), "NFP day should be NaN with calendar overlay"
        # Non-NFP eligible days should still be 0.0
        assert col.iloc[0] == pytest.approx(0.0)

    def test_nfp_day_zero_without_overlay(self, tmp_path):
        """NFP day should be 0.0 (eligible) when NO calendar_overlay is provided."""
        df_rows = _make_daily_features_rows(10)
        df_rows[3]["is_nfp_day"] = True

        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, [])

        # Without overlay: NFP day should still be eligible (0.0)
        series_df, stats = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
        )
        col = series_df[strat["strategy_id"]]
        assert col.iloc[3] == pytest.approx(0.0), "NFP day should be 0.0 without overlay"

    def test_opex_day_nan_with_overlay(self, tmp_path):
        """OPEX day should be NaN when calendar_overlay is provided."""
        from trading_app.config import CALENDAR_SKIP_NFP_OPEX
        df_rows = _make_daily_features_rows(10)
        df_rows[5]["is_opex_day"] = True

        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, [])

        series_df, stats = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        col = series_df[strat["strategy_id"]]
        assert np.isnan(col.iloc[5]), "OPEX day should be NaN with calendar overlay"

    def test_overlay_reduces_eligible_count(self, tmp_path):
        """Calendar overlay should reduce eligible_days count in stats."""
        from trading_app.config import CALENDAR_SKIP_NFP_OPEX
        df_rows = _make_daily_features_rows(50)
        # Mark 5 days as NFP
        for i in [5, 15, 25, 35, 45]:
            df_rows[i]["is_nfp_day"] = True

        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, [])

        # Without overlay
        _, stats_no = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
        )
        # With overlay
        _, stats_cal = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        assert stats_cal[strat["strategy_id"]]["eligible_days"] == (
            stats_no[strat["strategy_id"]]["eligible_days"] - 5
        )

    def test_outcome_on_nfp_day_skipped(self, tmp_path):
        """Outcome on an NFP day should be skipped (NaN) when overlay is active."""
        from trading_app.config import CALENDAR_SKIP_NFP_OPEX
        df_rows = _make_daily_features_rows(10)
        df_rows[3]["is_nfp_day"] = True

        strat = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        # Outcome exists on the NFP day
        outcomes = [
            {"trading_day": df_rows[3]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 2.0}
        ]
        db_path = _setup_db_with_outcomes(tmp_path, [strat], df_rows, outcomes)

        series_df, stats = build_strategy_daily_series(
            db_path, [strat["strategy_id"]],
            calendar_overlay=CALENDAR_SKIP_NFP_OPEX,
        )
        col = series_df[strat["strategy_id"]]
        # NFP day outcome should NOT be overlaid
        assert np.isnan(col.iloc[3]), "NFP day with outcome should still be NaN"
        assert stats[strat["strategy_id"]]["overlays_skipped_ineligible"] >= 1


class TestCorrelationMatrix:

    def test_diagonal_is_one(self, tmp_path):
        """Diagonal of correlation matrix should be 1.0."""
        df_rows = _make_daily_features_rows(250)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="US_DATA_830", filter_type="NO_FILTER",
        )
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": 0.5 if i % 2 == 0 else -1.0}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "US_DATA_830",
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
        # A trades on days with orb_CME_REOPEN_size >= 8 (sparse ~ 40% of days)
        strat_a = _make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="CME_REOPEN", filter_type="ORB_G8",
        )
        # B trades on days with orb_US_DATA_830_size >= 8 (also sparse)
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_ORB_G8",
            orb_label="US_DATA_830", filter_type="ORB_G8",
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
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="US_DATA_830", filter_type="NO_FILTER",
        )
        # Both trade every day -> both NO_FILTER -> all 250 days eligible -> 250 overlap
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.3}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "US_DATA_830",
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
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )
        strat_b = _make_strategy(
            strategy_id="MGC_2300_E1_RR2.0_CB5_NO_FILTER",
            orb_label="US_DATA_830", filter_type="NO_FILTER",
        )
        # A wins when i is even, B wins when i is odd (anticorrelated)
        outcomes_a = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "CME_REOPEN",
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1",
             "pnl_r": 1.5 if i % 2 == 0 else -1.0}
            for i in range(200)
        ]
        outcomes_b = [
            {"trading_day": df_rows[i]["trading_day"], "orb_label": "US_DATA_830",
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


class TestNestedIntegration:
    """Tests for nested ORB strategy integration (Phase 8 fixes)."""

    def _setup_nested_db(self, tmp_path, baseline_strats, nested_strats,
                         daily_features_rows, outcome_rows,
                         nested_outcome_rows=None):
        """Create a test DB with both baseline and nested tables."""
        db_path = _setup_db_with_outcomes(
            tmp_path, baseline_strats, daily_features_rows, outcome_rows,
        )
        # Init nested schema on same DB
        from trading_app.nested.schema import init_nested_schema
        con = duckdb.connect(str(db_path))
        init_nested_schema(con=con)

        for s in nested_strats:
            # Insert into nested_strategies
            con.execute("""
                INSERT INTO nested_strategies
                (strategy_id, instrument, orb_label, orb_minutes,
                 entry_resolution, rr_target, confirm_bars, entry_model,
                 filter_type, filter_params,
                 sample_size, win_rate, avg_win_r, avg_loss_r, expectancy_r,
                 sharpe_ratio, max_drawdown_r, median_risk_points, avg_risk_points,
                 yearly_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                s["strategy_id"], s["instrument"], s["orb_label"],
                s.get("orb_minutes", 15), s.get("entry_resolution", 5),
                s["rr_target"], s["confirm_bars"], s["entry_model"],
                s["filter_type"], "{}",
                s["sample_size"], s["win_rate"], 1.8, 1.0, s["expectancy_r"],
                s.get("sharpe_ratio"), s.get("max_drawdown_r"),
                s.get("median_risk_points"), s.get("median_risk_points"), "{}",
            ])

            # Insert into nested_validated
            con.execute("""
                INSERT INTO nested_validated
                (strategy_id, promoted_from, instrument, orb_label, orb_minutes,
                 entry_resolution, rr_target, confirm_bars, entry_model,
                 filter_type, filter_params,
                 sample_size, win_rate, expectancy_r, years_tested,
                 all_years_positive, stress_test_passed,
                 sharpe_ratio, max_drawdown_r, yearly_results, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                s["strategy_id"], s["strategy_id"], s["instrument"],
                s["orb_label"], s.get("orb_minutes", 15),
                s.get("entry_resolution", 5),
                s["rr_target"], s["confirm_bars"], s["entry_model"],
                s["filter_type"], "{}",
                s["sample_size"], s["win_rate"], s["expectancy_r"], 4,
                True, True, s.get("sharpe_ratio"), s.get("max_drawdown_r"),
                "{}", "active",
            ])

        # Insert nested outcomes
        if nested_outcome_rows:
            for oc in nested_outcome_rows:
                con.execute("""
                    INSERT INTO nested_outcomes
                    (trading_day, symbol, orb_label, orb_minutes,
                     entry_resolution, rr_target, confirm_bars, entry_model,
                     outcome, pnl_r)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    oc["trading_day"], oc.get("symbol", "MGC"),
                    oc["orb_label"], oc.get("orb_minutes", 15),
                    oc.get("entry_resolution", 5),
                    oc["rr_target"], oc["confirm_bars"], oc["entry_model"],
                    oc.get("outcome", "win"), oc["pnl_r"],
                ])

        con.commit()
        con.close()
        return db_path

    def test_load_validated_strategies_include_nested(self, tmp_path):
        """Union of baseline + nested strategies with median_risk_points from JOIN."""
        baseline = [_make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G4",
            orb_label="CME_REOPEN", filter_type="ORB_G4",
            median_risk_points=8.5,
        )]
        nested = [_make_strategy(
            strategy_id="N_MGC_0900_E1_RR2.0_CB5_ORB_G4",
            orb_label="CME_REOPEN", filter_type="ORB_G4",
            expectancy_r=0.25, median_risk_points=12.0,
        )]
        df_rows = _make_daily_features_rows(10)
        db_path = self._setup_nested_db(tmp_path, baseline, nested, df_rows, [])

        results = load_validated_strategies(db_path, "MGC", 0.10, include_nested=True)
        assert len(results) == 2

        sources = {r["source"] for r in results}
        assert sources == {"baseline", "nested"}

        # Verify nested has median_risk_points from nested_strategies JOIN
        nested_r = [r for r in results if r["source"] == "nested"][0]
        assert nested_r["median_risk_points"] == pytest.approx(12.0)

    def test_load_validated_strategies_nested_table_missing(self, tmp_path):
        """include_nested=True with no nested tables returns baseline only."""
        strategies = [_make_strategy()]
        db_path = _setup_db(tmp_path, strategies)

        results = load_validated_strategies(db_path, "MGC", 0.10, include_nested=True)
        assert len(results) == 1
        assert results[0]["source"] == "baseline"

    def test_build_portfolio_include_nested(self, tmp_path):
        """Build portfolio with both baseline and nested strategies."""
        baseline = [_make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G4",
            orb_label="CME_REOPEN", filter_type="ORB_G4",
        )]
        nested = [_make_strategy(
            strategy_id="N_MGC_1000_E1_RR2.0_CB5_ORB_G4",
            orb_label="TOKYO_OPEN", filter_type="ORB_G4",
            expectancy_r=0.20,
        )]
        df_rows = _make_daily_features_rows(10)
        db_path = self._setup_nested_db(tmp_path, baseline, nested, df_rows, [])

        portfolio = build_portfolio(db_path=db_path, instrument="MGC", include_nested=True)
        assert len(portfolio.strategies) == 2

        sources = {s.source for s in portfolio.strategies}
        assert sources == {"baseline", "nested"}

    def test_build_strategy_daily_series_nested(self, tmp_path):
        """Nested strategy reads from nested_outcomes, baseline from orb_outcomes."""
        df_rows_5m = _make_daily_features_rows(50, orb_minutes=5)
        df_rows_15m = _make_daily_features_rows(50, orb_minutes=15)

        baseline = [_make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
        )]
        nested = [_make_strategy(
            strategy_id="N_MGC_0900_E1_RR2.0_CB5_NO_FILTER",
            orb_label="CME_REOPEN", filter_type="NO_FILTER",
            expectancy_r=0.25,
        )]

        # Baseline outcomes on first 20 days
        outcomes_baseline = [
            {"trading_day": df_rows_5m[i]["trading_day"], "orb_label": "CME_REOPEN",
             "orb_minutes": 5, "rr_target": 2.0, "confirm_bars": 5,
             "entry_model": "E1", "pnl_r": 0.5}
            for i in range(20)
        ]
        # Nested outcomes on first 15 days
        outcomes_nested = [
            {"trading_day": df_rows_15m[i]["trading_day"], "orb_label": "CME_REOPEN",
             "orb_minutes": 15, "entry_resolution": 5,
             "rr_target": 2.0, "confirm_bars": 5, "entry_model": "E1", "pnl_r": 0.8}
            for i in range(15)
        ]

        db_path = self._setup_nested_db(
            tmp_path, baseline, nested,
            df_rows_5m + df_rows_15m,
            outcomes_baseline,
            nested_outcome_rows=outcomes_nested,
        )

        series_df, stats = build_strategy_daily_series(db_path, [
            baseline[0]["strategy_id"], nested[0]["strategy_id"],
        ])
        assert baseline[0]["strategy_id"] in stats
        assert nested[0]["strategy_id"] in stats
        assert stats[baseline[0]["strategy_id"]]["traded_days"] == 20
        assert stats[nested[0]["strategy_id"]]["traded_days"] == 15

    def test_build_strategy_daily_series_mixed_orb_minutes(self, tmp_path):
        """REGRESSION: baseline G6 on 5m ORBs vs nested G8 on 15m ORBs.

        15m ORBs are 1.5x larger so G8 filter passes much more often on 15m.
        This verifies Fix 2: each strategy uses its own orb_minutes daily_features.
        """
        df_rows_5m = _make_daily_features_rows(100, orb_minutes=5)
        df_rows_15m = _make_daily_features_rows(100, orb_minutes=15)

        # Baseline: G6 filter on 5m ORBs (0900 sizes: 5.0-9.5)
        # G6 = orb_size >= 6.0 -> eligible when i%10 >= 2 -> 80% eligible
        baseline = [_make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G6",
            orb_label="CME_REOPEN", filter_type="ORB_G6",
        )]
        # Nested: G8 filter on 15m ORBs (0900 sizes: 7.5-14.25, i.e. 1.5x)
        # G8 = orb_size >= 8.0 -> eligible when (5.0 + i%10*0.5)*1.5 >= 8.0
        # i.e. 5.0+x >= 5.33 -> x >= 0.33 -> i%10 >= 1 -> 90% eligible
        nested = [_make_strategy(
            strategy_id="N_MGC_0900_E1_RR2.0_CB5_ORB_G8",
            orb_label="CME_REOPEN", filter_type="ORB_G8",
            expectancy_r=0.20,
        )]

        db_path = self._setup_nested_db(
            tmp_path, baseline, nested,
            df_rows_5m + df_rows_15m, [], nested_outcome_rows=[],
        )

        series_df, stats = build_strategy_daily_series(db_path, [
            baseline[0]["strategy_id"], nested[0]["strategy_id"],
        ])

        baseline_eligible = stats[baseline[0]["strategy_id"]]["eligible_days"]
        nested_eligible = stats[nested[0]["strategy_id"]]["eligible_days"]

        # Nested G8 on 15m should have MORE eligible days than baseline G6 on 5m
        # because 15m ORBs are 1.5x larger
        assert nested_eligible > baseline_eligible, (
            f"Nested G8 on 15m ({nested_eligible}) should have more eligible days "
            f"than baseline G6 on 5m ({baseline_eligible})"
        )

    def test_portfolio_json_roundtrip_nested(self, tmp_path):
        """JSON roundtrip preserves source field for nested strategies."""
        baseline = [_make_strategy(
            strategy_id="MGC_0900_E1_RR2.0_CB5_ORB_G4",
            orb_label="CME_REOPEN", filter_type="ORB_G4",
        )]
        nested = [_make_strategy(
            strategy_id="N_MGC_1000_E1_RR2.0_CB5_ORB_G4",
            orb_label="TOKYO_OPEN", filter_type="ORB_G4",
            expectancy_r=0.20,
        )]
        df_rows = _make_daily_features_rows(10)
        db_path = self._setup_nested_db(tmp_path, baseline, nested, df_rows, [])

        portfolio = build_portfolio(db_path=db_path, instrument="MGC", include_nested=True)
        json_str = portfolio.to_json()
        loaded = Portfolio.from_json(json_str)

        sources = {s.source for s in loaded.strategies}
        assert sources == {"baseline", "nested"}
        for orig, loaded_s in zip(portfolio.strategies, loaded.strategies):
            assert orig.source == loaded_s.source


class TestCorrelationFilter:
    """Tests for correlation-aware diversification (Phase 1 risk hardening)."""

    def test_correlated_cb_variants_rejected(self):
        """CB1 + CB2 on same ORB/filter at rho >= 0.85 should keep only the first."""
        candidates = [
            _make_strategy(strategy_id="s_cb1", orb_label="CME_REOPEN", confirm_bars=1, expectancy_r=0.30),
            _make_strategy(strategy_id="s_cb2", orb_label="CME_REOPEN", confirm_bars=2, expectancy_r=0.28),
            _make_strategy(strategy_id="s_cb3", orb_label="CME_REOPEN", confirm_bars=3, expectancy_r=0.26),
        ]
        # Fake correlation matrix: cb1-cb2 = 0.95, cb1-cb3 = 0.90, cb2-cb3 = 0.92
        import pandas as pd
        ids = ["s_cb1", "s_cb2", "s_cb3"]
        corr = pd.DataFrame(
            [[1.0, 0.95, 0.90], [0.95, 1.0, 0.92], [0.90, 0.92, 1.0]],
            index=ids, columns=ids,
        )
        selected = diversify_strategies(
            candidates, max_strategies=10, corr_matrix=corr, max_correlation=0.85,
        )
        # Only the first (highest ExpR) should survive
        assert len(selected) == 1
        assert selected[0]["strategy_id"] == "s_cb1"

    def test_uncorrelated_strategies_pass(self):
        """Strategies from different sessions with low correlation pass through."""
        candidates = [
            _make_strategy(strategy_id="s_0900", orb_label="CME_REOPEN", expectancy_r=0.30),
            _make_strategy(strategy_id="s_1800", orb_label="LONDON_METALS", expectancy_r=0.25),
            _make_strategy(strategy_id="s_2300", orb_label="US_DATA_830", expectancy_r=0.20),
        ]
        import pandas as pd
        ids = ["s_0900", "s_1800", "s_2300"]
        corr = pd.DataFrame(
            [[1.0, 0.15, 0.10], [0.15, 1.0, 0.20], [0.10, 0.20, 1.0]],
            index=ids, columns=ids,
        )
        selected = diversify_strategies(
            candidates, max_strategies=10, corr_matrix=corr, max_correlation=0.85,
        )
        assert len(selected) == 3

    def test_none_corr_matrix_backward_compatible(self):
        """corr_matrix=None gives identical behavior to old code."""
        candidates = [
            _make_strategy(strategy_id=f"s{i}", orb_label="CME_REOPEN", expectancy_r=0.30 - i*0.01)
            for i in range(5)
        ]
        selected = diversify_strategies(candidates, max_strategies=10, corr_matrix=None)
        assert len(selected) == 5

    def test_empty_candidates_with_corr_matrix(self):
        """Empty candidates with corr_matrix doesn't crash."""
        import pandas as pd
        corr = pd.DataFrame()
        selected = diversify_strategies([], max_strategies=10, corr_matrix=corr)
        assert selected == []

    def test_nan_correlation_does_not_block(self):
        """NaN correlation (insufficient overlap) should not block selection."""
        candidates = [
            _make_strategy(strategy_id="s_a", orb_label="CME_REOPEN", expectancy_r=0.30),
            _make_strategy(strategy_id="s_b", orb_label="CME_REOPEN", expectancy_r=0.25),
        ]
        import pandas as pd
        corr = pd.DataFrame(
            [[1.0, float("nan")], [float("nan"), 1.0]],
            index=["s_a", "s_b"], columns=["s_a", "s_b"],
        )
        selected = diversify_strategies(
            candidates, max_strategies=10, corr_matrix=corr, max_correlation=0.85,
        )
        assert len(selected) == 2


class TestFamilyHeadsOnly:
    """Tests for family_heads_only dedup in load_validated_strategies."""

    def test_returns_only_heads(self, tmp_path):
        """With family_heads_only=True, only head strategies are returned."""
        strategies = [
            _make_strategy(strategy_id="MGC_0900_E1_RR2.0_CB2_ORB_G5",
                           orb_label="CME_REOPEN", expectancy_r=0.30),
            _make_strategy(strategy_id="MGC_0900_E1_RR2.5_CB2_ORB_G5",
                           orb_label="CME_REOPEN", expectancy_r=0.25),
            _make_strategy(strategy_id="MGC_0900_E1_RR2.0_CB2_ORB_G8",
                           orb_label="CME_REOPEN", expectancy_r=0.40),
        ]
        db_path = _setup_db(tmp_path, strategies)

        # Create edge_families: s1 is head of family A, s3 is head of family B
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO edge_families
            (family_hash, instrument, member_count, trade_day_count,
             head_strategy_id, head_expectancy_r)
            VALUES
            ('hash_a', 'MGC', 2, 100, 'MGC_0900_E1_RR2.0_CB2_ORB_G5', 0.30),
            ('hash_b', 'MGC', 1, 50, 'MGC_0900_E1_RR2.0_CB2_ORB_G8', 0.40)
        """)
        con.commit()
        con.close()

        # With flag: only 2 heads
        results = load_validated_strategies(
            db_path, "MGC", 0.10, family_heads_only=True,
        )
        ids = {r["strategy_id"] for r in results}
        assert ids == {
            "MGC_0900_E1_RR2.0_CB2_ORB_G5",
            "MGC_0900_E1_RR2.0_CB2_ORB_G8",
        }

    def test_without_flag_returns_all(self, tmp_path):
        """Without family_heads_only, all 3 strategies are returned."""
        strategies = [
            _make_strategy(strategy_id="MGC_0900_E1_RR2.0_CB2_ORB_G5",
                           orb_label="CME_REOPEN", expectancy_r=0.30),
            _make_strategy(strategy_id="MGC_0900_E1_RR2.5_CB2_ORB_G5",
                           orb_label="CME_REOPEN", expectancy_r=0.25),
            _make_strategy(strategy_id="MGC_0900_E1_RR2.0_CB2_ORB_G8",
                           orb_label="CME_REOPEN", expectancy_r=0.40),
        ]
        db_path = _setup_db(tmp_path, strategies)

        results = load_validated_strategies(db_path, "MGC", 0.10)
        assert len(results) == 3

    def test_graceful_without_edge_families(self, tmp_path):
        """family_heads_only=True with no edge_families table returns all."""
        strategies = [_make_strategy()]
        db_path = _setup_db(tmp_path, strategies)

        # Drop edge_families table
        con = duckdb.connect(str(db_path))
        con.execute("DROP TABLE IF EXISTS edge_families")
        con.commit()
        con.close()

        results = load_validated_strategies(
            db_path, "MGC", 0.10, family_heads_only=True,
        )
        assert len(results) == 1


class TestClassificationWeights:
    """FIX5: CORE=1.0, REGIME=0.5, INVALID=excluded."""

    def test_core_gets_weight_1(self, tmp_path):
        strats = [_make_strategy(sample_size=200)]
        db_path = _setup_db(tmp_path, strats)
        pf = build_portfolio(db_path=db_path, instrument="MGC")
        assert len(pf.strategies) == 1
        assert pf.strategies[0].weight == 1.0
        assert pf.strategies[0].classification == "CORE"

    def test_regime_gets_weight_half(self, tmp_path):
        strats = [_make_strategy(sample_size=50)]
        db_path = _setup_db(tmp_path, strats)
        pf = build_portfolio(db_path=db_path, instrument="MGC")
        assert len(pf.strategies) == 1
        assert pf.strategies[0].weight == 0.5
        assert pf.strategies[0].classification == "REGIME"

    def test_invalid_excluded(self, tmp_path):
        strats = [_make_strategy(sample_size=20)]
        db_path = _setup_db(tmp_path, strats)
        pf = build_portfolio(db_path=db_path, instrument="MGC")
        assert len(pf.strategies) == 0

    def test_mixed_portfolio_weights(self, tmp_path):
        strats = [
            _make_strategy(strategy_id="CORE_1", sample_size=200, orb_label="TOKYO_OPEN"),
            _make_strategy(strategy_id="REGIME_1", sample_size=60, orb_label="LONDON_METALS"),
            _make_strategy(strategy_id="INVALID_1", sample_size=15, orb_label="US_DATA_830"),
        ]
        db_path = _setup_db(tmp_path, strats)
        pf = build_portfolio(db_path=db_path, instrument="MGC")
        ids = {s.strategy_id: s for s in pf.strategies}
        assert "CORE_1" in ids
        assert ids["CORE_1"].weight == 1.0
        assert "REGIME_1" in ids
        assert ids["REGIME_1"].weight == 0.5
        assert "INVALID_1" not in ids


class TestCLI:
    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["portfolio", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.portfolio import main
            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out
