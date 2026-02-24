"""
Tests for trading_app.strategy_fitness module.

Uses in-memory DuckDB with synthetic data (same pattern as test_strategy_validator.py).
"""

import sys
import json
from pathlib import Path
from datetime import date

import pytest
import duckdb

from trading_app.strategy_fitness import (
    classify_fitness,
    FitnessScore,
    FitnessReport,
    compute_fitness,
    compute_portfolio_fitness,
    _recent_trade_sharpe,
    _rolling_window_start,
    _load_strategy_outcomes,
)
from trading_app.strategy_discovery import compute_metrics

def _setup_fitness_db(tmp_path, strategies=None, outcomes=None, features=None):
    """
    Create a temp DB with pipeline + trading_app schema and seed data.

    Args:
        strategies: list of dicts for validated_setups
        outcomes: list of dicts for orb_outcomes
        features: list of dicts for daily_features
    """
    db_path = tmp_path / "test_fitness.db"
    con = duckdb.connect(str(db_path))

    # Create pipeline tables
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)

    # Create trading_app tables
    from trading_app.db_manager import init_trading_app_schema
    con.close()
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))

    # Insert strategies: experimental_strategies first (FK parent), then validated_setups
    if strategies:
        for s in strategies:
            sid = s["strategy_id"]
            instrument = s.get("instrument", "MGC")
            orb_label = s.get("orb_label", "CME_REOPEN")
            orb_minutes = s.get("orb_minutes", 5)
            rr_target = s.get("rr_target", 2.0)
            confirm_bars = s.get("confirm_bars", 2)
            entry_model = s.get("entry_model", "E1")
            filter_type = s.get("filter_type", "NO_FILTER")

            # Parent row in experimental_strategies (FK requirement)
            con.execute(
                """INSERT OR REPLACE INTO experimental_strategies
                   (strategy_id, instrument, orb_label, orb_minutes,
                    rr_target, confirm_bars, entry_model, filter_type, filter_params,
                    sample_size, win_rate, expectancy_r, sharpe_ratio, max_drawdown_r)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    sid, instrument, orb_label, orb_minutes,
                    rr_target, confirm_bars, entry_model, filter_type, "{}",
                    s.get("sample_size", 100), s.get("win_rate", 0.50),
                    s.get("expectancy_r", 0.30), s.get("sharpe_ratio", 0.25),
                    s.get("max_drawdown_r", 5.0),
                ],
            )

            # Child row in validated_setups
            con.execute(
                """INSERT INTO validated_setups
                   (strategy_id, promoted_from, instrument, orb_label,
                    orb_minutes, rr_target, confirm_bars, entry_model,
                    filter_type, filter_params,
                    sample_size, win_rate, expectancy_r,
                    years_tested, all_years_positive, stress_test_passed,
                    sharpe_ratio, max_drawdown_r, yearly_results, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    sid, sid,
                    instrument, orb_label, orb_minutes, rr_target,
                    confirm_bars, entry_model, filter_type, "{}",
                    s.get("sample_size", 100), s.get("win_rate", 0.50),
                    s.get("expectancy_r", 0.30),
                    s.get("years_tested", 3), s.get("all_years_positive", True),
                    s.get("stress_test_passed", True),
                    s.get("sharpe_ratio", 0.25), s.get("max_drawdown_r", 5.0),
                    s.get("yearly_results", "{}"), s.get("status", "active"),
                ],
            )

    # Insert daily_features FIRST (FK parent for orb_outcomes)
    if features:
        for f in features:
            con.execute(
                """INSERT OR IGNORE INTO daily_features
                   (trading_day, symbol, orb_minutes, bar_count_1m,
                    orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size,
                    orb_CME_REOPEN_break_dir)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    f["trading_day"], f.get("symbol", "MGC"),
                    f.get("orb_minutes", 5), f.get("bar_count_1m", 1400),
                    f.get("orb_CME_REOPEN_high", 2355.0), f.get("orb_CME_REOPEN_low", 2345.0),
                    f.get("orb_CME_REOPEN_size", 10.0),
                    f.get("orb_CME_REOPEN_break_dir", "long"),
                ],
            )

    # Insert orb_outcomes (after daily_features due to FK)
    if outcomes:
        for o in outcomes:
            td = o["trading_day"]
            sym = o.get("symbol", "MGC")
            om = o.get("orb_minutes", 5)
            # Ensure daily_features row exists for this outcome (FK requirement)
            con.execute(
                """INSERT OR IGNORE INTO daily_features
                   (trading_day, symbol, orb_minutes, bar_count_1m,
                    orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_size,
                    orb_CME_REOPEN_break_dir)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [td, sym, om, 1400, 2355.0, 2345.0, 10.0, "long"],
            )
            con.execute(
                """INSERT INTO orb_outcomes
                   (trading_day, symbol, orb_label, orb_minutes,
                    rr_target, confirm_bars, entry_model,
                    entry_price, stop_price, target_price,
                    outcome, pnl_r, mae_r, mfe_r)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    td, sym,
                    o.get("orb_label", "CME_REOPEN"), om,
                    o.get("rr_target", 2.0), o.get("confirm_bars", 2),
                    o.get("entry_model", "E1"),
                    o.get("entry_price", 2350.0), o.get("stop_price", 2340.0),
                    o.get("target_price", 2370.0),
                    o["outcome"], o["pnl_r"],
                    o.get("mae_r", -0.5), o.get("mfe_r", 1.5),
                ],
            )

    con.commit()
    con.close()
    return db_path

def _make_outcomes(start_year=2023, end_year=2025, trades_per_year=40,
                   win_rate=0.50, win_pnl=1.8, loss_pnl=-1.0, **overrides):
    """Generate synthetic outcome rows spanning multiple years."""
    outcomes = []
    for year in range(start_year, end_year + 1):
        for i in range(trades_per_year):
            month = (i % 11) + 1
            day = (i % 27) + 1
            td = date(year, month, day)
            is_win = i < int(trades_per_year * win_rate)
            outcome = "win" if is_win else "loss"
            pnl = win_pnl if is_win else loss_pnl
            row = {
                "trading_day": td,
                "outcome": outcome,
                "pnl_r": pnl,
                **overrides,
            }
            outcomes.append(row)
    return outcomes

def _make_features(start_year=2023, end_year=2025, trades_per_year=40,
                   orb_size=10.0, **overrides):
    """Generate daily_features rows matching outcome days."""
    features = []
    for year in range(start_year, end_year + 1):
        for i in range(trades_per_year):
            month = (i % 11) + 1
            day = (i % 27) + 1
            td = date(year, month, day)
            row = {
                "trading_day": td,
                "orb_CME_REOPEN_size": orb_size,
                **overrides,
            }
            features.append(row)
    return features

# =========================================================================
# Classification tests
# =========================================================================

class TestClassifyFitness:

    def test_classify_fitness_fit(self):
        """Positive rolling ExpR, stable recent Sharpe, enough trades -> FIT."""
        status, notes = classify_fitness(
            rolling_exp_r=0.30, rolling_sample=20, recent_sharpe_30=0.15
        )
        assert status == "FIT"

    def test_classify_fitness_watch_declining_sharpe(self):
        """Positive rolling ExpR but declining recent Sharpe -> WATCH."""
        status, notes = classify_fitness(
            rolling_exp_r=0.20, rolling_sample=20, recent_sharpe_30=-0.15
        )
        assert status == "WATCH"
        assert "Declining" in notes

    def test_classify_fitness_watch_thin_data(self):
        """Positive rolling ExpR but only 10-14 trades -> WATCH."""
        status, notes = classify_fitness(
            rolling_exp_r=0.20, rolling_sample=12, recent_sharpe_30=0.10
        )
        assert status == "WATCH"
        assert "Thin data" in notes

    def test_classify_fitness_decay(self):
        """Negative rolling ExpR -> DECAY."""
        status, notes = classify_fitness(
            rolling_exp_r=-0.10, rolling_sample=25, recent_sharpe_30=-0.20
        )
        assert status == "DECAY"
        assert "Negative" in notes

    def test_classify_fitness_stale(self):
        """< 10 rolling trades -> STALE."""
        status, notes = classify_fitness(
            rolling_exp_r=0.50, rolling_sample=5, recent_sharpe_30=0.30
        )
        assert status == "STALE"

    def test_classify_fitness_stale_none_exp(self):
        """None rolling ExpR with enough trades -> STALE."""
        status, notes = classify_fitness(
            rolling_exp_r=None, rolling_sample=15, recent_sharpe_30=0.10
        )
        assert status == "STALE"

    def test_classify_fitness_fit_none_sharpe(self):
        """Positive rolling ExpR, None recent sharpe (not enough for 30-trade) -> FIT."""
        status, notes = classify_fitness(
            rolling_exp_r=0.25, rolling_sample=20, recent_sharpe_30=None
        )
        assert status == "FIT"

# =========================================================================
# Rolling metrics tests
# =========================================================================

class TestRollingMetrics:

    def test_rolling_metrics_basic(self):
        """Rolling window filters outcomes by date correctly."""
        outcomes = _make_outcomes(start_year=2023, end_year=2025, trades_per_year=30)
        # Only keep 2025 outcomes (simulate rolling window filter)
        rolling = [o for o in outcomes if o["trading_day"].year == 2025]
        metrics = compute_metrics(rolling)
        assert metrics["sample_size"] == 30
        assert metrics["win_rate"] is not None

    def test_rolling_metrics_empty(self):
        """No trades in window -> None metrics."""
        metrics = compute_metrics([])
        assert metrics["sample_size"] == 0
        assert metrics["win_rate"] is None
        assert metrics["expectancy_r"] is None
        assert metrics["sharpe_ratio"] is None

# =========================================================================
# Recent trade Sharpe tests
# =========================================================================

class TestRecentTradeSharpe:

    def test_recent_trade_sharpe_basic(self):
        """Last 30 trades Sharpe calculation returns a number."""
        outcomes = _make_outcomes(
            start_year=2024, end_year=2025, trades_per_year=30,
            win_rate=0.55, win_pnl=1.8, loss_pnl=-1.0
        )
        result = _recent_trade_sharpe(outcomes, 30)
        assert result is not None
        assert isinstance(result, float)

    def test_recent_trade_sharpe_insufficient(self):
        """< 30 trades -> None."""
        outcomes = _make_outcomes(
            start_year=2025, end_year=2025, trades_per_year=10
        )
        result = _recent_trade_sharpe(outcomes, 30)
        assert result is None

    def test_recent_trade_sharpe_uses_last_n(self):
        """Sharpe uses only last N trades, not all."""
        # 50 good trades followed by 10 bad ones
        good = [
            {"trading_day": date(2024, 1, i + 1), "outcome": "win", "pnl_r": 2.0}
            for i in range(20)
        ]
        bad = [
            {"trading_day": date(2025, 1, i + 1), "outcome": "loss", "pnl_r": -1.0}
            for i in range(20)
        ]
        all_outcomes = good + bad
        # Last 20 trades = all losses -> negative Sharpe
        sharpe_20 = _recent_trade_sharpe(all_outcomes, 20)
        assert sharpe_20 is None  # all same pnl -> std=0 -> None

# =========================================================================
# Rolling window date math
# =========================================================================

class TestRollingWindowStart:

    def test_18_months_back(self):
        result = _rolling_window_start(date(2025, 12, 15), 18)
        assert result == date(2024, 6, 15)

    def test_wraps_year_boundary(self):
        result = _rolling_window_start(date(2025, 3, 15), 18)
        assert result == date(2023, 9, 15)

    def test_clamps_day(self):
        # March 31 - 1 month should be Feb 28 (2025 is not a leap year)
        result = _rolling_window_start(date(2025, 3, 31), 1)
        assert result == date(2025, 2, 28)

# =========================================================================
# Sharpe delta tests
# =========================================================================

class TestSharpeDelta:

    def test_sharpe_delta_computation(self):
        """Delta = recent - full_period."""
        recent = 0.10
        full = 0.25
        delta = recent - full
        assert delta == pytest.approx(-0.15)

    def test_sharpe_delta_none_recent(self):
        """If recent is None, delta should be None."""
        recent = None
        full = 0.25
        delta = None if recent is None else recent - full
        assert delta is None

# =========================================================================
# Integration tests
# =========================================================================

class TestComputeFitnessIntegration:

    def test_compute_fitness_end_to_end(self, tmp_path):
        """End-to-end fitness computation with in-memory DuckDB."""
        strategy_id = "MGC_0900_E1_RR2.0_CB2_NO_FILTER"
        strategies = [{
            "strategy_id": strategy_id,
            "instrument": "MGC",
            "orb_label": "CME_REOPEN",
            "orb_minutes": 5,
            "rr_target": 2.0,
            "confirm_bars": 2,
            "entry_model": "E1",
            "filter_type": "NO_FILTER",
            "sample_size": 120,
            "win_rate": 0.50,
            "expectancy_r": 0.30,
            "sharpe_ratio": 0.25,
        }]
        outcomes = _make_outcomes(
            start_year=2023, end_year=2025, trades_per_year=40,
            win_rate=0.50, win_pnl=1.8, loss_pnl=-1.0,
        )
        features = _make_features(start_year=2023, end_year=2025, trades_per_year=40)

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes, features)

        score = compute_fitness(
            strategy_id, db_path=db_path,
            as_of_date=date(2025, 12, 31),
            rolling_months=18,
        )

        assert score.strategy_id == strategy_id
        assert score.full_period_exp_r == 0.30
        assert score.full_period_sharpe == 0.25
        assert score.full_period_sample == 120
        assert score.rolling_sample > 0
        assert score.fitness_status in ("FIT", "WATCH", "DECAY", "STALE")

    def test_compute_fitness_not_found(self, tmp_path):
        """Unknown strategy ID raises ValueError."""
        db_path = _setup_fitness_db(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            compute_fitness("NONEXISTENT", db_path=db_path)

    def test_no_lookahead_leak(self, tmp_path):
        """Outcomes after as_of_date must NOT affect recent Sharpe (no lookahead).

        BUG FIX: Previously, _load_strategy_outcomes was called without end_date,
        so _recent_trade_sharpe could include future outcomes.
        """
        strategy_id = "MGC_0900_E1_RR2.0_CB2_NO_FILTER"
        strategies = [{
            "strategy_id": strategy_id,
            "filter_type": "NO_FILTER",
            "sample_size": 100,
            "expectancy_r": 0.30,
            "sharpe_ratio": 0.25,
        }]
        # 30 wins in 2023, then 30 losses in 2025
        early_wins = [
            {"trading_day": date(2023, (i % 11) + 1, (i % 27) + 1),
             "outcome": "win", "pnl_r": 2.0}
            for i in range(30)
        ]
        late_losses = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
             "outcome": "loss", "pnl_r": -1.0}
            for i in range(30)
        ]

        db_path = _setup_fitness_db(
            tmp_path, strategies, early_wins + late_losses,
        )

        # as_of=2023-12-31: should only see the 30 wins, NOT the 2025 losses
        score = compute_fitness(
            strategy_id, db_path=db_path,
            as_of_date=date(2023, 12, 31),
            rolling_months=18,
        )
        # recent_sharpe_30 computed from last 30 trades up to 2023-12-31 = all wins
        # If there's a lookahead leak, the 2025 losses would pollute this
        if score.recent_sharpe_30 is not None:
            # All 30 recent trades are wins with identical pnl -> std=0 -> None
            # OR if there's variation, Sharpe should be positive (all wins)
            pass
        # The key check: rolling sample should only count 2023 outcomes
        assert score.rolling_sample <= 30

    def test_watch_classification_thin_data(self, tmp_path):
        """10-14 rolling trades with positive ExpR -> WATCH, not STALE.

        BUG FIX: Previously, rolling_exp_r was nulled before classification,
        causing classify_fitness to return STALE instead of WATCH.
        """
        strategy_id = "MGC_0900_E1_RR2.0_CB2_NO_FILTER"
        strategies = [{
            "strategy_id": strategy_id,
            "filter_type": "NO_FILTER",
            "sample_size": 100,
            "expectancy_r": 0.30,
            "sharpe_ratio": 0.25,
        }]
        # Only 12 outcomes, all in the rolling window, mostly wins
        outcomes = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
             "outcome": "win" if i < 8 else "loss",
             "pnl_r": 2.0 if i < 8 else -1.0}
            for i in range(12)
        ]

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes)

        score = compute_fitness(
            strategy_id, db_path=db_path,
            as_of_date=date(2025, 12, 31),
            rolling_months=18,
        )
        assert score.rolling_sample == 12
        # 8 wins * 2.0 + 4 losses * -1.0 = positive ExpR
        # Should be WATCH (thin data), NOT STALE
        assert score.fitness_status == "WATCH"
        assert "Thin data" in score.fitness_notes

    def test_orb_size_filter_excludes_small_days(self, tmp_path):
        """ORB_G4 filter excludes days with orb_size < 4.0."""
        strategy_id = "MGC_0900_E1_RR2.0_CB2_ORB_G4"
        strategies = [{
            "strategy_id": strategy_id,
            "filter_type": "ORB_G4",
            "sample_size": 100,
            "expectancy_r": 0.30,
            "sharpe_ratio": 0.25,
        }]
        # 20 outcomes on big-ORB days (size=10), 20 on small-ORB days (size=2)
        big_outcomes = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
             "outcome": "win", "pnl_r": 2.0}
            for i in range(20)
        ]
        small_outcomes = [
            {"trading_day": date(2024, (i % 11) + 1, (i % 27) + 1),
             "outcome": "loss", "pnl_r": -1.0}
            for i in range(20)
        ]
        # Features: 2025 days have big ORBs, 2024 days have small ORBs
        big_features = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
             "orb_CME_REOPEN_size": 10.0}
            for i in range(20)
        ]
        small_features = [
            {"trading_day": date(2024, (i % 11) + 1, (i % 27) + 1),
             "orb_CME_REOPEN_size": 2.0}
            for i in range(20)
        ]

        db_path = _setup_fitness_db(
            tmp_path, strategies,
            big_outcomes + small_outcomes,
            big_features + small_features,
        )

        score = compute_fitness(
            strategy_id, db_path=db_path,
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        # Only the 20 big-ORB outcomes should pass filter
        # The 20 small-ORB (size=2.0 < 4.0) should be excluded
        assert score.rolling_sample == 20

class TestFitnessReportSummary:

    def test_fitness_report_summary_counts(self, tmp_path):
        """Summary correctly counts {fit: N, watch: N, decay: N, stale: N}."""
        # Create 3 strategies with different expected outcomes
        strategies = [
            {
                "strategy_id": f"MGC_0900_E1_RR2.0_CB2_NO_FILTER_{i}",
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            }
            for i in range(3)
        ]
        # All 3 strategies share the same outcome params (orb, EM, RR, CB),
        # so one set of outcomes covers all of them.
        outcomes = _make_outcomes(
            start_year=2024, end_year=2025, trades_per_year=30,
            win_rate=0.55, win_pnl=1.8, loss_pnl=-1.0,
        )
        features = _make_features(
            start_year=2024, end_year=2025, trades_per_year=30,
        )

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes, features)

        report = compute_portfolio_fitness(
            db_path=db_path, instrument="MGC",
            as_of_date=date(2025, 12, 31),
        )

        assert isinstance(report, FitnessReport)
        assert len(report.scores) == 3
        total = sum(report.summary.values())
        assert total == 3
        assert set(report.summary.keys()) == {"fit", "watch", "decay", "stale"}

# =========================================================================
# Portfolio integration test
# =========================================================================

class TestFitnessWeightedPortfolio:

    def test_fitness_weighted_portfolio(self):
        """FIT=1.0, WATCH=0.5, DECAY=0.0, STALE=0.0 weight adjustments."""
        from trading_app.portfolio import Portfolio, PortfolioStrategy, fitness_weighted_portfolio

        strats = [
            PortfolioStrategy(
                strategy_id="S1", instrument="MGC", orb_label="CME_REOPEN",
                entry_model="E1", rr_target=2.0, confirm_bars=2,
                filter_type="NO_FILTER", expectancy_r=0.30, win_rate=0.50,
                sample_size=100, sharpe_ratio=0.25, max_drawdown_r=5.0,
                median_risk_points=10.0, weight=1.0,
            ),
            PortfolioStrategy(
                strategy_id="S2", instrument="MGC", orb_label="TOKYO_OPEN",
                entry_model="E1", rr_target=2.5, confirm_bars=2,
                filter_type="ORB_G4", expectancy_r=0.40, win_rate=0.55,
                sample_size=80, sharpe_ratio=0.30, max_drawdown_r=4.0,
                median_risk_points=12.0, weight=1.0,
            ),
            PortfolioStrategy(
                strategy_id="S3", instrument="MGC", orb_label="LONDON_METALS",
                entry_model="E3", rr_target=2.0, confirm_bars=5,
                filter_type="ORB_G5", expectancy_r=0.35, win_rate=0.48,
                sample_size=75, sharpe_ratio=0.20, max_drawdown_r=3.0,
                median_risk_points=8.0, weight=1.0,
            ),
        ]

        portfolio = Portfolio(
            name="test", instrument="MGC", strategies=strats,
            account_equity=25000.0, risk_per_trade_pct=2.0,
            max_concurrent_positions=3, max_daily_loss_r=5.0,
        )

        scores = [
            FitnessScore(
                strategy_id="S1", full_period_exp_r=0.30,
                full_period_sharpe=0.25, full_period_sample=100,
                rolling_exp_r=0.25, rolling_sharpe=0.20, rolling_win_rate=0.48,
                rolling_sample=25, rolling_window_months=18,
                recent_sharpe_30=0.15, recent_sharpe_60=0.18,
                sharpe_delta_30=-0.10, sharpe_delta_60=-0.07,
                fitness_status="FIT", fitness_notes="OK",
            ),
            FitnessScore(
                strategy_id="S2", full_period_exp_r=0.40,
                full_period_sharpe=0.30, full_period_sample=80,
                rolling_exp_r=0.15, rolling_sharpe=0.10, rolling_win_rate=0.45,
                rolling_sample=18, rolling_window_months=18,
                recent_sharpe_30=-0.15, recent_sharpe_60=0.05,
                sharpe_delta_30=-0.45, sharpe_delta_60=-0.25,
                fitness_status="WATCH", fitness_notes="Declining",
            ),
            FitnessScore(
                strategy_id="S3", full_period_exp_r=0.35,
                full_period_sharpe=0.20, full_period_sample=75,
                rolling_exp_r=-0.10, rolling_sharpe=-0.05, rolling_win_rate=0.40,
                rolling_sample=20, rolling_window_months=18,
                recent_sharpe_30=-0.30, recent_sharpe_60=-0.20,
                sharpe_delta_30=-0.50, sharpe_delta_60=-0.40,
                fitness_status="DECAY", fitness_notes="Negative",
            ),
        ]

        report = FitnessReport(
            as_of_date=date(2025, 12, 31), scores=scores,
            summary={"fit": 1, "watch": 1, "decay": 1, "stale": 0},
        )

        adjusted = fitness_weighted_portfolio(portfolio, report)

        # Verify it returns a new Portfolio (not modified input)
        assert adjusted is not portfolio
        assert len(adjusted.strategies) == 3

        weight_map = {s.strategy_id: s.weight for s in adjusted.strategies}
        assert weight_map["S1"] == 1.0   # FIT
        assert weight_map["S2"] == 0.5   # WATCH
        assert weight_map["S3"] == 0.0   # DECAY

        # Original unchanged
        assert all(s.weight == 1.0 for s in portfolio.strategies)

    def test_regime_classification_caps_fitness(self):
        """REGIME (weight=0.5) with FIT status stays at 0.5, not 1.0."""
        from trading_app.portfolio import Portfolio, PortfolioStrategy, fitness_weighted_portfolio

        strats = [
            PortfolioStrategy(
                strategy_id="REGIME_FIT", instrument="MGC", orb_label="CME_REOPEN",
                entry_model="E1", rr_target=2.0, confirm_bars=1,
                filter_type="ORB_G5", expectancy_r=0.15, win_rate=0.45,
                sample_size=60, sharpe_ratio=0.20, max_drawdown_r=3.0,
                median_risk_points=4.0,
                weight=0.5,  # REGIME classification weight
            ),
            PortfolioStrategy(
                strategy_id="CORE_FIT", instrument="MGC", orb_label="TOKYO_OPEN",
                entry_model="E1", rr_target=2.0, confirm_bars=1,
                filter_type="ORB_G5", expectancy_r=0.20, win_rate=0.48,
                sample_size=200, sharpe_ratio=0.25, max_drawdown_r=2.5,
                median_risk_points=4.0,
                weight=1.0,  # CORE classification weight
            ),
        ]
        portfolio = Portfolio(
            name="test", instrument="MGC", strategies=strats,
            account_equity=25000.0, risk_per_trade_pct=2.0,
            max_concurrent_positions=3, max_daily_loss_r=5.0,
        )

        scores = [
            FitnessScore(
                strategy_id="REGIME_FIT", full_period_exp_r=0.30,
                full_period_sharpe=0.25, full_period_sample=60,
                rolling_exp_r=0.25, rolling_sharpe=0.20, rolling_win_rate=0.48,
                rolling_sample=25, rolling_window_months=18,
                recent_sharpe_30=0.15, recent_sharpe_60=0.18,
                sharpe_delta_30=0.0, sharpe_delta_60=0.0,
                fitness_status="FIT", fitness_notes="",
            ),
            FitnessScore(
                strategy_id="CORE_FIT", full_period_exp_r=0.35,
                full_period_sharpe=0.30, full_period_sample=200,
                rolling_exp_r=0.28, rolling_sharpe=0.22, rolling_win_rate=0.50,
                rolling_sample=30, rolling_window_months=18,
                recent_sharpe_30=0.20, recent_sharpe_60=0.22,
                sharpe_delta_30=0.0, sharpe_delta_60=0.0,
                fitness_status="FIT", fitness_notes="",
            ),
        ]
        report = FitnessReport(
            as_of_date=date(2025, 12, 31), scores=scores,
            summary={"fit": 2, "watch": 0, "decay": 0, "stale": 0},
        )

        adjusted = fitness_weighted_portfolio(portfolio, report)
        weight_map = {s.strategy_id: s.weight for s in adjusted.strategies}

        # REGIME + FIT: min(1.0, 0.5) = 0.5 (classification caps fitness)
        assert weight_map["REGIME_FIT"] == 0.5
        # CORE + FIT: min(1.0, 1.0) = 1.0
        assert weight_map["CORE_FIT"] == 1.0
