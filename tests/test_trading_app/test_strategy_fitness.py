"""
Tests for trading_app.strategy_fitness module.

Uses in-memory DuckDB with synthetic data (same pattern as test_strategy_validator.py).
"""

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from trading_app.strategy_discovery import compute_metrics
from trading_app.strategy_fitness import (
    MIN_ROLLING_FIT,
    MIN_ROLLING_WATCH,
    SHARPE_DECAY_THRESHOLD,
    DecayDiagnosis,
    FitnessReport,
    FitnessScore,
    _compute_fitness_from_cache,
    _load_strategy_outcomes,
    _recent_trade_sharpe,
    _rolling_window_start,
    classify_fitness,
    compute_fitness,
    compute_portfolio_fitness,
    diagnose_decay,
)


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
                    sid,
                    instrument,
                    orb_label,
                    orb_minutes,
                    rr_target,
                    confirm_bars,
                    entry_model,
                    filter_type,
                    "{}",
                    s.get("sample_size", 100),
                    s.get("win_rate", 0.50),
                    s.get("expectancy_r", 0.30),
                    s.get("sharpe_ratio", 0.25),
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
                    sid,
                    sid,
                    instrument,
                    orb_label,
                    orb_minutes,
                    rr_target,
                    confirm_bars,
                    entry_model,
                    filter_type,
                    "{}",
                    s.get("sample_size", 100),
                    s.get("win_rate", 0.50),
                    s.get("expectancy_r", 0.30),
                    s.get("years_tested", 3),
                    s.get("all_years_positive", True),
                    s.get("stress_test_passed", True),
                    s.get("sharpe_ratio", 0.25),
                    s.get("max_drawdown_r", 5.0),
                    s.get("yearly_results", "{}"),
                    s.get("status", "active"),
                ],
            )

    # Insert daily_features FIRST (FK parent for orb_outcomes).
    # daily_features rows come from BOTH the `features` arg AND every outcome's
    # day (the FK parent each outcome needs). Build one combined frame, dedup on
    # the PK (trading_day, symbol, orb_minutes) to honour INSERT OR IGNORE
    # semantics, then bulk-load via a single registered-frame INSERT...SELECT.
    # Per-row con.execute against a file-backed DuckDB costs ~0.4s/row (PK
    # conflict probe + fsync); the bulk path is ~340x faster (54s -> 0.16s).
    feature_cols = [
        "trading_day",
        "symbol",
        "orb_minutes",
        "bar_count_1m",
        "orb_CME_REOPEN_high",
        "orb_CME_REOPEN_low",
        "orb_CME_REOPEN_size",
        "orb_CME_REOPEN_break_dir",
    ]
    feature_rows = []
    for f in features or []:
        feature_rows.append(
            [
                f["trading_day"],
                f.get("symbol", "MGC"),
                f.get("orb_minutes", 5),
                f.get("bar_count_1m", 1400),
                f.get("orb_CME_REOPEN_high", 2355.0),
                f.get("orb_CME_REOPEN_low", 2345.0),
                f.get("orb_CME_REOPEN_size", 10.0),
                f.get("orb_CME_REOPEN_break_dir", "long"),
            ]
        )
    for o in outcomes or []:
        # Each outcome needs a daily_features FK parent (was the per-outcome
        # INSERT OR IGNORE). Same defaults as the original inline insert.
        feature_rows.append(
            [
                o["trading_day"],
                o.get("symbol", "MGC"),
                o.get("orb_minutes", 5),
                1400,
                2355.0,
                2345.0,
                10.0,
                "long",
            ]
        )
    if feature_rows:
        feat_df = pd.DataFrame(feature_rows, columns=feature_cols).drop_duplicates(
            subset=["trading_day", "symbol", "orb_minutes"], keep="first"
        )
        con.register("_seed_features", feat_df)
        con.execute(
            f"INSERT INTO daily_features ({', '.join(feature_cols)}) "
            f"SELECT {', '.join(feature_cols)} FROM _seed_features"
        )
        con.unregister("_seed_features")

    # Insert orb_outcomes (after daily_features due to FK) via the same bulk path.
    if outcomes:
        outcome_cols = [
            "trading_day",
            "symbol",
            "orb_label",
            "orb_minutes",
            "rr_target",
            "confirm_bars",
            "entry_model",
            "entry_price",
            "stop_price",
            "target_price",
            "outcome",
            "pnl_r",
            "mae_r",
            "mfe_r",
        ]
        outcome_rows = [
            [
                o["trading_day"],
                o.get("symbol", "MGC"),
                o.get("orb_label", "CME_REOPEN"),
                o.get("orb_minutes", 5),
                o.get("rr_target", 2.0),
                o.get("confirm_bars", 2),
                o.get("entry_model", "E1"),
                o.get("entry_price", 2350.0),
                o.get("stop_price", 2340.0),
                o.get("target_price", 2370.0),
                o["outcome"],
                o["pnl_r"],
                o.get("mae_r", -0.5),
                o.get("mfe_r", 1.5),
            ]
            for o in outcomes
        ]
        out_df = pd.DataFrame(outcome_rows, columns=outcome_cols)
        con.register("_seed_outcomes", out_df)
        con.execute(
            f"INSERT INTO orb_outcomes ({', '.join(outcome_cols)}) SELECT {', '.join(outcome_cols)} FROM _seed_outcomes"
        )
        con.unregister("_seed_outcomes")

    con.commit()
    con.close()
    return db_path


def _make_outcomes(
    start_year=2023, end_year=2025, trades_per_year=40, win_rate=0.50, win_pnl=1.8, loss_pnl=-1.0, **overrides
):
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


def _make_features(start_year=2023, end_year=2025, trades_per_year=40, orb_size=10.0, **overrides):
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
        status, notes = classify_fitness(rolling_exp_r=0.30, rolling_sample=20, recent_sharpe_30=0.15)
        assert status == "FIT"

    def test_classify_fitness_watch_declining_sharpe(self):
        """Positive rolling ExpR but declining recent Sharpe -> WATCH."""
        status, notes = classify_fitness(rolling_exp_r=0.20, rolling_sample=20, recent_sharpe_30=-0.15)
        assert status == "WATCH"
        assert "Declining" in notes

    def test_classify_fitness_watch_thin_data(self):
        """Positive rolling ExpR but only 10-14 trades -> WATCH."""
        status, notes = classify_fitness(rolling_exp_r=0.20, rolling_sample=12, recent_sharpe_30=0.10)
        assert status == "WATCH"
        assert "Thin data" in notes

    def test_classify_fitness_decay(self):
        """Negative rolling ExpR -> DECAY."""
        status, notes = classify_fitness(rolling_exp_r=-0.10, rolling_sample=25, recent_sharpe_30=-0.20)
        assert status == "DECAY"
        assert "Negative" in notes

    def test_classify_fitness_stale(self):
        """< 10 rolling trades -> STALE."""
        status, notes = classify_fitness(rolling_exp_r=0.50, rolling_sample=5, recent_sharpe_30=0.30)
        assert status == "STALE"

    def test_classify_fitness_stale_none_exp(self):
        """None rolling ExpR with enough trades -> STALE."""
        status, notes = classify_fitness(rolling_exp_r=None, rolling_sample=15, recent_sharpe_30=0.10)
        assert status == "STALE"

    def test_classify_fitness_fit_none_sharpe(self):
        """Positive rolling ExpR, None recent sharpe (not enough for 30-trade) -> FIT."""
        status, notes = classify_fitness(rolling_exp_r=0.25, rolling_sample=20, recent_sharpe_30=None)
        assert status == "FIT"

    # --- Boundary tests (mutation-hardening) ---------------------------------
    # The fitness verdict (FIT/WATCH/DECAY/STALE) decides what is deployable, so
    # every threshold comparison must be pinned ON its boundary, not just at
    # interior values. Mutation testing (2026-06-07) showed the `<`/`<=` operators
    # on lines 119/125/129/132 of strategy_fitness.py survived because no test sat
    # exactly on the boundary each mutant moves. These tests kill those mutants by
    # asserting behavior at the exact threshold AND one step below it. Constants
    # are imported (not inlined) so a future threshold change updates intent here.

    def test_classify_fitness_stale_watch_boundary(self):
        """rolling_sample exactly at MIN_ROLLING_WATCH -> NOT STALE (kills L119 < -> <=).

        At sample == MIN_ROLLING_WATCH the `<` makes it past the STALE gate; a
        `<=` mutant would wrongly return STALE. One below must still be STALE.
        """
        at_watch, _ = classify_fitness(rolling_exp_r=0.20, rolling_sample=MIN_ROLLING_WATCH, recent_sharpe_30=0.10)
        assert at_watch != "STALE", "sample==MIN_ROLLING_WATCH must clear the STALE gate"

        below, _ = classify_fitness(rolling_exp_r=0.20, rolling_sample=MIN_ROLLING_WATCH - 1, recent_sharpe_30=0.10)
        assert below == "STALE", "sample below MIN_ROLLING_WATCH must be STALE"

    def test_classify_fitness_decay_zero_expectancy_boundary(self):
        """rolling_exp_r exactly 0.0 -> DECAY (kills L125 <= -> <).

        `rolling_exp_r <= 0` flags zero-expectancy as DECAY. A `<` mutant would let
        exactly-zero ExpR pass through to WATCH/FIT — a strategy earning nothing
        would not be flagged. Pin the zero boundary; a tiny positive must NOT decay.
        """
        at_zero, notes = classify_fitness(rolling_exp_r=0.0, rolling_sample=MIN_ROLLING_FIT + 5, recent_sharpe_30=0.10)
        assert at_zero == "DECAY", "exactly-zero rolling ExpR must be DECAY"
        assert "Negative" in notes

        just_positive, _ = classify_fitness(
            rolling_exp_r=0.0001, rolling_sample=MIN_ROLLING_FIT + 5, recent_sharpe_30=0.10
        )
        assert just_positive != "DECAY", "tiny positive ExpR must not be DECAY"

    def test_classify_fitness_watch_fit_sample_boundary(self):
        """rolling_sample exactly at MIN_ROLLING_FIT -> FIT, one below -> WATCH (kills L129 < -> <=).

        The FIT gate requires sample >= MIN_ROLLING_FIT. At exactly the threshold
        the trade is FIT-eligible; a `<=` mutant would push it to WATCH (thin data).
        """
        at_fit, _ = classify_fitness(rolling_exp_r=0.20, rolling_sample=MIN_ROLLING_FIT, recent_sharpe_30=0.10)
        assert at_fit == "FIT", "sample==MIN_ROLLING_FIT with stable Sharpe must be FIT"

        below_fit, notes = classify_fitness(
            rolling_exp_r=0.20, rolling_sample=MIN_ROLLING_FIT - 1, recent_sharpe_30=0.10
        )
        assert below_fit == "WATCH", "sample one below MIN_ROLLING_FIT must be WATCH (thin)"
        assert "Thin data" in notes

    def test_classify_fitness_sharpe_decay_threshold_boundary(self):
        """recent_sharpe exactly at SHARPE_DECAY_THRESHOLD -> WATCH (kills L132 <= -> <).

        `recent_sharpe_30 <= SHARPE_DECAY_THRESHOLD` flags a declining Sharpe as
        WATCH. At exactly the threshold (-0.1) the strategy is declining -> WATCH;
        a `<` mutant would let exactly -0.1 stay FIT. Just above the threshold = FIT.
        """
        at_threshold, notes = classify_fitness(
            rolling_exp_r=0.20,
            rolling_sample=MIN_ROLLING_FIT + 5,
            recent_sharpe_30=SHARPE_DECAY_THRESHOLD,
        )
        assert at_threshold == "WATCH", "Sharpe exactly at decay threshold must be WATCH"
        assert "Declining" in notes

        just_above, _ = classify_fitness(
            rolling_exp_r=0.20,
            rolling_sample=MIN_ROLLING_FIT + 5,
            recent_sharpe_30=SHARPE_DECAY_THRESHOLD + 0.01,
        )
        assert just_above == "FIT", "Sharpe just above decay threshold must be FIT"


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
            start_year=2024, end_year=2025, trades_per_year=30, win_rate=0.55, win_pnl=1.8, loss_pnl=-1.0
        )
        result = _recent_trade_sharpe(outcomes, 30)
        assert result is not None
        assert isinstance(result, float)

    def test_recent_trade_sharpe_insufficient(self):
        """< 30 trades -> None."""
        outcomes = _make_outcomes(start_year=2025, end_year=2025, trades_per_year=10)
        result = _recent_trade_sharpe(outcomes, 30)
        assert result is None

    def test_recent_trade_sharpe_uses_last_n(self):
        """Sharpe uses only last N trades, not all."""
        # 50 good trades followed by 10 bad ones
        good = [{"trading_day": date(2024, 1, i + 1), "outcome": "win", "pnl_r": 2.0} for i in range(20)]
        bad = [{"trading_day": date(2025, 1, i + 1), "outcome": "loss", "pnl_r": -1.0} for i in range(20)]
        all_outcomes = good + bad
        # Last 20 trades = all losses -> negative Sharpe
        sharpe_20 = _recent_trade_sharpe(all_outcomes, 20)
        assert sharpe_20 is None  # all same pnl -> std=0 -> None

    # --- Value tests (mutation-hardening) ------------------------------------
    # The existing tests above assert only is-None / is-not-None / isinstance —
    # they verify the function RETURNS something, never that the number is RIGHT.
    # Mutation testing (2026-06-07) showed 27 arithmetic mutants on the Sharpe
    # formula (lines 157-164: mean div, variance (r-mean)**2/(n-1), sqrt, final
    # mean/std) all SURVIVED for exactly this reason. These tests pin the EXACT
    # Sharpe value of a hand-computed series, so any operator/constant mutation in
    # the formula changes the number and fails the assert.

    def test_recent_trade_sharpe_exact_value(self):
        """Hand-computed Sharpe for a known series kills the arithmetic mutants.

        Series pnl_r = [2.0, -1.0, 2.0, -1.0] (4 trades):
          mean      = 0.5
          variance  = sum((r-0.5)^2)/(4-1) = (2.25+2.25+2.25+2.25)/3 = 3.0   [sample, n-1]
          std       = sqrt(3.0) = 1.7320508...
          sharpe    = mean/std = 0.5/1.7320508 = 0.2886751...
        Mutating n-1->n, (r-mean)->(r+mean), **2->**0, /->*, sqrt power, or the
        final mean/std all shift this number away from 0.2886751.
        """
        outcomes = [
            {"trading_day": date(2025, 1, 1), "outcome": "win", "pnl_r": 2.0},
            {"trading_day": date(2025, 1, 2), "outcome": "loss", "pnl_r": -1.0},
            {"trading_day": date(2025, 1, 3), "outcome": "win", "pnl_r": 2.0},
            {"trading_day": date(2025, 1, 4), "outcome": "loss", "pnl_r": -1.0},
        ]
        result = _recent_trade_sharpe(outcomes, 4)
        assert result == pytest.approx(0.2886751, abs=1e-6)

    def test_recent_trade_sharpe_negative_value(self):
        """A losing-skewed series yields a specific NEGATIVE Sharpe (sign + magnitude).

        Series pnl_r = [-2.0, 1.0, -2.0, 1.0]:
          mean = -0.5, variance = 3.0, std = sqrt(3), sharpe = -0.2886751.
        Pins the sign so a mutant flipping a subtraction or the final division
        (which could flip sign or magnitude) is caught.
        """
        outcomes = [
            {"trading_day": date(2025, 2, 1), "outcome": "loss", "pnl_r": -2.0},
            {"trading_day": date(2025, 2, 2), "outcome": "win", "pnl_r": 1.0},
            {"trading_day": date(2025, 2, 3), "outcome": "loss", "pnl_r": -2.0},
            {"trading_day": date(2025, 2, 4), "outcome": "win", "pnl_r": 1.0},
        ]
        result = _recent_trade_sharpe(outcomes, 4)
        assert result == pytest.approx(-0.2886751, abs=1e-6)

    def test_recent_trade_sharpe_exact_n_boundary(self):
        """len(traded) exactly == n_trades returns a value (kills L148 < -> <=).

        `if len(traded) < n_trades: return None` — at exactly n_trades the function
        must proceed and return a number. A `<=` mutant would wrongly return None.
        """
        outcomes = [
            {"trading_day": date(2025, 3, d), "outcome": "win" if d % 2 else "loss", "pnl_r": 1.5 if d % 2 else -1.0}
            for d in range(1, 4)  # exactly 3 trades
        ]
        result = _recent_trade_sharpe(outcomes, 3)
        assert result is not None, "exactly n_trades must produce a Sharpe, not None"

    def test_recent_trade_sharpe_min_two_boundary(self):
        """Exactly 1 trade after filter -> None (kills L154 < 2 boundary mutants).

        `if len(r_values) < 2: return None` guards variance (needs >=2 points). With
        a single trade, variance is undefined -> None. NumberReplacer on the `2` or
        `<`->`<=`/`==` mutants change this boundary; one trade must return None.
        """
        outcomes = [{"trading_day": date(2025, 4, 1), "outcome": "win", "pnl_r": 1.0}]
        assert _recent_trade_sharpe(outcomes, 1) is None


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
        strategy_id = "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER"
        strategies = [
            {
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
            }
        ]
        outcomes = _make_outcomes(
            start_year=2023,
            end_year=2025,
            trades_per_year=40,
            win_rate=0.50,
            win_pnl=1.8,
            loss_pnl=-1.0,
        )
        features = _make_features(start_year=2023, end_year=2025, trades_per_year=40)

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes, features)

        score = compute_fitness(
            strategy_id,
            db_path=db_path,
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
        strategy_id = "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER"
        strategies = [
            {
                "strategy_id": strategy_id,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            }
        ]
        # 30 wins in 2023, then 30 losses in 2025
        early_wins = [
            {"trading_day": date(2023, (i % 11) + 1, (i % 27) + 1), "outcome": "win", "pnl_r": 2.0} for i in range(30)
        ]
        late_losses = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1), "outcome": "loss", "pnl_r": -1.0} for i in range(30)
        ]

        db_path = _setup_fitness_db(
            tmp_path,
            strategies,
            early_wins + late_losses,
        )

        # as_of=2023-12-31: should only see the 30 wins, NOT the 2025 losses
        score = compute_fitness(
            strategy_id,
            db_path=db_path,
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
        strategy_id = "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER"
        strategies = [
            {
                "strategy_id": strategy_id,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            }
        ]
        # Only 12 outcomes, all in the rolling window, mostly wins
        outcomes = [
            {
                "trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
                "outcome": "win" if i < 8 else "loss",
                "pnl_r": 2.0 if i < 8 else -1.0,
            }
            for i in range(12)
        ]

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes)

        score = compute_fitness(
            strategy_id,
            db_path=db_path,
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
        strategy_id = "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4"
        strategies = [
            {
                "strategy_id": strategy_id,
                "filter_type": "ORB_G4",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            }
        ]
        # 20 outcomes on big-ORB days (size=10), 20 on small-ORB days (size=2)
        big_outcomes = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1), "outcome": "win", "pnl_r": 2.0} for i in range(20)
        ]
        small_outcomes = [
            {"trading_day": date(2024, (i % 11) + 1, (i % 27) + 1), "outcome": "loss", "pnl_r": -1.0} for i in range(20)
        ]
        # Features: 2025 days have big ORBs, 2024 days have small ORBs
        big_features = [
            {"trading_day": date(2025, (i % 11) + 1, (i % 27) + 1), "orb_CME_REOPEN_size": 10.0} for i in range(20)
        ]
        small_features = [
            {"trading_day": date(2024, (i % 11) + 1, (i % 27) + 1), "orb_CME_REOPEN_size": 2.0} for i in range(20)
        ]

        db_path = _setup_fitness_db(
            tmp_path,
            strategies,
            big_outcomes + small_outcomes,
            big_features + small_features,
        )

        score = compute_fitness(
            strategy_id,
            db_path=db_path,
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        # Only the 20 big-ORB outcomes should pass filter
        # The 20 small-ORB (size=2.0 < 4.0) should be excluded
        assert score.rolling_sample == 20


class TestComputeFitnessFromCache:
    """Direct-call mutation tests for `_compute_fitness_from_cache`.

    Targets the cache-path survivor cluster (largest remaining in this module):
    fail-closed unknown filter, the eligible-day filter branch, the
    `min_rolling_trades` None-blanking boundary, and the delta_30/60 subtraction.
    Each assertion is written to KILL a specific mutant — asserts on computed
    VALUES (rolling_sample, rolling_exp_r, deltas), not just "returns a score".
    """

    # All these strategies share one canonical 5-tuple cache key.
    _KEY = ("CME_REOPEN", 5, "E1", 2.0, 2)

    def _params(self, filter_type="NO_FILTER", stop_multiplier=1.0, sharpe_ratio=0.25):
        return {
            "orb_label": "CME_REOPEN",
            "orb_minutes": 5,
            "entry_model": "E1",
            "rr_target": 2.0,
            "confirm_bars": 2,
            "filter_type": filter_type,
            "instrument": "MGC",
            "expectancy_r": 0.30,
            "sharpe_ratio": sharpe_ratio,
            "sample_size": 120,
            "stop_multiplier": stop_multiplier,
        }

    def _outcomes(self, n, *, win, pnl_r, year=2025, month_base=1):
        """N outcomes on distinct days inside the rolling window (year 2025)."""
        return [
            {
                "trading_day": date(year, ((month_base + i) % 11) + 1, (i % 27) + 1),
                "outcome": "win" if win else "loss",
                "pnl_r": pnl_r,
            }
            for i in range(n)
        ]

    def test_known_filter_keeps_only_eligible_days(self, tmp_path):
        """ORB_G4 eligibility branch (sf.py:585-593): only days whose feature
        passes filt.matches_row survive. ASYMMETRIC fixture (20 eligible wins vs
        8 ineligible losses) + value assertion so an `in -> not in` membership
        flip is observable: the flip would keep 8 losing days, not 20 winning
        ones. Kills the membership flip AND the matches_row call mutants."""
        big_days = self._outcomes(20, win=True, pnl_r=2.0, year=2025, month_base=0)
        small_days = self._outcomes(8, win=False, pnl_r=-1.0, year=2024, month_base=5)
        outcome_cache = {self._KEY: big_days + small_days}

        # Feature cache: 2025 days have big ORB (size=10 -> passes ORB_G4),
        # 2024 days have small ORB (size=2 -> excluded by ORB_G4 >= 4.0).
        feature_cache = {}
        for o in big_days:
            feature_cache[(o["trading_day"], 5)] = {"orb_CME_REOPEN_size": 10.0}
        for o in small_days:
            feature_cache[(o["trading_day"], 5)] = {"orb_CME_REOPEN_size": 2.0}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_ORB_G4",
            self._params(filter_type="ORB_G4"),
            outcome_cache,
            feature_cache,
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        # Only the 20 big-ORB winning days pass; the 8 small-ORB losing days are
        # excluded. Asserting BOTH count and ExpR makes the membership flip
        # observable (flip -> sample 8, ExpR -1.0).
        assert score.rolling_sample == 20
        assert score.rolling_exp_r == 2.0

    def test_unknown_filter_fails_closed_to_zero(self, tmp_path):
        """Fail-closed unknown filter (sf.py:581-584): an unregistered
        filter_type yields ZERO outcomes, never all-pass. Kills the mutant that
        deletes `all_outcomes = []`."""
        outcome_cache = {self._KEY: self._outcomes(30, win=True, pnl_r=2.0)}
        feature_cache = {}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_BOGUS_FILTER",
            self._params(filter_type="DEFINITELY_NOT_A_REAL_FILTER"),
            outcome_cache,
            feature_cache,
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        # Fail-closed: unknown filter drops ALL outcomes.
        assert score.rolling_sample == 0

    def test_no_filter_keeps_all_outcomes(self, tmp_path):
        """NO_FILTER path bypasses the eligibility branch entirely — every
        outcome in the window counts. Anchors the branch the fail-closed test
        contrasts against."""
        outcome_cache = {self._KEY: self._outcomes(30, win=True, pnl_r=2.0)}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
            self._params(filter_type="NO_FILTER"),
            outcome_cache,
            {},
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        assert score.rolling_sample == 30

    def test_rolling_exp_r_blanked_below_min_trades(self):
        """min_rolling_trades None-blanking (sf.py:626-629): with sample <
        min_rolling_trades, rolling_exp_r/sharpe/wr are blanked to None while
        rolling_sample is preserved. Kills `<` -> `<=` and delete-blanking."""
        # Exactly MIN_ROLLING_FIT - 1 = 14 outcomes -> below the FIT floor.
        n = MIN_ROLLING_FIT - 1
        outcome_cache = {self._KEY: self._outcomes(n, win=True, pnl_r=2.0)}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
            self._params(),
            outcome_cache,
            {},
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
            min_rolling_trades=MIN_ROLLING_FIT,
        )
        assert score.rolling_sample == n
        assert score.rolling_exp_r is None
        assert score.rolling_sharpe is None
        assert score.rolling_win_rate is None

    def test_rolling_exp_r_kept_at_min_trades_boundary(self):
        """Boundary partner: with sample == min_rolling_trades, rolling_exp_r is
        NOT blanked (the `<` is strict). Kills the `<` -> `<=` mutant which would
        wrongly blank at the boundary."""
        n = MIN_ROLLING_FIT  # exactly at the floor
        outcome_cache = {self._KEY: self._outcomes(n, win=True, pnl_r=2.0)}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
            self._params(),
            outcome_cache,
            {},
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
            min_rolling_trades=MIN_ROLLING_FIT,
        )
        assert score.rolling_sample == n
        # All wins at +2.0 -> ExpR == 2.0, NOT None.
        assert score.rolling_exp_r is not None
        assert score.rolling_exp_r == 2.0

    def test_sharpe_delta_is_recent_minus_full(self):
        """delta_30/60 subtraction (sf.py:616-621): sharpe_delta_30 == recent_30
        - full_sharpe. Kills `-` -> `+` and operand-swap mutants."""
        # 35 mixed outcomes so _recent_trade_sharpe(30) is computable (needs >=30
        # traded and >=2 distinct values for a finite Sharpe).
        outcomes = []
        for i in range(35):
            win = i % 2 == 0
            outcomes.append(
                {
                    "trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
                    "outcome": "win" if win else "loss",
                    "pnl_r": 2.0 if win else -1.0,
                }
            )
        outcome_cache = {self._KEY: outcomes}
        full_sharpe = 0.25

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
            self._params(sharpe_ratio=full_sharpe),
            outcome_cache,
            {},
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        assert score.recent_sharpe_30 is not None
        # Exact identity: delta == recent - full.
        assert score.sharpe_delta_30 == score.recent_sharpe_30 - full_sharpe

    def test_sharpe_delta_none_when_full_sharpe_none(self):
        """delta guard (sf.py:618/620): when full_sharpe is None, deltas stay
        None even if recent_30 is computable. Kills the guard-flip mutant."""
        outcomes = []
        for i in range(35):
            win = i % 2 == 0
            outcomes.append(
                {
                    "trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
                    "outcome": "win" if win else "loss",
                    "pnl_r": 2.0 if win else -1.0,
                }
            )
        params = self._params()
        params["sharpe_ratio"] = None  # full_sharpe None -> deltas must be None
        outcome_cache = {self._KEY: outcomes}

        score = _compute_fitness_from_cache(
            "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
            params,
            outcome_cache,
            {},
            as_of_date=date(2025, 12, 31),
            rolling_months=24,
        )
        assert score.recent_sharpe_30 is not None
        assert score.sharpe_delta_30 is None
        assert score.sharpe_delta_60 is None


class TestFitnessReportSummary:
    def test_fitness_report_summary_counts(self, tmp_path):
        """Summary correctly counts {fit: N, watch: N, decay: N, stale: N}."""
        # Create 3 strategies with different expected outcomes
        strategies = [
            {
                "strategy_id": f"MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER_{i}",
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
            start_year=2024,
            end_year=2025,
            trades_per_year=30,
            win_rate=0.55,
            win_pnl=1.8,
            loss_pnl=-1.0,
        )
        features = _make_features(
            start_year=2024,
            end_year=2025,
            trades_per_year=30,
        )

        db_path = _setup_fitness_db(tmp_path, strategies, outcomes, features)

        report = compute_portfolio_fitness(
            db_path=db_path,
            instrument="MGC",
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
                strategy_id="S1",
                instrument="MGC",
                orb_label="CME_REOPEN",
                entry_model="E1",
                rr_target=2.0,
                confirm_bars=2,
                filter_type="NO_FILTER",
                expectancy_r=0.30,
                win_rate=0.50,
                sample_size=100,
                sharpe_ratio=0.25,
                max_drawdown_r=5.0,
                median_risk_points=10.0,
                weight=1.0,
            ),
            PortfolioStrategy(
                strategy_id="S2",
                instrument="MGC",
                orb_label="TOKYO_OPEN",
                entry_model="E1",
                rr_target=2.5,
                confirm_bars=2,
                filter_type="ORB_G4",
                expectancy_r=0.40,
                win_rate=0.55,
                sample_size=80,
                sharpe_ratio=0.30,
                max_drawdown_r=4.0,
                median_risk_points=12.0,
                weight=1.0,
            ),
            PortfolioStrategy(
                strategy_id="S3",
                instrument="MGC",
                orb_label="LONDON_METALS",
                entry_model="E3",
                rr_target=2.0,
                confirm_bars=5,
                filter_type="ORB_G5",
                expectancy_r=0.35,
                win_rate=0.48,
                sample_size=75,
                sharpe_ratio=0.20,
                max_drawdown_r=3.0,
                median_risk_points=8.0,
                weight=1.0,
            ),
        ]

        portfolio = Portfolio(
            name="test",
            instrument="MGC",
            strategies=strats,
            account_equity=25000.0,
            risk_per_trade_pct=2.0,
            max_concurrent_positions=3,
            max_daily_loss_r=5.0,
        )

        scores = [
            FitnessScore(
                strategy_id="S1",
                full_period_exp_r=0.30,
                full_period_sharpe=0.25,
                full_period_sample=100,
                rolling_exp_r=0.25,
                rolling_sharpe=0.20,
                rolling_win_rate=0.48,
                rolling_sample=25,
                rolling_window_months=18,
                recent_sharpe_30=0.15,
                recent_sharpe_60=0.18,
                sharpe_delta_30=-0.10,
                sharpe_delta_60=-0.07,
                fitness_status="FIT",
                fitness_notes="OK",
            ),
            FitnessScore(
                strategy_id="S2",
                full_period_exp_r=0.40,
                full_period_sharpe=0.30,
                full_period_sample=80,
                rolling_exp_r=0.15,
                rolling_sharpe=0.10,
                rolling_win_rate=0.45,
                rolling_sample=18,
                rolling_window_months=18,
                recent_sharpe_30=-0.15,
                recent_sharpe_60=0.05,
                sharpe_delta_30=-0.45,
                sharpe_delta_60=-0.25,
                fitness_status="WATCH",
                fitness_notes="Declining",
            ),
            FitnessScore(
                strategy_id="S3",
                full_period_exp_r=0.35,
                full_period_sharpe=0.20,
                full_period_sample=75,
                rolling_exp_r=-0.10,
                rolling_sharpe=-0.05,
                rolling_win_rate=0.40,
                rolling_sample=20,
                rolling_window_months=18,
                recent_sharpe_30=-0.30,
                recent_sharpe_60=-0.20,
                sharpe_delta_30=-0.50,
                sharpe_delta_60=-0.40,
                fitness_status="DECAY",
                fitness_notes="Negative",
            ),
        ]

        report = FitnessReport(
            as_of_date=date(2025, 12, 31),
            scores=scores,
            summary={"fit": 1, "watch": 1, "decay": 1, "stale": 0},
        )

        adjusted = fitness_weighted_portfolio(portfolio, report)

        # Verify it returns a new Portfolio (not modified input)
        assert adjusted is not portfolio
        assert len(adjusted.strategies) == 3

        weight_map = {s.strategy_id: s.weight for s in adjusted.strategies}
        assert weight_map["S1"] == 1.0  # FIT
        assert weight_map["S2"] == 0.5  # WATCH
        assert weight_map["S3"] == 0.0  # DECAY

        # Original unchanged
        assert all(s.weight == 1.0 for s in portfolio.strategies)

    def test_regime_classification_caps_fitness(self):
        """REGIME (weight=0.5) with FIT status stays at 0.5, not 1.0."""
        from trading_app.portfolio import Portfolio, PortfolioStrategy, fitness_weighted_portfolio

        strats = [
            PortfolioStrategy(
                strategy_id="REGIME_FIT",
                instrument="MGC",
                orb_label="CME_REOPEN",
                entry_model="E1",
                rr_target=2.0,
                confirm_bars=1,
                filter_type="ORB_G5",
                expectancy_r=0.15,
                win_rate=0.45,
                sample_size=60,
                sharpe_ratio=0.20,
                max_drawdown_r=3.0,
                median_risk_points=4.0,
                weight=0.5,  # REGIME classification weight
            ),
            PortfolioStrategy(
                strategy_id="CORE_FIT",
                instrument="MGC",
                orb_label="TOKYO_OPEN",
                entry_model="E1",
                rr_target=2.0,
                confirm_bars=1,
                filter_type="ORB_G5",
                expectancy_r=0.20,
                win_rate=0.48,
                sample_size=200,
                sharpe_ratio=0.25,
                max_drawdown_r=2.5,
                median_risk_points=4.0,
                weight=1.0,  # CORE classification weight
            ),
        ]
        portfolio = Portfolio(
            name="test",
            instrument="MGC",
            strategies=strats,
            account_equity=25000.0,
            risk_per_trade_pct=2.0,
            max_concurrent_positions=3,
            max_daily_loss_r=5.0,
        )

        scores = [
            FitnessScore(
                strategy_id="REGIME_FIT",
                full_period_exp_r=0.30,
                full_period_sharpe=0.25,
                full_period_sample=60,
                rolling_exp_r=0.25,
                rolling_sharpe=0.20,
                rolling_win_rate=0.48,
                rolling_sample=25,
                rolling_window_months=18,
                recent_sharpe_30=0.15,
                recent_sharpe_60=0.18,
                sharpe_delta_30=0.0,
                sharpe_delta_60=0.0,
                fitness_status="FIT",
                fitness_notes="",
            ),
            FitnessScore(
                strategy_id="CORE_FIT",
                full_period_exp_r=0.35,
                full_period_sharpe=0.30,
                full_period_sample=200,
                rolling_exp_r=0.28,
                rolling_sharpe=0.22,
                rolling_win_rate=0.50,
                rolling_sample=30,
                rolling_window_months=18,
                recent_sharpe_30=0.20,
                recent_sharpe_60=0.22,
                sharpe_delta_30=0.0,
                sharpe_delta_60=0.0,
                fitness_status="FIT",
                fitness_notes="",
            ),
        ]
        report = FitnessReport(
            as_of_date=date(2025, 12, 31),
            scores=scores,
            summary={"fit": 2, "watch": 0, "decay": 0, "stale": 0},
        )

        adjusted = fitness_weighted_portfolio(portfolio, report)
        weight_map = {s.strategy_id: s.weight for s in adjusted.strategies}

        # REGIME + FIT: min(1.0, 0.5) = 0.5 (classification caps fitness)
        assert weight_map["REGIME_FIT"] == 0.5
        # CORE + FIT: min(1.0, 1.0) = 1.0
        assert weight_map["CORE_FIT"] == 1.0


# =========================================================================
# Decay diagnostic tests
# =========================================================================


def _setup_family_db(tmp_path, strategies, outcomes, family_hash="fam_abc", family_size=None, robustness="ROBUST"):
    """Create DB with strategies linked to an edge family."""
    db_path = _setup_fitness_db(tmp_path, strategies, outcomes)
    con = duckdb.connect(str(db_path))

    n_members = family_size or len(strategies)
    head_sid = strategies[0]["strategy_id"]

    # Create edge_families row
    con.execute(
        """
        INSERT INTO edge_families
        (family_hash, instrument, member_count, trade_day_count,
         head_strategy_id, head_expectancy_r, head_sharpe_ann,
         robustness_status, cv_expectancy, median_expectancy_r,
         avg_sharpe_ann, min_member_trades, trade_tier)
        VALUES (?, 'MGC', ?, 100, ?, 0.30, 1.0, ?, 0.15, 0.28, 1.0, 80, 'CORE')
    """,
        [family_hash, n_members, head_sid, robustness],
    )

    # Link strategies to family
    for s in strategies:
        is_head = s["strategy_id"] == head_sid
        con.execute(
            """
            UPDATE validated_setups
            SET family_hash = ?, is_family_head = ?
            WHERE strategy_id = ?
        """,
            [family_hash, is_head, s["strategy_id"]],
        )

    con.commit()
    con.close()
    return db_path


class TestDecayDiagnostics:
    def test_regime_shift_all_siblings_decay(self, tmp_path):
        """When all siblings are also DECAY -> REGIME_SHIFT."""
        # 3 strategies, all with negative recent outcomes -> all DECAY
        strategies = [
            {
                "strategy_id": f"MGC_CME_REOPEN_E1_RR{rr}_CB2_NO_FILTER",
                "rr_target": rr,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            }
            for rr in [1.5, 2.0, 2.5]
        ]
        # All strategies share same NO_FILTER + same EM/CB -> same outcomes
        # Make all outcomes losses in recent window
        outcomes = [
            {
                "trading_day": date(2025, (i % 11) + 1, (i % 27) + 1),
                "outcome": "loss",
                "pnl_r": -1.0,
                "rr_target": rr,
                "confirm_bars": 2,
            }
            for rr in [1.5, 2.0, 2.5]
            for i in range(20)
        ]
        db_path = _setup_family_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                strategies[0]["strategy_id"],
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "REGIME_SHIFT"
        assert diag.family_size == 3

    def test_overfit_siblings_fit(self, tmp_path):
        """When siblings are FIT but target is DECAY -> OVERFIT."""
        # Strategy at RR2.0 decays, but RR1.5 and RR2.5 are fine
        strategies = [
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                "rr_target": 2.0,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR1.5_CB2_NO_FILTER",
                "rr_target": 1.5,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.5_CB2_NO_FILTER",
                "rr_target": 2.5,
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
        ]
        # RR2.0 gets losses, RR1.5/2.5 get wins
        outcomes = []
        for i in range(20):
            td = date(2025, (i % 11) + 1, (i % 27) + 1)
            # RR2.0 = loss
            outcomes.append({"trading_day": td, "outcome": "loss", "pnl_r": -1.0, "rr_target": 2.0, "confirm_bars": 2})
            # RR1.5 = win
            outcomes.append({"trading_day": td, "outcome": "win", "pnl_r": 1.5, "rr_target": 1.5, "confirm_bars": 2})
            # RR2.5 = win
            outcomes.append({"trading_day": td, "outcome": "win", "pnl_r": 2.5, "rr_target": 2.5, "confirm_bars": 2})

        db_path = _setup_family_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "OVERFIT"
        assert diag.siblings_fit == 2
        assert diag.siblings_decay == 0

    def test_singleton_no_peers(self, tmp_path):
        """Single-member family -> SINGLETON."""
        strategies = [
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
        ]
        outcomes = _make_outcomes(
            start_year=2024,
            end_year=2025,
            trades_per_year=20,
            win_rate=0.40,
            win_pnl=1.5,
            loss_pnl=-1.0,
        )
        db_path = _setup_family_db(tmp_path, strategies, outcomes, family_size=1, robustness="SINGLETON")

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                strategies[0]["strategy_id"],
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "SINGLETON"

    def test_no_family_hash(self, tmp_path):
        """Strategy with no family_hash -> NO_FAMILY."""
        strategies = [
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
        ]
        outcomes = _make_outcomes(start_year=2024, end_year=2025, trades_per_year=20)
        db_path = _setup_fitness_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                strategies[0]["strategy_id"],
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "NO_FAMILY"

    def test_decay_diagnosis_dataclass_fields(self, tmp_path):
        """DecayDiagnosis has all expected fields."""
        strategies = [
            {
                "strategy_id": "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                "filter_type": "NO_FILTER",
                "sample_size": 100,
                "expectancy_r": 0.30,
                "sharpe_ratio": 0.25,
            },
        ]
        outcomes = _make_outcomes(start_year=2024, end_year=2025, trades_per_year=20)
        db_path = _setup_fitness_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                strategies[0]["strategy_id"],
                as_of_date=date(2025, 12, 31),
            )
        finally:
            con.close()

        assert isinstance(diag, DecayDiagnosis)
        assert diag.strategy_id == strategies[0]["strategy_id"]
        assert diag.diagnosis in ("REGIME_SHIFT", "OVERFIT", "FRAGMENTED", "SINGLETON", "NO_FAMILY")

    # --- Tier-2 mutation-killing tests: diagnose_decay BOUNDARIES + branches ---
    # The tests above assert on STRUCTURE (which diagnosis enum, dataclass shape)
    # but never sit ON the decay-fraction boundary or exercise the FRAGMENTED
    # branch, so the comparison/branch mutants on lines 997-1015 survive (coverage
    # 100%, mutation kill 0% — the exact gap this campaign targets). Each test
    # below pins a specific mutant: the assertion fails if the operator/branch is
    # flipped. Constructed so the target strategy itself decays (DECAY/WATCH) and
    # siblings' fitness is controlled per rr_target via their win/loss outcomes.

    def _family_strategies(self, decay_rrs, fit_rrs):
        """Strategies sharing one family — target is the first decay_rr.

        decay_rrs get all-loss recent outcomes (→ DECAY); fit_rrs get all-win
        recent outcomes (→ FIT). Each rr_target is a distinct sibling.
        """
        strategies = []
        for rr in [*decay_rrs, *fit_rrs]:
            strategies.append(
                {
                    "strategy_id": f"MGC_CME_REOPEN_E1_RR{rr}_CB2_NO_FILTER",
                    "rr_target": rr,
                    "filter_type": "NO_FILTER",
                    "sample_size": 100,
                    "expectancy_r": 0.30,
                    "sharpe_ratio": 0.25,
                }
            )
        return strategies

    def _family_outcomes(self, decay_rrs, fit_rrs):
        """20 recent outcomes per rr: losses for decay_rrs, wins for fit_rrs."""
        outcomes = []
        for i in range(20):
            td = date(2025, (i % 11) + 1, (i % 27) + 1)
            for rr in decay_rrs:
                outcomes.append(
                    {"trading_day": td, "outcome": "loss", "pnl_r": -1.0, "rr_target": rr, "confirm_bars": 2}
                )
            for rr in fit_rrs:
                outcomes.append({"trading_day": td, "outcome": "win", "pnl_r": rr, "rr_target": rr, "confirm_bars": 2})
        return outcomes

    def test_decay_frac_below_half_is_fragmented(self, tmp_path):
        """1 decaying sibling of 4 (decay_frac 0.25 < 0.50) -> FRAGMENTED.

        Kills the FRAGMENTED-branch mutants (lines 1006/1011): if the OVERFIT
        guard `counts["DECAY"] == 0 and counts["WATCH"] == 0` is mutated, a set
        with one DECAY sibling misroutes to OVERFIT. Target = a FIT rr so its
        OWN status doesn't dominate; we read the sibling mix.
        """
        # Target RR1.5 (FIT); siblings: RR2.0 decays, RR2.5/RR3.0 FIT -> of the 3
        # siblings, 1 DECAY + 2 FIT = decay_frac 1/3 = 0.33 < 0.50 -> FRAGMENTED.
        strategies = self._family_strategies(decay_rrs=[2.0], fit_rrs=[1.5, 2.5, 3.0])
        outcomes = self._family_outcomes(decay_rrs=[2.0], fit_rrs=[1.5, 2.5, 3.0])
        db_path = _setup_family_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                "MGC_CME_REOPEN_E1_RR1.5_CB2_NO_FILTER",
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "FRAGMENTED"
        assert diag.siblings_decay == 1
        assert diag.siblings_fit == 2

    def test_decay_frac_exactly_half_is_regime_shift(self, tmp_path):
        """decay_frac == 0.50 exactly -> REGIME_SHIFT (boundary is inclusive).

        Kills the `decay_frac >= 0.50` -> `decay_frac > 0.50` mutant on line
        1003: with 2 decaying + 2 FIT siblings the fraction is exactly 0.50, so
        `>=` routes to REGIME_SHIFT but `>` flips it to FRAGMENTED.
        """
        # Target RR1.5 (FIT). Siblings: RR2.0/RR2.5 decay, RR3.0/RR3.5 FIT ->
        # 2 DECAY + 2 FIT of 4 siblings = decay_frac 0.50 exactly.
        strategies = self._family_strategies(decay_rrs=[2.0, 2.5], fit_rrs=[1.5, 3.0, 3.5])
        outcomes = self._family_outcomes(decay_rrs=[2.0, 2.5], fit_rrs=[1.5, 3.0, 3.5])
        db_path = _setup_family_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                "MGC_CME_REOPEN_E1_RR1.5_CB2_NO_FILTER",
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "REGIME_SHIFT"
        assert diag.siblings_decay == 2
        assert diag.siblings_fit == 2

    def test_all_fit_siblings_is_overfit_with_exact_counts(self, tmp_path):
        """0 decaying siblings -> OVERFIT, with exact sibling-count arithmetic.

        Kills the sibling-count accumulator mutants (line 992 `+ 1`) and the
        OVERFIT-branch guard: asserting the exact counts (3 FIT, 0 DECAY, 0
        WATCH) fails if the increment or the zero-checks are mutated.
        """
        # Target RR2.0 (FIT). Siblings RR1.5/RR2.5/RR3.0 all FIT -> 3 FIT, 0
        # decay -> OVERFIT (target's own decay, if any, is isolated).
        strategies = self._family_strategies(decay_rrs=[], fit_rrs=[2.0, 1.5, 2.5, 3.0])
        outcomes = self._family_outcomes(decay_rrs=[], fit_rrs=[2.0, 1.5, 2.5, 3.0])
        db_path = _setup_family_db(tmp_path, strategies, outcomes)

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            diag = diagnose_decay(
                con,
                "MGC_CME_REOPEN_E1_RR2.0_CB2_NO_FILTER",
                as_of_date=date(2025, 12, 31),
                rolling_months=18,
            )
        finally:
            con.close()

        assert diag.diagnosis == "OVERFIT"
        assert diag.siblings_fit == 3
        assert diag.siblings_decay == 0
        assert diag.siblings_watch == 0


def test_compute_fitness_raises_valueerror_for_missing_strategy(tmp_path):
    """ValueError must propagate — not be swallowed by except Exception."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))
    from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA

    con.execute(BARS_1M_SCHEMA)
    con.execute(BARS_5M_SCHEMA)
    con.execute(DAILY_FEATURES_SCHEMA)
    con.close()

    from trading_app.db_manager import init_trading_app_schema

    init_trading_app_schema(db_path=db_path)

    with pytest.raises(ValueError, match="not found"):
        compute_fitness("NONEXISTENT_STRATEGY_ID", db_path=db_path)
