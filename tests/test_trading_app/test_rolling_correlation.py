"""Tests for trading_app/rolling_correlation.py — 12 tests, all in-memory DuckDB."""

import math
import random
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from trading_app.rolling_correlation import (
    compute_co_loss_pct,
    compute_drawdown_correlation,
    compute_rolling_correlation,
    summarize_correlation_risk,
)


# =========================================================================
# Helpers
# =========================================================================

def _create_test_db(tmp_path: Path) -> Path:
    """Create a minimal DB with required schema for rolling correlation tests."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Minimal daily_features (needed by FK but not directly by rolling_correlation)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_features (
            symbol TEXT NOT NULL,
            trading_day DATE NOT NULL,
            orb_minutes INTEGER NOT NULL,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS orb_outcomes (
            trading_day       DATE NOT NULL,
            symbol            TEXT NOT NULL,
            orb_label         TEXT NOT NULL,
            orb_minutes       INTEGER NOT NULL,
            rr_target         DOUBLE NOT NULL,
            confirm_bars      INTEGER NOT NULL,
            entry_model       TEXT NOT NULL,
            outcome           TEXT,
            pnl_r             DOUBLE,
            ambiguous_bar     BOOLEAN DEFAULT FALSE,
            ts_outcome        TEXT,
            ts_pnl_r          DOUBLE,
            ts_exit_ts        TIMESTAMPTZ,
            PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes,
                         rr_target, confirm_bars, entry_model)
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS validated_setups (
            strategy_id       TEXT PRIMARY KEY,
            instrument        TEXT NOT NULL,
            orb_label         TEXT NOT NULL,
            orb_minutes       INTEGER NOT NULL,
            rr_target         DOUBLE NOT NULL,
            confirm_bars      INTEGER NOT NULL,
            entry_model       TEXT NOT NULL,
            filter_type       TEXT NOT NULL,
            is_family_head    BOOLEAN DEFAULT FALSE,
            status            TEXT NOT NULL DEFAULT 'ACTIVE'
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS strategy_trade_days (
            strategy_id TEXT NOT NULL,
            trading_day DATE NOT NULL,
            PRIMARY KEY (strategy_id, trading_day)
        )
    """)

    con.commit()
    con.close()
    return db_path


def _insert_strategy(con, sid, instrument="MGC", orb_label="CME_REOPEN",
                     orb_minutes=5, rr_target=2.0, confirm_bars=1,
                     entry_model="E1", filter_type="ORB_G5"):
    """Insert a validated_setups row."""
    con.execute(
        "INSERT INTO validated_setups "
        "(strategy_id, instrument, orb_label, orb_minutes, rr_target, "
        "confirm_bars, entry_model, filter_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [sid, instrument, orb_label, orb_minutes, rr_target, confirm_bars,
         entry_model, filter_type],
    )


def _insert_pnl(con, sid, instrument, orb_label, orb_minutes, rr_target,
                 confirm_bars, entry_model, trading_day, pnl_r):
    """Insert a trade day + outcome row."""
    con.execute(
        "INSERT OR IGNORE INTO daily_features (symbol, trading_day, orb_minutes) "
        "VALUES (?, ?, ?)",
        [instrument, trading_day, orb_minutes],
    )
    con.execute(
        "INSERT INTO strategy_trade_days (strategy_id, trading_day) VALUES (?, ?)",
        [sid, trading_day],
    )
    con.execute(
        "INSERT INTO orb_outcomes "
        "(trading_day, symbol, orb_label, orb_minutes, rr_target, confirm_bars, "
        "entry_model, outcome, pnl_r) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [trading_day, instrument, orb_label, orb_minutes, rr_target, confirm_bars,
         entry_model, "win" if pnl_r > 0 else "loss", pnl_r],
    )


def _populate_two_strategies(tmp_path, pnl_a_values, pnl_b_values,
                             start_date=date(2024, 1, 1)):
    """
    Create DB with two strategies (A and B) and daily PnL values.

    Both strategies share the same parameter set but different filter_type
    so their orb_outcomes rows don't collide on the PK.
    Uses separate orb_labels to avoid PK conflicts.
    """
    db_path = _create_test_db(tmp_path)
    con = duckdb.connect(str(db_path))

    sid_a = "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G5"
    sid_b = "MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G5"

    _insert_strategy(con, sid_a, orb_label="CME_REOPEN")
    _insert_strategy(con, sid_b, orb_label="TOKYO_OPEN")

    for i, pnl in enumerate(pnl_a_values):
        td = start_date + timedelta(days=i)
        _insert_pnl(con, sid_a, "MGC", "CME_REOPEN", 5, 2.0, 1, "E1", td, pnl)

    for i, pnl in enumerate(pnl_b_values):
        td = start_date + timedelta(days=i)
        _insert_pnl(con, sid_b, "MGC", "TOKYO_OPEN", 5, 2.0, 1, "E1", td, pnl)

    con.commit()
    con.close()
    return db_path, sid_a, sid_b


# =========================================================================
# Rolling correlation tests
# =========================================================================

class TestRollingCorrelation:

    def test_rolling_corr_identical_series(self, tmp_path):
        """Two strategies with identical daily PnL -> correlation = 1.0 in every window."""
        n = 200
        pnl = [0.5 if i % 3 == 0 else -1.0 for i in range(n)]
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl, pnl)

        results = compute_rolling_correlation(
            str(db_path), [sid_a, sid_b],
            window_days=60, min_overlap=30, step_days=20,
        )

        assert len(results) > 0
        for r in results:
            if r["correlation"] is not None:
                assert r["correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_rolling_corr_uncorrelated(self, tmp_path):
        """Two strategies with independent random PnL -> correlation near 0."""
        rng = random.Random(42)
        n = 300
        pnl_a = [rng.gauss(0, 1) for _ in range(n)]
        pnl_b = [rng.gauss(0, 1) for _ in range(n)]
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl_a, pnl_b)

        results = compute_rolling_correlation(
            str(db_path), [sid_a, sid_b],
            window_days=100, min_overlap=50, step_days=30,
        )

        assert len(results) > 0
        for r in results:
            if r["correlation"] is not None:
                assert abs(r["correlation"]) < 0.3

    def test_rolling_corr_insufficient_overlap(self, tmp_path):
        """Pair with < min_overlap in a window -> correlation is None."""
        # Strategy A trades days 0-29, strategy B trades days 100-129
        # No overlap at all
        db_path = _create_test_db(tmp_path)
        con = duckdb.connect(str(db_path))

        sid_a = "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G5"
        sid_b = "MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G5"
        _insert_strategy(con, sid_a, orb_label="CME_REOPEN")
        _insert_strategy(con, sid_b, orb_label="TOKYO_OPEN")

        start = date(2024, 1, 1)
        for i in range(30):
            td = start + timedelta(days=i)
            _insert_pnl(con, sid_a, "MGC", "CME_REOPEN", 5, 2.0, 1, "E1", td, 0.5)
        for i in range(100, 130):
            td = start + timedelta(days=i)
            _insert_pnl(con, sid_b, "MGC", "TOKYO_OPEN", 5, 2.0, 1, "E1", td, -0.5)

        con.commit()
        con.close()

        results = compute_rolling_correlation(
            str(db_path), [sid_a, sid_b],
            window_days=60, min_overlap=30, step_days=20,
        )

        # All windows should have None correlation (no shared days)
        for r in results:
            assert r["correlation"] is None

    def test_rolling_corr_window_count(self, tmp_path):
        """Verify correct number of windows for a given date range and step size."""
        n = 200
        pnl = [1.0] * n
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl, pnl)

        window_days = 60
        step_days = 20

        results = compute_rolling_correlation(
            str(db_path), [sid_a, sid_b],
            window_days=window_days, min_overlap=30, step_days=step_days,
        )

        # Expected: first window ends at index 59, then step by 20
        # Indices: 59, 79, 99, 119, 139, 159, 179, 199
        expected_windows = len(range(window_days - 1, n, step_days))
        # Each window produces 1 pair result (only 2 strategies = 1 pair)
        assert len(results) == expected_windows


# =========================================================================
# Drawdown correlation tests
# =========================================================================

class TestDrawdownCorrelation:

    def test_drawdown_corr_simultaneous(self, tmp_path):
        """Two strategies that draw down at the same time -> high co_drawdown_pct."""
        # Both have a big losing streak in the same period
        n = 100
        pnl = [0.5] * 20 + [-1.0] * 30 + [0.5] * 50  # heavy drawdown in middle
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl, pnl)

        result = compute_drawdown_correlation(
            str(db_path), [sid_a, sid_b],
            drawdown_threshold_r=-2.0,
        )

        pair = (sid_a, sid_b)
        assert pair in result
        # Both have identical drawdowns -> 100% overlap
        assert result[pair]["co_drawdown_pct"] == pytest.approx(1.0, abs=0.01)
        assert result[pair]["a_dd_days"] == result[pair]["b_dd_days"]
        assert result[pair]["overlap_dd_days"] == result[pair]["a_dd_days"]

    def test_drawdown_corr_offset(self, tmp_path):
        """Two strategies that draw down at different times -> low co_drawdown_pct."""
        n = 100
        # A draws down early, B draws down late
        pnl_a = [-1.0] * 30 + [0.5] * 70
        pnl_b = [0.5] * 70 + [-1.0] * 30
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl_a, pnl_b)

        result = compute_drawdown_correlation(
            str(db_path), [sid_a, sid_b],
            drawdown_threshold_r=-2.0,
        )

        pair = (sid_a, sid_b)
        assert pair in result
        # Drawdowns don't overlap -> co_drawdown_pct should be low
        assert result[pair]["co_drawdown_pct"] < 0.3

    def test_drawdown_corr_no_drawdown(self, tmp_path):
        """Strategy with no drawdown -> 0 overlap days."""
        n = 100
        # A is always winning (never dips below threshold), B has drawdown
        pnl_a = [1.0] * n
        pnl_b = [-1.0] * 30 + [0.5] * 70
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl_a, pnl_b)

        result = compute_drawdown_correlation(
            str(db_path), [sid_a, sid_b],
            drawdown_threshold_r=-2.0,
        )

        pair = (sid_a, sid_b)
        assert pair in result
        assert result[pair]["a_dd_days"] == 0
        assert result[pair]["overlap_dd_days"] == 0


# =========================================================================
# Co-loss tests
# =========================================================================

class TestCoLoss:

    def test_co_loss_all_same_direction(self, tmp_path):
        """Both lose on every shared day -> co_loss_pct = 1.0."""
        n = 50
        pnl = [-0.5] * n
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl, pnl)

        result = compute_co_loss_pct(str(db_path), [sid_a, sid_b])

        pair = (sid_a, sid_b)
        assert pair in result
        assert result[pair]["co_loss_pct"] == pytest.approx(1.0)
        assert result[pair]["both_traded_days"] == n
        assert result[pair]["both_negative_days"] == n

    def test_co_loss_opposite_days(self, tmp_path):
        """When A loses B wins and vice versa -> co_loss_pct = 0.0."""
        n = 50
        pnl_a = [-1.0 if i % 2 == 0 else 1.0 for i in range(n)]
        pnl_b = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
        db_path, sid_a, sid_b = _populate_two_strategies(tmp_path, pnl_a, pnl_b)

        result = compute_co_loss_pct(str(db_path), [sid_a, sid_b])

        pair = (sid_a, sid_b)
        assert pair in result
        assert result[pair]["co_loss_pct"] == pytest.approx(0.0)

    def test_co_loss_no_shared_days(self, tmp_path):
        """Strategies never trade same day -> both_traded_days = 0, no division by zero."""
        db_path = _create_test_db(tmp_path)
        con = duckdb.connect(str(db_path))

        sid_a = "MGC_CME_REOPEN_E1_RR2.0_CB1_ORB_G5"
        sid_b = "MGC_TOKYO_OPEN_E1_RR2.0_CB1_ORB_G5"
        _insert_strategy(con, sid_a, orb_label="CME_REOPEN")
        _insert_strategy(con, sid_b, orb_label="TOKYO_OPEN")

        # A trades on odd days, B on even days
        start = date(2024, 1, 1)
        for i in range(0, 50, 2):
            td = start + timedelta(days=i)
            _insert_pnl(con, sid_a, "MGC", "CME_REOPEN", 5, 2.0, 1, "E1", td, -1.0)
        for i in range(1, 50, 2):
            td = start + timedelta(days=i)
            _insert_pnl(con, sid_b, "MGC", "TOKYO_OPEN", 5, 2.0, 1, "E1", td, -1.0)

        con.commit()
        con.close()

        result = compute_co_loss_pct(str(db_path), [sid_a, sid_b])

        pair = (sid_a, sid_b)
        assert pair in result
        assert result[pair]["both_traded_days"] == 0
        assert result[pair]["both_negative_days"] == 0
        assert result[pair]["co_loss_pct"] == 0.0


# =========================================================================
# Summarize tests
# =========================================================================

class TestSummarize:

    def test_summarize_flags_high_corr_pair(self):
        """Pair with peak corr 0.90 gets flagged."""
        rolling = [
            {"window_end": date(2024, 6, 1), "pair": ("A", "B"),
             "correlation": 0.90, "overlap_days": 100},
            {"window_end": date(2024, 9, 1), "pair": ("A", "B"),
             "correlation": 0.70, "overlap_days": 100},
        ]
        drawdown = {("A", "B"): {
            "co_drawdown_pct": 0.1, "a_dd_days": 10,
            "b_dd_days": 20, "overlap_dd_days": 2,
        }}
        co_loss = {("A", "B"): {
            "co_loss_pct": 0.20, "both_traded_days": 100,
            "both_negative_days": 20,
        }}

        flagged = summarize_correlation_risk(rolling, drawdown, co_loss)

        assert len(flagged) == 1
        assert flagged[0]["pair"] == ("A", "B")
        assert flagged[0]["peak_rolling_corr"] == pytest.approx(0.90)

    def test_summarize_clean_portfolio(self):
        """All pairs below thresholds -> empty flagged list."""
        rolling = [
            {"window_end": date(2024, 6, 1), "pair": ("A", "B"),
             "correlation": 0.30, "overlap_days": 100},
            {"window_end": date(2024, 9, 1), "pair": ("A", "B"),
             "correlation": 0.40, "overlap_days": 100},
        ]
        drawdown = {("A", "B"): {
            "co_drawdown_pct": 0.1, "a_dd_days": 5,
            "b_dd_days": 8, "overlap_dd_days": 1,
        }}
        co_loss = {("A", "B"): {
            "co_loss_pct": 0.20, "both_traded_days": 100,
            "both_negative_days": 20,
        }}

        flagged = summarize_correlation_risk(rolling, drawdown, co_loss)

        assert len(flagged) == 0
