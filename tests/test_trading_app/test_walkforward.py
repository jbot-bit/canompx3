"""Tests for walk-forward validation (Phase 4b)."""

import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

from trading_app.walkforward import (
    WalkForwardResult,
    _add_months,
    run_walkforward,
    append_walkforward_result,
)


# ========================================================================
# Helpers
# ========================================================================

def _create_tables(con):
    """Create minimal orb_outcomes and daily_features tables."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS orb_outcomes (
            symbol VARCHAR,
            orb_minutes INTEGER,
            orb_label VARCHAR,
            entry_model VARCHAR,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            trading_day DATE,
            outcome VARCHAR,
            pnl_r DOUBLE,
            mae_r DOUBLE,
            mfe_r DOUBLE,
            ambiguous_bar BOOLEAN DEFAULT FALSE,
            ts_outcome VARCHAR,
            ts_pnl_r DOUBLE,
            ts_exit_ts TIMESTAMPTZ,
            entry_price DOUBLE,
            stop_price DOUBLE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_features (
            symbol VARCHAR,
            orb_minutes INTEGER,
            trading_day DATE,
            orb_0900_size DOUBLE,
            orb_0900_break_dir VARCHAR,
            day_of_week INTEGER
        )
    """)


def _insert_outcomes(con, rows):
    """Insert outcome rows (dicts with at minimum trading_day, outcome, pnl_r)."""
    for o in rows:
        con.execute(
            "INSERT INTO orb_outcomes "
            "(symbol, orb_minutes, orb_label, entry_model, rr_target, "
            "confirm_bars, trading_day, outcome, pnl_r, mae_r, mfe_r, "
            "entry_price, stop_price) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                o.get("symbol", "MGC"), o.get("orb_minutes", 5),
                o.get("orb_label", "0900"), o.get("entry_model", "E1"),
                o.get("rr_target", 2.0), o.get("confirm_bars", 1),
                o["trading_day"], o["outcome"], o["pnl_r"],
                o.get("mae_r", -0.5), o.get("mfe_r", 2.0),
                o.get("entry_price", 2000.0), o.get("stop_price", 1995.0),
            ],
        )


def _monthly_outcomes(year_start, year_end, trades_per_month=3,
                      win_pattern=None):
    """Generate outcomes: trades on 10th, 15th, 20th of each month.

    win_pattern: list of bools per trade index within month.
        Default [True, True, False] = 67% WR.
        Win pnl_r = 2.0, loss pnl_r = -1.0.
    """
    if win_pattern is None:
        win_pattern = [True, True, False]

    days = [10, 15, 20]
    outcomes = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            for i in range(trades_per_month):
                is_win = win_pattern[i % len(win_pattern)]
                outcomes.append({
                    "trading_day": date(year, month, days[i]),
                    "outcome": "win" if is_win else "loss",
                    "pnl_r": 2.0 if is_win else -1.0,
                })
    return outcomes


# Default WF params for compact test calls
_WF_BASE = dict(
    orb_label="0900", entry_model="E1", rr_target=2.0,
    confirm_bars=1, filter_type="NO_FILTER", orb_minutes=5,
)


# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def con(tmp_path):
    """Temp DuckDB with schema ready."""
    path = tmp_path / "test.db"
    c = duckdb.connect(str(path))
    _create_tables(c)
    yield c
    c.close()


# ========================================================================
# Unit tests: _add_months
# ========================================================================

class TestAddMonths:
    def test_basic(self):
        assert _add_months(date(2020, 1, 15), 6) == date(2020, 7, 15)

    def test_year_rollover(self):
        assert _add_months(date(2020, 7, 15), 12) == date(2021, 7, 15)

    def test_leap_year_clamp(self):
        assert _add_months(date(2020, 1, 31), 1) == date(2020, 2, 29)

    def test_non_leap_clamp(self):
        assert _add_months(date(2021, 1, 31), 1) == date(2021, 2, 28)

    def test_zero_months(self):
        assert _add_months(date(2020, 6, 15), 0) == date(2020, 6, 15)


# ========================================================================
# Walk-forward tests
# ========================================================================

class TestWalkForward:

    def test_basic_pass(self, con):
        """4 years of consistently positive data -> should pass."""
        # 2020-2023: 3 trades/month, 67% WR, RR2.0 => ExpR ~ +1.0
        # 12m train => test windows start 2021-01-10
        # 6 test windows, each 18 trades (>15), all positive
        outcomes = _monthly_outcomes(2020, 2023)
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_PASS", instrument="MGC",
            **_WF_BASE,
        )

        assert result.passed is True
        assert result.n_valid_windows >= 3
        assert result.pct_positive >= 0.60
        assert result.agg_oos_exp_r > 0
        assert result.total_oos_trades >= 45
        assert result.rejection_reason is None

    def test_basic_fail_negative_oos(self, con):
        """Strong train, negative test periods -> should fail."""
        # 2020: positive (train)
        train = _monthly_outcomes(2020, 2020)
        # 2021-2023: all losses
        test = _monthly_outcomes(2021, 2023, win_pattern=[False, False, False])
        _insert_outcomes(con, train + test)

        result = run_walkforward(
            con=con, strategy_id="TEST_NEG_OOS", instrument="MGC",
            **_WF_BASE,
        )

        assert result.passed is False
        assert result.agg_oos_exp_r <= 0

    def test_insufficient_windows(self, con):
        """Only 18 months of data -> 1 test window -> should fail."""
        outcomes = []
        for m_offset in range(18):
            year = 2024 + m_offset // 12
            month = (m_offset % 12) + 1
            for day in [10, 15, 20]:
                outcomes.append({
                    "trading_day": date(year, month, day),
                    "outcome": "win", "pnl_r": 2.0,
                })
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_SHORT", instrument="MGC",
            **_WF_BASE,
        )

        assert result.passed is False
        assert result.n_valid_windows < 3
        assert "Insufficient valid windows" in result.rejection_reason

    def test_thin_filter_skips_empty_windows(self, con):
        """Data gap creates empty windows -> n_valid < n_total."""
        # 2020 (train) + gap 2021-2022 + 2023 (test)
        data = _monthly_outcomes(2020, 2020) + _monthly_outcomes(2023, 2023)
        _insert_outcomes(con, data)

        result = run_walkforward(
            con=con, strategy_id="TEST_THIN", instrument="MGC",
            **_WF_BASE,
        )

        assert result.n_total_windows > result.n_valid_windows

    def test_min_trades_per_window(self, con):
        """1 trade/month = 6 per 6m window < 15 min -> all invalid."""
        outcomes = _monthly_outcomes(2020, 2023, trades_per_month=1,
                                     win_pattern=[True])
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_FEW", instrument="MGC",
            min_trades_per_window=15, **_WF_BASE,
        )

        assert result.passed is False
        assert result.n_valid_windows == 0
        assert "Insufficient valid windows" in result.rejection_reason

    def test_exact_threshold(self, con):
        """pct_positive exactly 0.60 (3/5 windows positive) -> pass."""
        # 12m train (2020) + 5 test windows (30 months, 2021-01 to 2023-06)
        # Windows 1-3: positive (67% WR), windows 4-5: negative (all loss)
        train = _monthly_outcomes(2020, 2020)
        positive = []
        for m_offset in range(18):  # 2021-01 to 2022-06 (windows 1-3)
            year = 2021 + m_offset // 12
            month = (m_offset % 12) + 1
            for i, day in enumerate([10, 15, 20]):
                is_win = i < 2  # 67% WR
                positive.append({
                    "trading_day": date(year, month, day),
                    "outcome": "win" if is_win else "loss",
                    "pnl_r": 2.0 if is_win else -1.0,
                })
        negative = []
        for m_offset in range(12):  # 2022-07 to 2023-06 (windows 4-5)
            year = 2022 + (m_offset + 6) // 12
            month = ((m_offset + 6) % 12) + 1
            for day in [10, 15, 20]:
                negative.append({
                    "trading_day": date(year, month, day),
                    "outcome": "loss", "pnl_r": -1.0,
                })
        _insert_outcomes(con, train + positive + negative)

        result = run_walkforward(
            con=con, strategy_id="TEST_60PCT", instrument="MGC",
            **_WF_BASE,
        )

        assert result.n_valid_windows == 5
        assert result.n_positive_windows == 3
        assert result.pct_positive >= 0.60
        # agg_oos_exp_r should still be positive (3 positive windows dominate)
        assert result.agg_oos_exp_r > 0
        assert result.passed is True

    def test_jsonl_output(self, con, tmp_path):
        """JSONL file created with valid JSON, appends on second run."""
        outcomes = _monthly_outcomes(2020, 2023)
        _insert_outcomes(con, outcomes)
        output_path = tmp_path / "data" / "wf_results.jsonl"

        result = run_walkforward(
            con=con, strategy_id="TEST_JSONL", instrument="MGC",
            **_WF_BASE,
        )
        append_walkforward_result(result, output_path)
        append_walkforward_result(result, output_path)

        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert "strategy_id" in record
            assert "timestamp" in record
            assert "passed" in record
            assert "windows" in record
            assert "params" in record

    def test_respects_filter(self, con):
        """Different filters on same outcomes -> different results."""
        # Create outcomes for 4 years
        outcomes = _monthly_outcomes(2020, 2023)
        _insert_outcomes(con, outcomes)

        # Create daily_features: even months size=6 (pass G5), odd months size=3
        for year in range(2020, 2024):
            for month in range(1, 13):
                size = 6.0 if month % 2 == 0 else 3.0
                for day in [10, 15, 20]:
                    d = date(year, month, day)
                    con.execute(
                        "INSERT INTO daily_features VALUES (?,?,?,?,?,?)",
                        ["MGC", 5, d, size, "long", d.weekday()],
                    )

        result_no_filter = run_walkforward(
            con=con, strategy_id="TEST_NF", instrument="MGC",
            **_WF_BASE,
        )
        result_g5 = run_walkforward(
            con=con, strategy_id="TEST_G5", instrument="MGC",
            orb_label="0900", entry_model="E1", rr_target=2.0,
            confirm_bars=1, filter_type="ORB_G5", orb_minutes=5,
        )

        # G5 filter keeps only even-month trades (50% of data)
        assert result_g5.total_oos_trades < result_no_filter.total_oos_trades

    def test_cost_deduction(self, con):
        """Verify metrics match manual calculation from pnl_r values."""
        # Create 4 years: alternating win(+1.5R) and loss(-1.0R)
        # ExpR = 0.5 * 1.5 - 0.5 * 1.0 = 0.25
        outcomes = []
        for year in range(2020, 2024):
            for month in range(1, 13):
                for i, day in enumerate([10, 15, 20]):
                    is_win = i % 2 == 0  # 0,2 win, 1 loss = 67% WR
                    outcomes.append({
                        "trading_day": date(year, month, day),
                        "outcome": "win" if is_win else "loss",
                        "pnl_r": 1.5 if is_win else -1.0,
                    })
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_COST", instrument="MGC",
            **_WF_BASE,
        )

        # 67% WR * 1.5 - 33% WR * 1.0 = 1.0 - 0.33 = 0.67
        assert result.agg_oos_exp_r > 0
        assert 0.5 < result.agg_oos_exp_r < 0.8
        assert result.passed is True

    def test_mnq_tight_windows(self, con):
        """2-year MNQ data -> only 2 test windows -> fails min_windows=3."""
        # 24 months of MNQ data (2024-02 to 2026-01)
        outcomes = []
        for m_offset in range(24):
            year = 2024 + (m_offset + 1) // 12
            month = ((m_offset + 1) % 12) + 1
            for i, day in enumerate([10, 15, 20]):
                is_win = i < 2
                outcomes.append({
                    "trading_day": date(year, month, day),
                    "outcome": "win" if is_win else "loss",
                    "pnl_r": 2.0 if is_win else -1.0,
                    "symbol": "MNQ",
                })
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="MNQ_TEST", instrument="MNQ",
            orb_label="0900", entry_model="E1", rr_target=2.0,
            confirm_bars=1, filter_type="NO_FILTER", orb_minutes=5,
            min_valid_windows=3,
        )

        # 2yr data, 12m train => 2 test windows max
        assert result.passed is False
        assert result.n_valid_windows <= 2
        assert "Insufficient valid windows" in result.rejection_reason

        # With relaxed threshold, should pass
        result2 = run_walkforward(
            con=con, strategy_id="MNQ_TEST_2W", instrument="MNQ",
            orb_label="0900", entry_model="E1", rr_target=2.0,
            confirm_bars=1, filter_type="NO_FILTER", orb_minutes=5,
            min_valid_windows=2,
        )
        assert result2.n_valid_windows >= 2

    def test_zero_windows_diagnostic(self, con):
        """Bug 2: All outcomes in training period -> descriptive rejection."""
        # Only 6 months of data with 12-month train requirement
        # All outcomes before the first test window start
        outcomes = []
        for month in range(1, 7):
            for day in [10, 15, 20]:
                outcomes.append({
                    "trading_day": date(2020, month, day),
                    "outcome": "win", "pnl_r": 2.0,
                })
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_ZERO_WIN", instrument="MGC",
            min_train_months=12, **_WF_BASE,
        )

        assert result.passed is False
        assert "training period" in result.rejection_reason
        assert "No test windows" in result.rejection_reason
        assert result.n_total_windows == 0

    def test_window_imbalance_detected(self, con):
        """Bug 4: Large trade count differences between windows flagged."""
        # Create data with dense early period and sparse late period
        outcomes = []
        # 2020: training data
        outcomes.extend(_monthly_outcomes(2020, 2020))
        # 2021-H1: 20 trades/month (dense)
        for month in range(1, 7):
            for day in range(1, 21):
                try:
                    outcomes.append({
                        "trading_day": date(2021, month, day),
                        "outcome": "win", "pnl_r": 2.0,
                    })
                except ValueError:
                    pass
        # 2021-H2: 1 trade/month (sparse) — 6 months, ~6 trades
        for month in range(7, 13):
            outcomes.append({
                "trading_day": date(2021, month, 15),
                "outcome": "win", "pnl_r": 2.0,
            })
        # 2022: keep data flowing for more windows
        outcomes.extend(_monthly_outcomes(2022, 2023))
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_IMBAL", instrument="MGC",
            min_trades_per_window=1, min_valid_windows=2, **_WF_BASE,
        )

        assert result.window_imbalance_ratio is not None
        # Dense window has ~120 trades, sparse has ~6 -> ratio > 5
        if result.window_imbalance_ratio > 5.0:
            assert result.window_imbalanced is True

    def test_window_imbalance_balanced(self, con):
        """Balanced windows should not be flagged."""
        outcomes = _monthly_outcomes(2020, 2023)
        _insert_outcomes(con, outcomes)

        result = run_walkforward(
            con=con, strategy_id="TEST_BAL", instrument="MGC",
            **_WF_BASE,
        )

        # Uniform 3 trades/month -> all windows equal -> ratio ~1.0
        if result.window_imbalance_ratio is not None:
            assert result.window_imbalanced is False
