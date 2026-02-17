"""
Tests for scripts/report_edge_portfolio.py.
"""

import sys
from pathlib import Path
from datetime import date

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_app.db_manager import init_trading_app_schema


@pytest.fixture
def db_path(tmp_path):
    """Create temp DB with full schema + test data for report.

    2 families, 2 heads:
    - head_0900 (0900 E1 RR2.0 CB2 G5): trades on Jan 2,3,5
    - head_1800 (1800 E3 RR1.5 CB1 G4): trades on Jan 2,6
    Day Jan 2 has multi-session overlap.
    """
    path = tmp_path / "test.db"
    con = duckdb.connect(str(path))

    # Minimal daily_features table
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL, symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL, bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.close()

    init_trading_app_schema(db_path=path)

    con = duckdb.connect(str(path))

    # Insert daily_features rows (required by orb_outcomes FK)
    for d in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5), date(2024, 1, 6)]:
        con.execute("""
            INSERT INTO daily_features (trading_day, symbol, orb_minutes, bar_count_1m)
            VALUES (?, 'MGC', 5, 1440)
        """, [d])

    # Insert validated_setups
    for sid, orb, em, rr, cb, filt, expr in [
        ("MGC_0900_E1_RR2.0_CB2_ORB_G5", "0900", "E1", 2.0, 2, "ORB_G5", 0.30),
        ("MGC_1800_E3_RR1.5_CB1_ORB_G4", "1800", "E3", 1.5, 1, "ORB_G4", 0.20),
    ]:
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, years_tested,
             all_years_positive, stress_test_passed, status)
            VALUES (?, 'MGC', ?, 5, ?, ?, ?, ?, 100, 0.55, ?,
                    3, TRUE, TRUE, 'active')
        """, [sid, orb, rr, cb, em, filt, expr])

    # Insert edge_families (both ROBUST)
    con.execute("""
        INSERT INTO edge_families
        (family_hash, instrument, member_count, trade_day_count,
         head_strategy_id, head_expectancy_r, robustness_status)
        VALUES
        ('hash_0900', 'MGC', 5, 3, 'MGC_0900_E1_RR2.0_CB2_ORB_G5', 0.30, 'ROBUST'),
        ('hash_1800', 'MGC', 5, 2, 'MGC_1800_E3_RR1.5_CB1_ORB_G4', 0.20, 'ROBUST')
    """)

    # Insert strategy_trade_days
    for sid, days in [
        ("MGC_0900_E1_RR2.0_CB2_ORB_G5",
         [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]),
        ("MGC_1800_E3_RR1.5_CB1_ORB_G4",
         [date(2024, 1, 2), date(2024, 1, 6)]),
    ]:
        for d in days:
            con.execute(
                "INSERT INTO strategy_trade_days VALUES (?, ?)", [sid, d]
            )

    # Insert orb_outcomes (matching trades)
    outcomes = [
        # 0900 head: 3 trades (2 wins, 1 loss)
        (date(2024, 1, 2), "0900", 5, 2.0, 2, "E1", "win", 2.0, 100.0, 98.0),
        (date(2024, 1, 3), "0900", 5, 2.0, 2, "E1", "loss", -1.0, 100.0, 98.0),
        (date(2024, 1, 5), "0900", 5, 2.0, 2, "E1", "win", 2.0, 100.0, 98.0),
        # 1800 head: 2 trades (1 win, 1 loss)
        (date(2024, 1, 2), "1800", 5, 1.5, 1, "E3", "win", 1.5, 100.0, 98.5),
        (date(2024, 1, 6), "1800", 5, 1.5, 1, "E3", "loss", -1.0, 100.0, 98.5),
    ]
    for td, orb, om, rr, cb, em, outcome, pnl, entry, stop in outcomes:
        con.execute("""
            INSERT INTO orb_outcomes
            (trading_day, symbol, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, outcome, pnl_r,
             entry_price, stop_price)
            VALUES (?, 'MGC', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [td, orb, om, rr, cb, em, outcome, pnl, entry, stop])

    con.commit()
    con.close()
    return path


class TestReportInstrument:

    def test_correct_trade_count(self, db_path):
        from scripts.reports.report_edge_portfolio import report_instrument

        result = report_instrument(db_path, "MGC")
        assert result is not None
        assert result["total_trades"] == 5
        assert result["family_count"] == 2

    def test_overlap_detection(self, db_path):
        """Jan 2 has both 0900 and 1800 trades -> 1 multi-session day."""
        from scripts.reports.report_edge_portfolio import report_instrument

        result = report_instrument(db_path, "MGC")
        assert result["multi_session_days"] == 1
        # 4 unique days: Jan 2, 3, 5, 6
        assert result["unique_days"] == 4

    def test_per_orb_breakdown(self, db_path):
        from scripts.reports.report_edge_portfolio import report_instrument

        result = report_instrument(db_path, "MGC")
        assert "0900" in result["per_orb"]
        assert "1800" in result["per_orb"]
        assert result["per_orb"]["0900"]["trades"] == 3
        assert result["per_orb"]["1800"]["trades"] == 2

    def test_yearly_breakdown(self, db_path):
        from scripts.reports.report_edge_portfolio import report_instrument

        result = report_instrument(db_path, "MGC")
        assert 2024 in result["yearly"]
        assert result["yearly"][2024]["trades"] == 5

    def test_no_families_returns_none(self, db_path):
        """Instrument with no families returns None."""
        from scripts.reports.report_edge_portfolio import report_instrument

        result = report_instrument(db_path, "MNQ")
        assert result is None

    def test_no_edge_families_table_returns_none(self, tmp_path):
        """If edge_families table doesn't exist, returns None."""
        from scripts.reports.report_edge_portfolio import report_instrument

        path = tmp_path / "empty.db"
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE dummy (id INTEGER)")
        con.close()

        result = report_instrument(path, "MGC")
        assert result is None


class TestDailyLedger:

    def test_two_trades_same_day_sum(self):
        """Two trades on same day sum to one daily return."""
        from scripts.reports.report_edge_portfolio import _compute_daily_ledger

        trades = [
            {"trading_day": date(2024, 1, 2), "pnl_r": 2.0},
            {"trading_day": date(2024, 1, 2), "pnl_r": 1.5},
            {"trading_day": date(2024, 1, 3), "pnl_r": -1.0},
        ]
        daily_returns, overlap = _compute_daily_ledger(trades)
        assert len(daily_returns) == 2
        assert overlap == 1  # Jan 2 has 2 trades

        by_day = dict(daily_returns)
        assert by_day[date(2024, 1, 2)] == pytest.approx(3.5)
        assert by_day[date(2024, 1, 3)] == pytest.approx(-1.0)

    def test_no_trades_empty(self):
        from scripts.reports.report_edge_portfolio import _compute_daily_ledger

        daily_returns, overlap = _compute_daily_ledger([])
        assert daily_returns == []
        assert overlap == 0


class TestPortfolioStats:

    def test_positive_sharpe(self):
        """Mostly positive daily returns should yield positive Sharpe."""
        from scripts.reports.report_edge_portfolio import _compute_portfolio_stats

        # Vary returns to get nonzero std (constant returns -> std=0 -> Sharpe=None)
        daily_returns = [
            (date(2024, 1, i), 0.3 + (i % 3) * 0.2) for i in range(1, 31)
        ]
        stats = _compute_portfolio_stats(daily_returns)
        assert stats["sharpe_ann"] is not None
        assert stats["sharpe_ann"] > 0
        assert stats["max_dd_r"] == 0.0  # all positive returns
        assert stats["total_r"] > 0

    def test_empty_returns(self):
        from scripts.reports.report_edge_portfolio import _compute_portfolio_stats

        stats = _compute_portfolio_stats([])
        assert stats["trading_days"] == 0
        assert stats["sharpe_ann"] is None
        assert stats["total_r"] == 0.0

    def test_max_dd_computed(self):
        """Max DD should capture peak-to-trough."""
        from scripts.reports.report_edge_portfolio import _compute_portfolio_stats

        daily_returns = [
            (date(2024, 1, 1), 5.0),   # cumulative = 5, peak = 5
            (date(2024, 1, 2), -3.0),  # cumulative = 2, dd = 3
            (date(2024, 1, 3), -2.0),  # cumulative = 0, dd = 5
            (date(2024, 1, 4), 1.0),   # cumulative = 1, dd = 4
        ]
        stats = _compute_portfolio_stats(daily_returns)
        assert stats["max_dd_r"] == pytest.approx(5.0)


class TestPurgedFilter:

    def test_purged_excluded_by_default(self, tmp_path):
        """PURGED families excluded unless --include-purged."""
        from scripts.reports.report_edge_portfolio import report_instrument

        path = tmp_path / "purge_test.db"
        con = duckdb.connect(str(path))
        con.execute("""
            CREATE TABLE daily_features (
                trading_day DATE NOT NULL, symbol TEXT NOT NULL,
                orb_minutes INTEGER NOT NULL, bar_count_1m INTEGER,
                PRIMARY KEY (symbol, trading_day, orb_minutes)
            )
        """)
        con.close()

        init_trading_app_schema(db_path=path)

        con = duckdb.connect(str(path))

        # Insert daily_features rows (required by orb_outcomes FK)
        for d in [date(2024, 1, 2), date(2024, 1, 3)]:
            con.execute("""
                INSERT INTO daily_features (trading_day, symbol, orb_minutes, bar_count_1m)
                VALUES (?, 'MGC', 5, 1440)
            """, [d])

        # One ROBUST, one PURGED
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, years_tested,
             all_years_positive, stress_test_passed, status)
            VALUES
            ('s_robust', 'MGC', '0900', 5, 2.0, 2, 'E1', 'ORB_G5', 100,
             0.55, 0.30, 3, TRUE, TRUE, 'active'),
            ('s_purged', 'MGC', '1800', 5, 1.5, 1, 'E3', 'ORB_G4', 100,
             0.55, 0.20, 3, TRUE, TRUE, 'active')
        """)
        con.execute("""
            INSERT INTO edge_families
            (family_hash, instrument, member_count, trade_day_count,
             head_strategy_id, head_expectancy_r, robustness_status)
            VALUES
            ('hash_r', 'MGC', 5, 3, 's_robust', 0.30, 'ROBUST'),
            ('hash_p', 'MGC', 1, 2, 's_purged', 0.20, 'PURGED')
        """)

        # Trade days + outcomes for both
        for sid, days in [
            ("s_robust", [date(2024, 1, 2)]),
            ("s_purged", [date(2024, 1, 3)]),
        ]:
            for d in days:
                con.execute(
                    "INSERT INTO strategy_trade_days VALUES (?, ?)", [sid, d]
                )

        con.execute("""
            INSERT INTO orb_outcomes
            (trading_day, symbol, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, outcome, pnl_r,
             entry_price, stop_price)
            VALUES
            ('2024-01-02', 'MGC', '0900', 5, 2.0, 2, 'E1', 'win', 2.0, 100.0, 98.0),
            ('2024-01-03', 'MGC', '1800', 5, 1.5, 1, 'E3', 'win', 1.5, 100.0, 98.5)
        """)
        con.commit()
        con.close()

        # Default: only ROBUST
        result = report_instrument(path, "MGC", include_purged=False)
        assert result["family_count"] == 1
        assert result["total_trades"] == 1

        # Include purged: both
        result = report_instrument(path, "MGC", include_purged=True)
        assert result["family_count"] == 2
        assert result["total_trades"] == 2
