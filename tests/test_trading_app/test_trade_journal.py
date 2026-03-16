"""Tests for TradeJournal — persistent live trade recording."""

from datetime import date

import duckdb
import pytest

from trading_app.live.trade_journal import TradeJournal, generate_trade_id


@pytest.fixture
def journal_path(tmp_path):
    return tmp_path / "test_journal.db"


@pytest.fixture
def journal(journal_path):
    j = TradeJournal(journal_path, mode="test")
    yield j
    j.close()


class TestGenerateTradeId:
    def test_unique(self):
        ids = {generate_trade_id() for _ in range(100)}
        assert len(ids) == 100

    def test_is_string(self):
        assert isinstance(generate_trade_id(), str)


class TestTradeJournalInit:
    def test_creates_table(self, journal_path):
        j = TradeJournal(journal_path, mode="live")
        con = duckdb.connect(str(journal_path))
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert "live_trades" in tables
        con.close()
        j.close()

    def test_idempotent_create(self, journal_path):
        j1 = TradeJournal(journal_path, mode="live")
        j1.close()
        j2 = TradeJournal(journal_path, mode="live")
        j2.close()

    def test_invalid_path_fails_open(self, tmp_path):
        """Journal with bad path should not raise — fails open."""
        bad_path = tmp_path / "nonexistent_dir" / "sub" / "journal.db"
        j = TradeJournal(bad_path, mode="live")
        assert j._con is None
        j.close()


class TestRecordEntry:
    def test_basic_entry(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="MGC_TOKYO_OPEN_E2_CB1_ORB_G4_long_RR3.0",
            direction="long",
            entry_model="E2",
            engine_entry=2950.0,
            fill_entry=2950.2,
            broker="tradovate",
            order_id_entry=12345,
            contracts=1,
        )
        rows = journal._con.execute("SELECT * FROM live_trades WHERE trade_id = ?", [tid]).fetchall()
        assert len(rows) == 1

    def test_entry_without_fill(self, journal):
        """E2 stop-market: fill comes later via poller."""
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="test_strat",
            direction="long",
            entry_model="E2",
            engine_entry=2950.0,
            fill_entry=None,
            broker="tradovate",
            order_id_entry=99999,
        )
        row = journal._con.execute("SELECT fill_entry FROM live_trades WHERE trade_id = ?", [tid]).fetchone()
        assert row[0] is None

    def test_signal_only_entry(self, journal):
        j = TradeJournal(journal._db_path, mode="signal")
        tid = generate_trade_id()
        j.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MNQ",
            strategy_id="signal_strat",
            direction="short",
            entry_model="E1",
            engine_entry=19500.0,
        )
        row = j._con.execute("SELECT session_mode FROM live_trades WHERE trade_id = ?", [tid]).fetchone()
        assert row[0] == "signal"
        j.close()


class TestUpdateEntryFill:
    def test_updates_fill_price(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="test_strat",
            direction="long",
            entry_model="E2",
            engine_entry=2950.0,
            fill_entry=None,
        )
        journal.update_entry_fill(trade_id=tid, fill_entry=2950.5)
        row = journal._con.execute("SELECT fill_entry FROM live_trades WHERE trade_id = ?", [tid]).fetchone()
        assert row[0] == pytest.approx(2950.5)


class TestRecordExit:
    def test_full_round_trip(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="test_strat",
            direction="long",
            entry_model="E2",
            engine_entry=2950.0,
            fill_entry=2950.2,
            broker="tradovate",
            order_id_entry=100,
        )
        journal.record_exit(
            trade_id=tid,
            engine_exit=2953.0,
            fill_exit=2952.8,
            actual_r=0.85,
            expected_r=0.30,
            slippage_pts=0.4,
            pnl_dollars=28.0,
            exit_reason="target",
            order_id_exit=101,
            cusum_alarm=False,
        )
        row = journal._con.execute(
            "SELECT actual_r, exit_reason, exited_at FROM live_trades WHERE trade_id = ?", [tid]
        ).fetchone()
        assert row[0] == pytest.approx(0.85)
        assert row[1] == "target"
        assert row[2] is not None  # exited_at timestamp set

    def test_kill_switch_exit(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="ks_strat",
            direction="short",
            entry_model="E1",
            engine_entry=2960.0,
        )
        journal.record_exit(
            trade_id=tid,
            exit_reason="kill_switch",
        )
        row = journal._con.execute("SELECT exit_reason, actual_r FROM live_trades WHERE trade_id = ?", [tid]).fetchone()
        assert row[0] == "kill_switch"
        assert row[1] is None  # no R computed for emergency flatten

    def test_scratch_exit(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MNQ",
            strategy_id="scratch_strat",
            direction="long",
            entry_model="E2",
            engine_entry=19500.0,
            fill_entry=19500.5,
        )
        journal.record_exit(
            trade_id=tid,
            engine_exit=19500.0,
            fill_exit=19499.8,
            actual_r=-0.15,
            expected_r=0.20,
            slippage_pts=1.2,
            exit_reason="scratch",
        )
        row = journal._con.execute("SELECT exit_reason, actual_r FROM live_trades WHERE trade_id = ?", [tid]).fetchone()
        assert row[0] == "scratch"
        assert row[1] == pytest.approx(-0.15)


class TestDailySummary:
    def test_summary_with_trades(self, journal):
        day = date(2026, 3, 14)
        for i, (r, slip) in enumerate([(0.85, 0.2), (-1.0, 0.3), (0.40, 0.1)]):
            tid = generate_trade_id()
            journal.record_entry(
                trade_id=tid,
                trading_day=day,
                instrument="MGC",
                strategy_id=f"strat_{i}",
                direction="long",
                entry_model="E2",
                engine_entry=2950.0 + i,
            )
            journal.record_exit(
                trade_id=tid,
                actual_r=r,
                slippage_pts=slip,
                pnl_dollars=r * 100,
                exit_reason="target" if r > 0 else "stop",
            )
        summary = journal.daily_summary(day)
        assert summary["n_trades"] == 3
        assert summary["total_r"] == pytest.approx(0.25)  # 0.85 - 1.0 + 0.40
        assert summary["total_slippage_pts"] == pytest.approx(0.6)

    def test_summary_empty_day(self, journal):
        summary = journal.daily_summary(date(2026, 1, 1))
        assert summary["n_trades"] == 0

    def test_summary_excludes_incomplete(self, journal):
        """Trades without exit should NOT appear in summary."""
        day = date(2026, 3, 14)
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=day,
            instrument="MGC",
            strategy_id="incomplete",
            direction="long",
            entry_model="E1",
            engine_entry=2950.0,
        )
        summary = journal.daily_summary(day)
        assert summary["n_trades"] == 0


class TestIncompleteTrades:
    def test_detects_crash_orphans(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="orphan_strat",
            direction="long",
            entry_model="E2",
            engine_entry=2950.0,
        )
        incomplete = journal.incomplete_trades()
        assert len(incomplete) == 1
        assert incomplete[0]["strategy_id"] == "orphan_strat"

    def test_completed_not_incomplete(self, journal):
        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="done_strat",
            direction="long",
            entry_model="E1",
            engine_entry=2950.0,
        )
        journal.record_exit(trade_id=tid, exit_reason="target", actual_r=1.0)
        assert len(journal.incomplete_trades()) == 0


class TestOrphanedExit:
    def test_exit_for_missing_entry_logs_critical(self, journal, caplog):
        """Exit for nonexistent trade_id should log CRITICAL, not raise."""
        import logging

        with caplog.at_level(logging.CRITICAL):
            journal.record_exit(trade_id="nonexistent_trade_id", exit_reason="target", actual_r=1.0)
        assert "ORPHANED" in caplog.text

    def test_exit_for_existing_entry_no_orphan_warning(self, journal, caplog):
        """Normal exit should not log ORPHANED."""
        import logging

        tid = generate_trade_id()
        journal.record_entry(
            trade_id=tid,
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="normal",
            direction="long",
            entry_model="E1",
            engine_entry=2950.0,
        )
        with caplog.at_level(logging.CRITICAL):
            journal.record_exit(trade_id=tid, exit_reason="target", actual_r=1.0)
        assert "ORPHANED" not in caplog.text


class TestFailOpen:
    def test_record_entry_after_close(self, journal_path):
        """Writing after close should not raise."""
        j = TradeJournal(journal_path, mode="test")
        j.close()
        # Should not raise — fails open
        j.record_entry(
            trade_id=generate_trade_id(),
            trading_day=date(2026, 3, 14),
            instrument="MGC",
            strategy_id="post_close",
            direction="long",
            entry_model="E1",
            engine_entry=2950.0,
        )

    def test_record_exit_after_close(self, journal_path):
        j = TradeJournal(journal_path, mode="test")
        j.close()
        j.record_exit(trade_id="nonexistent", exit_reason="test")

    def test_daily_summary_after_close(self, journal_path):
        j = TradeJournal(journal_path, mode="test")
        j.close()
        result = j.daily_summary(date(2026, 3, 14))
        assert result["n_trades"] == 0
        assert "error" in result

    def test_double_close_safe(self, journal_path):
        j = TradeJournal(journal_path, mode="test")
        j.close()
        j.close()  # should not raise
