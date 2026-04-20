from datetime import UTC, date, datetime

import duckdb
import pytest

from trading_app.paper_trade_store import (
    PaperTradeCollisionError,
    PaperTradeRecord,
    ensure_paper_trades_schema,
    upsert_backfill_trade,
    write_completed_trade,
)


def _record(*, execution_source: str, pnl_r: float, trading_day: date = date(2026, 3, 7)) -> PaperTradeRecord:
    now = datetime.now(UTC)
    return PaperTradeRecord(
        trading_day=trading_day,
        orb_label="NYSE_OPEN",
        entry_time=now,
        direction="long",
        entry_price=20000.0,
        stop_price=19900.0,
        target_price=20200.0,
        exit_price=20100.0,
        exit_time=now,
        exit_reason="target",
        pnl_r=pnl_r,
        slippage_ticks=1.0,
        strategy_id="MNQ_NYSE_OPEN_TEST",
        lane_name="NYSE_OPEN_test",
        instrument="MNQ",
        orb_minutes=5,
        rr_target=2.0,
        filter_type="COST_LT12",
        entry_model="E2",
        execution_source=execution_source,
        pnl_dollar=125.0,
        notes=f"source={execution_source}",
    )


class TestPaperTradeStore:
    def test_write_completed_trade_bootstraps_full_schema(self, tmp_path):
        db_path = tmp_path / "paper_trades.db"

        write_completed_trade(_record(execution_source="live", pnl_r=1.25), db_path=db_path)

        con = duckdb.connect(str(db_path))
        cols = {row[1] for row in con.execute("PRAGMA table_info('paper_trades')").fetchall()}
        row = con.execute(
            "SELECT execution_source, pnl_dollar, notes, pnl_r FROM paper_trades WHERE strategy_id = ?",
            ["MNQ_NYSE_OPEN_TEST"],
        ).fetchone()
        con.close()

        assert {"execution_source", "pnl_dollar", "notes"} <= cols
        assert row == ("live", 125.0, "source=live", 1.25)

    def test_backfill_never_overwrites_live_row(self, tmp_path):
        db_path = tmp_path / "paper_trades.db"
        live = _record(execution_source="live", pnl_r=1.25)
        write_completed_trade(live, db_path=db_path)

        con = duckdb.connect(str(db_path))
        preserved = upsert_backfill_trade(con, _record(execution_source="backfill", pnl_r=0.25))
        row = con.execute(
            "SELECT execution_source, pnl_r FROM paper_trades WHERE strategy_id = ?",
            [live.strategy_id],
        ).fetchone()
        con.close()

        assert preserved is False
        assert row == ("live", 1.25)

    def test_live_write_replaces_backfill_row(self, tmp_path):
        db_path = tmp_path / "paper_trades.db"
        con = duckdb.connect(str(db_path))
        ensure_paper_trades_schema(con)
        inserted = upsert_backfill_trade(con, _record(execution_source="backfill", pnl_r=0.25))
        con.close()

        write_completed_trade(_record(execution_source="shadow", pnl_r=-0.5), db_path=db_path)

        con = duckdb.connect(str(db_path))
        row = con.execute(
            "SELECT execution_source, pnl_r, notes FROM paper_trades WHERE strategy_id = ?",
            ["MNQ_NYSE_OPEN_TEST"],
        ).fetchone()
        con.close()

        assert inserted is True
        assert row == ("shadow", -0.5, "source=shadow")

    def test_live_write_refuses_to_overwrite_existing_live_row(self, tmp_path):
        """A second real-execution write on the same key must fail closed."""
        db_path = tmp_path / "paper_trades.db"
        first = _record(execution_source="live", pnl_r=1.25)
        write_completed_trade(first, db_path=db_path)

        second = _record(execution_source="live", pnl_r=-0.5)
        with pytest.raises(PaperTradeCollisionError):
            write_completed_trade(second, db_path=db_path)

        con = duckdb.connect(str(db_path))
        row = con.execute(
            "SELECT execution_source, pnl_r FROM paper_trades WHERE strategy_id = ?",
            [first.strategy_id],
        ).fetchone()
        con.close()

        assert row == ("live", 1.25)
