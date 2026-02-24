"""
Tests for trading_app.outcome_builder module.

Tests compute_single_outcome() and build_outcomes() with synthetic data.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import pytest
import pandas as pd
import duckdb

from trading_app.outcome_builder import (
    compute_single_outcome,
    build_outcomes,
    RR_TARGETS,
    CONFIRM_BARS_OPTIONS,
)
from trading_app.config import ENTRY_MODELS
from trading_app.entry_rules import EntrySignal
from pipeline.cost_model import get_cost_spec

# ============================================================================
# HELPERS
# ============================================================================

def _cost():
    return get_cost_spec("MGC")

def _make_bars(start_ts, prices, interval_minutes=1):
    """
    Create bars_df from list of (open, high, low, close, volume) tuples.
    Starts at start_ts, each bar is interval_minutes apart.
    """
    rows = []
    ts = start_ts
    for o, h, l, c, v in prices:
        rows.append({
            "ts_utc": ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v),
        })
        ts = ts + timedelta(minutes=interval_minutes)
    return pd.DataFrame(rows)

# ============================================================================
# compute_single_outcome tests (E1: next bar open entry)
# ============================================================================

class TestComputeSingleOutcome:
    """Tests for the core outcome computation function using E1 model."""

    def test_long_win_rr2(self):
        """Long trade hits target at RR=2.0 with E1 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701 > orb_high=2700).
        # Bar 1: E1 entry at open=2703. Risk = 2703 - 2690 = 13. Target = 2703 + 26 = 2729
        # Bar 2-3: price rises to hit target
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2710, 2700, 2710, 100),
                (2710, 2720, 2709, 2718, 100),
                (2718, 2735, 2717, 2730, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "win"
        assert result["entry_price"] == 2703.0  # E1: next bar open
        assert result["stop_price"] == orb_low
        assert result["target_price"] == pytest.approx(2703.0 + 13.0 * 2.0, abs=0.01)
        assert result["pnl_r"] is not None
        assert result["pnl_r"] > 0

    def test_short_loss(self):
        """Short trade hits stop with E1 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: short confirm (close=2689 < orb_low=2690).
        # Bar 1: E1 entry at open=2688. Stop = 2700. Risk = 12.
        # Bar 2: price reverses and hits stop
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2692, 2695, 2688, 2689, 100),
                (2688, 2691, 2687, 2688, 100),
                (2688, 2701, 2687, 2700, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="short",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_price"] == orb_high

    def test_no_confirm_returns_nulls(self):
        """When confirm_bars=3 but only 1 bar closes outside, no entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2702, 2694, 2695, 100),
                (2695, 2698, 2694, 2697, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=3,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] is None
        assert result["entry_ts"] is None
        assert result["entry_price"] is None

    def test_scratch_when_no_exit(self):
        """Trade that never hits target or stop = scratch."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Narrow range, doesn't hit target or stop.
        # trading_day_end is very close, so scratch
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2705, 2698, 2702, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "scratch"

    def test_ambiguous_bar_conservative_loss(self):
        """Bar that hits both target and stop -> conservative loss."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Risk=13, target=2703+26=2729, stop=2690.
        # Bar 1 also huge range — hits both target and stop on fill bar
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2735, 2685, 2710, 200),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0

    def test_mae_mfe_tracked(self):
        """MAE and MFE are computed even for losses."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Favorable excursion (high=2710).
        # Bar 2: adverse excursion — stop hit (low=2689 < 2690)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2710, 2698, 2705, 100),
                (2705, 2706, 2689, 2690, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E1",
        )

        assert result["outcome"] == "loss"
        assert result["mfe_r"] is not None
        assert result["mfe_r"] > 0
        assert result["mae_r"] is not None
        assert result["mae_r"] > 0

    def test_zero_risk_returns_nulls(self):
        """If orb_high == orb_low (zero ORB), E3 produces zero risk."""
        # With E3, entry_price = orb_high and stop_price = orb_low.
        # When orb_high == orb_low, risk = 0 -> null result.
        orb_high, orb_low = 2700.0, 2700.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Need a bar that closes outside ORB (> 2700) to confirm,
        # then a retrace bar that touches 2700
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2700, 2702, 2699, 2701, 100),  # confirm: close > 2700
                (2701, 2702, 2699, 2700, 100),  # retrace: low <= 2700
            ],
        )

        result = compute_single_outcome(
            bars_df=bars,
            break_ts=break_ts,
            orb_high=orb_high,
            orb_low=orb_low,
            break_dir="long",
            rr_target=2.0,
            confirm_bars=1,
            trading_day_end=td_end,
            cost_spec=_cost(),
            entry_model="E3",
        )

        # E3 entry at 2700, stop at 2700 -> risk = 0 -> null result
        assert result["outcome"] is None
        assert result["entry_price"] is None

    def test_rr_targets_grid(self):
        """Different RR targets produce different target prices with E1."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701).
        # Bar 1: E1 entry at open=2703. Risk = 2703-2690 = 13.
        # Bar 2: massive range to ensure all RR targets hit
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2703, 2706, 2700, 2705, 100),
                (2705, 2750, 2700, 2750, 100),
            ],
        )

        targets_seen = set()
        for rr in RR_TARGETS:
            result = compute_single_outcome(
                bars_df=bars,
                break_ts=break_ts,
                orb_high=orb_high,
                orb_low=orb_low,
                break_dir="long",
                rr_target=rr,
                confirm_bars=1,
                trading_day_end=td_end,
                cost_spec=_cost(),
                entry_model="E1",
            )
            # E1 entry at open=2703, risk=13, target = 2703 + 13*rr
            assert result["target_price"] == pytest.approx(2703.0 + 13.0 * rr, abs=0.01)
            targets_seen.add(result["target_price"])

        assert len(targets_seen) == len(RR_TARGETS)

    def test_confirm_bars_2(self):
        """confirm_bars=2 needs 2 consecutive closes outside ORB."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2705, 2700, 2703, 100),
                (2703, 2730, 2702, 2725, 100),
            ],
        )

        r1 = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
        )
        r2 = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=2,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
        )

        assert r1["entry_ts"] is not None
        assert r2["entry_ts"] is not None
        assert r2["entry_ts"] > r1["entry_ts"]

# ============================================================================
# Entry model specific tests
# ============================================================================

class TestEntryModelE1:
    """E1: next bar open after confirm."""

    def test_e1_entry_is_next_bar_open(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm: close > orb_high
                (2703, 2730, 2702, 2725, 100),  # E1 entry: open=2703
                (2725, 2740, 2720, 2735, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
        )

        assert result["entry_price"] == 2703.0  # open of bar after confirm
        assert result["stop_price"] == orb_low
        # Risk = 2703 - 2690 = 13 points
        assert result["target_price"] == pytest.approx(2703.0 + 13.0 * 2.0, abs=0.01)

class TestEntryModelE3:
    """E3: limit at ORB level with retrace."""

    def test_e3_entry_at_orb_level_on_retrace(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm bar: close > orb_high
                (2701, 2705, 2699, 2703, 100),  # low=2699 <= orb_high=2700, retrace!
                (2703, 2730, 2702, 2725, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E3",
        )

        assert result["entry_price"] == orb_high  # limit fill at ORB level
        assert result["stop_price"] == orb_low
        # Risk = 2700 - 2690 = 10 (same as ORB size for E3)
        assert result["target_price"] == pytest.approx(2700.0 + 10.0 * 2.0, abs=0.01)

    def test_e3_no_retrace_no_fill(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2701, 2710, 2701, 2708, 100),  # low=2701 > orb_high=2700, NO retrace
                (2708, 2720, 2707, 2718, 100),  # still no retrace
            ],
        )

        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E3",
        )

        assert result["outcome"] is None
        assert result["entry_price"] is None

# ============================================================================
# build_outcomes integration tests (with temp DB)
# ============================================================================

class TestBuildOutcomes:
    """Integration tests using a temporary DuckDB database."""

    def _setup_db(self, tmp_path):
        """Create a temp DB with schema + minimal data for 1 trading day."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        base_ts = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        bars = []
        price = 2700.0
        for i in range(300):
            ts = base_ts + timedelta(minutes=i)
            o = price + i * 0.1
            h = o + 2
            l = o - 1
            c = o + 1
            bars.append((ts.isoformat(), "MGC", "GCG4", o, h, l, c, 100))

        con.executemany(
            """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        orb_high = 2700.0 + 4 * 0.1 + 2
        orb_low = 2700.0 - 1
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                date(2024, 1, 5), "MGC", 5, 300,
                orb_high, orb_low, "long",
                (base_ts + timedelta(minutes=6)).isoformat(),
            ],
        )
        con.commit()
        con.close()

        return db_path

    def test_build_writes_rows(self, tmp_path):
        """build_outcomes writes rows to orb_outcomes table."""
        db_path = self._setup_db(tmp_path)

        count = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            orb_minutes=5,
        )

        # 1 ORB break: E1 (6 RR * 5 CB = 30) + E3 (6 RR * 1 CB, if retrace found)
        assert count >= 30

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual >= 30

    def test_dry_run_no_writes(self, tmp_path):
        """dry_run=True produces no DB writes."""
        db_path = self._setup_db(tmp_path)

        count = build_outcomes(
            db_path=db_path,
            instrument="MGC",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            orb_minutes=5,
            dry_run=True,
        )

        assert count > 0

        con = duckdb.connect(str(db_path), read_only=True)
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]
        if "orb_outcomes" in tables:
            actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
            assert actual == 0
        con.close()

    def test_idempotent(self, tmp_path):
        """Running twice produces same row count (INSERT OR REPLACE)."""
        db_path = self._setup_db(tmp_path)

        count1 = build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )
        count2 = build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()

        assert actual == count1

    def test_no_break_day_no_rows(self, tmp_path):
        """Day with no break produces no orb_outcomes rows."""
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
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [date(2024, 1, 5), "MGC", 5, 100, 2700.0, 2690.0],
        )
        con.commit()
        con.close()

        count = build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )

        assert count == 0

    def test_entry_model_column_populated(self, tmp_path):
        """entry_model column has correct values in DB."""
        db_path = self._setup_db(tmp_path)
        build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )

        con = duckdb.connect(str(db_path), read_only=True)
        models = {r[0] for r in con.execute(
            "SELECT DISTINCT entry_model FROM orb_outcomes"
        ).fetchall()}
        con.close()
        assert models == {"E0", "E1", "E3"}

class TestCheckpointResume:
    """Tests for checkpoint/resume crash resilience."""

    def _setup_db(self, tmp_path):
        """Reuse the same setup as TestBuildOutcomes."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        base_ts = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        bars = []
        price = 2700.0
        for i in range(300):
            ts = base_ts + timedelta(minutes=i)
            o = price + i * 0.1
            h = o + 2
            l = o - 1
            c = o + 1
            bars.append((ts.isoformat(), "MGC", "GCG4", o, h, l, c, 100))

        con.executemany(
            """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
               VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
            bars,
        )

        orb_high = 2700.0 + 4 * 0.1 + 2
        orb_low = 2700.0 - 1
        con.execute(
            """INSERT INTO daily_features
               (trading_day, symbol, orb_minutes, bar_count_1m,
                orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
            [
                date(2024, 1, 5), "MGC", 5, 300,
                orb_high, orb_low, "long",
                (base_ts + timedelta(minutes=6)).isoformat(),
            ],
        )
        con.commit()
        con.close()

        return db_path

    def test_second_run_skips_computed_days(self, tmp_path):
        """Second run returns 0 new outcomes because all days are already computed."""
        db_path = self._setup_db(tmp_path)

        count1 = build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )
        assert count1 > 0

        count2 = build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )
        assert count2 == 0

        # Row count unchanged
        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual == count1

    def _setup_db_multi_day(self, tmp_path, num_days=11):
        """Create a temp DB with schema + data for multiple trading days."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        from trading_app.db_manager import init_trading_app_schema
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        for day_offset in range(num_days):
            td = date(2024, 1, 5 + day_offset)
            base_ts = datetime(2024, 1, 4 + day_offset, 23, 0, tzinfo=timezone.utc)
            bars = []
            price = 2700.0
            for i in range(60):
                ts = base_ts + timedelta(minutes=i)
                o = price + i * 0.1
                h = o + 2
                l = o - 1
                c = o + 1
                bars.append((ts.isoformat(), "MGC", "GCG4", o, h, l, c, 100))

            con.executemany(
                """INSERT INTO bars_1m (ts_utc, symbol, source_symbol, open, high, low, close, volume)
                   VALUES (?::TIMESTAMPTZ, ?, ?, ?, ?, ?, ?, ?)""",
                bars,
            )

            orb_high = 2700.0 + 4 * 0.1 + 2
            orb_low = 2700.0 - 1
            con.execute(
                """INSERT INTO daily_features
                   (trading_day, symbol, orb_minutes, bar_count_1m,
                    orb_CME_REOPEN_high, orb_CME_REOPEN_low, orb_CME_REOPEN_break_dir, orb_CME_REOPEN_break_ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?::TIMESTAMPTZ)""",
                [
                    td, "MGC", 5, 60,
                    orb_high, orb_low, "long",
                    (base_ts + timedelta(minutes=6)).isoformat(),
                ],
            )

        con.commit()
        con.close()
        return db_path

    def test_heartbeat_file_created(self, tmp_path):
        """Heartbeat file is written during non-dry-run build with 10+ days."""
        db_path = self._setup_db_multi_day(tmp_path, num_days=11)

        build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )

        heartbeat_path = tmp_path / "outcome_builder.heartbeat"
        assert heartbeat_path.exists()
        content = heartbeat_path.read_text()
        assert "MGC" in content
        assert "/" in content  # progress format: "10/11"

    def test_heartbeat_not_created_on_dry_run(self, tmp_path):
        """Heartbeat file is NOT written during dry-run."""
        db_path = self._setup_db(tmp_path)

        build_outcomes(
            db_path=db_path, instrument="MGC",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
            dry_run=True,
        )

        heartbeat_path = tmp_path / "outcome_builder.heartbeat"
        assert not heartbeat_path.exists()


class TestTimeStop:
    """Tests for T80 conditional time-stop annotation."""

    def test_keys_present_with_threshold_session(self):
        """compute_single_outcome returns ts_* keys for session with time-stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [(2698, 2701, 2695, 2701, 100),
             (2703, 2710, 2700, 2710, 100),
             (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        assert "ts_outcome" in result
        assert "ts_pnl_r" in result
        assert "ts_exit_ts" in result

    def test_no_threshold_session_leaves_ts_null(self):
        """Session with no time-stop (2300) -> ts_* = None."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [(2698, 2701, 2695, 2701, 100),
             (2703, 2710, 2700, 2710, 100),
             (2718, 2735, 2717, 2730, 100)],
        )
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="US_DATA_830",
        )
        assert result["ts_outcome"] is None
        assert result["ts_pnl_r"] is None
        assert result["ts_exit_ts"] is None

    def test_loss_after_threshold_gets_time_stopped(self):
        """Trade that loses after 30+ min at 1000 gets ts_outcome=time_stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        # Bar 0: confirm close above ORB high
        # Bar 1: E1 entry at open=2703, stop=2690, risk=13
        # Bars 2-40: price drifts below entry but above stop
        # Bar 61: hits stop
        bar_data = [
            (2698, 2701, 2695, 2701, 100),  # confirm
            (2703, 2705, 2698, 2700, 100),  # entry bar
        ]
        for i in range(58):
            price = 2700 - i * 0.1
            bar_data.append((price, price + 1, price - 1, price, 100))
        bar_data.append((2691, 2692, 2688, 2689, 100))  # hits stop

        bars = _make_bars(datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc), bar_data)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        # Baseline: full stop loss
        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        # Time-stop: fires at ~30m with partial loss (better than -1R)
        assert result["ts_outcome"] == "time_stop"
        assert result["ts_pnl_r"] < 0
        assert result["ts_pnl_r"] > -1.0

    def test_win_before_threshold_ts_matches_baseline(self):
        """Trade that wins before T80 -> ts_* matches baseline."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        # Quick win: target hit on bar 2 (2 min after entry, well before 30m)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2703, 2710, 2700, 2710, 100),  # entry bar
                (2710, 2720, 2709, 2718, 100),
                (2718, 2735, 2717, 2730, 100),  # hits target for RR2
            ],
        )
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        assert result["outcome"] == "win"
        assert result["ts_outcome"] == "win"
        assert result["ts_pnl_r"] == result["pnl_r"]

    def test_positive_at_threshold_keeps_running(self):
        """Trade above entry at threshold bar but not at target -> keeps running."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

        # Bar 0: confirm. Bar 1: entry at 2703, stop=2690, risk=13
        # Bars 2-40: price above entry (positive MTM) but below target (2729 for RR2)
        # Eventually scratches (no hit)
        bar_data = [
            (2698, 2701, 2695, 2701, 100),  # confirm
            (2703, 2706, 2700, 2705, 100),  # entry bar
        ]
        for _ in range(58):
            bar_data.append((2706, 2710, 2704, 2707, 100))  # positive but below target

        bars = _make_bars(datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc), bar_data)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E1",
            orb_label="TOKYO_OPEN",
        )
        # Baseline: scratch (no hit)
        assert result["outcome"] == "scratch"
        # Time-stop: positive at threshold bar -> keeps running, ts = baseline
        assert result["ts_outcome"] == "scratch"
        assert result["ts_pnl_r"] == result["pnl_r"]

    def test_schema_has_time_stop_columns(self, tmp_path):
        """Schema migration adds ts_outcome, ts_pnl_r, ts_exit_ts."""
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        from trading_app.db_manager import init_trading_app_schema
        db_path = tmp_path / "test.db"
        with duckdb.connect(str(db_path)) as con:
            con.execute(BARS_1M_SCHEMA)
            con.execute(BARS_5M_SCHEMA)
            con.execute(DAILY_FEATURES_SCHEMA)
        init_trading_app_schema(db_path=db_path)
        with duckdb.connect(str(db_path)) as con:
            cols = {r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'orb_outcomes'"
            ).fetchall()}
        assert "ts_outcome" in cols
        assert "ts_pnl_r" in cols
        assert "ts_exit_ts" in cols


class TestCLI:
    """Test CLI --help doesn't crash."""

    def test_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["outcome_builder", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from trading_app.outcome_builder import main
            main()
        assert exc_info.value.code == 0
        assert "instrument" in capsys.readouterr().out
