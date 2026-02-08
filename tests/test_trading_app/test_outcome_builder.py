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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
# compute_single_outcome tests (E2: confirm bar close entry)
# ============================================================================

class TestComputeSingleOutcome:
    """Tests for the core outcome computation function using E2 model."""

    def test_long_win_rr2(self):
        """Long trade hits target at RR=2.0 with E2 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701 > orb_high=2700). E2 entry at 2701.
        # Risk = 2701 - 2690 = 11. Target = 2701 + 22 = 2723
        # Bar 1-3: price rises to hit target
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2710, 2700, 2710, 100),
                (2710, 2720, 2709, 2718, 100),
                (2718, 2725, 2717, 2724, 100),
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
            entry_model="E2",
        )

        assert result["outcome"] == "win"
        assert result["entry_price"] == 2701.0  # E2: confirm bar close
        assert result["stop_price"] == orb_low
        assert result["target_price"] == pytest.approx(2701.0 + 11.0 * 2.0, abs=0.01)
        assert result["pnl_r"] is not None
        assert result["pnl_r"] > 0

    def test_short_loss(self):
        """Short trade hits stop with E2 entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: short confirm (close=2689 < orb_low=2690). E2 entry at 2689.
        # Stop = orb_high = 2700. Risk = 2700 - 2689 = 11.
        # Bar 1-2: price reverses and hits stop
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2692, 2695, 2688, 2689, 100),
                (2689, 2691, 2687, 2688, 100),
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
            entry_model="E2",
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
            entry_model="E2",
        )

        assert result["outcome"] is None
        assert result["entry_ts"] is None
        assert result["entry_price"] is None

    def test_scratch_when_no_exit(self):
        """Trade that never hits target or stop = scratch."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701). E2 entry at 2701.
        # Bar 1: narrow range, doesn't hit target (2723) or stop (2690)
        # trading_day_end is very close, so scratch
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2703, 2698, 2702, 100),
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
            entry_model="E2",
        )

        assert result["outcome"] == "scratch"

    def test_ambiguous_bar_conservative_loss(self):
        """Bar that hits both target and stop -> conservative loss."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701). E2 entry at 2701.
        # Risk = 11, target = 2723
        # Bar 1: huge range — hits both target (high=2725 > 2723) and stop (low=2685 < 2690)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2725, 2685, 2710, 200),
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
            entry_model="E2",
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0

    def test_mae_mfe_tracked(self):
        """MAE and MFE are computed even for losses."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701). E2 entry at 2701.
        # Bar 1: favorable excursion (high=2708), then
        # Bar 2: adverse excursion — stop hit (low=2689 < 2690)
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2708, 2698, 2705, 100),
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
            entry_model="E2",
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
        """Different RR targets produce different target prices with E2."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bar 0: confirm (close=2701). E2 entry at 2701. Risk = 11.
        # Bar 1: massive range to ensure all RR targets hit
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2750, 2700, 2750, 100),
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
                entry_model="E2",
            )
            # E2 entry at 2701, risk=11, target = 2701 + 11*rr
            assert result["target_price"] == pytest.approx(2701.0 + 11.0 * rr, abs=0.01)
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
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E2",
        )
        r2 = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=2,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E2",
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


class TestEntryModelE2:
    """E2: confirm bar close."""

    def test_e2_entry_is_confirm_close(self):
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm: close=2701
                (2701, 2730, 2700, 2725, 100),
            ],
        )

        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(), entry_model="E2",
        )

        assert result["entry_price"] == 2701.0  # confirm bar close
        assert result["stop_price"] == orb_low
        # Risk = 2701 - 2690 = 11 points
        assert result["target_price"] == pytest.approx(2701.0 + 11.0 * 2.0, abs=0.01)


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
                orb_0900_high, orb_0900_low, orb_0900_break_dir, orb_0900_break_ts)
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

        # 1 ORB break * 6 RR * 5 CB * 3 entry_models = 90 rows
        assert count >= 90

        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual >= 90

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
                orb_0900_high, orb_0900_low)
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
        assert models == {"E1", "E2", "E3"}


class TestCLI:
    """Test CLI --help doesn't crash."""

    def test_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, "trading_app/outcome_builder.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert r.returncode == 0
        assert "instrument" in r.stdout
