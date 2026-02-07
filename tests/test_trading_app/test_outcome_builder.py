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
# compute_single_outcome tests
# ============================================================================

class TestComputeSingleOutcome:
    """Tests for the core outcome computation function."""

    def test_long_win_rr2(self):
        """Long trade hits target at RR=2.0."""
        # ORB: high=2700, low=2690. Break long.
        # Entry at 2700 (orb_high), stop at 2690, risk=10.
        # RR=2 target = 2700 + 20 = 2720.
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Bars: break bar, then confirm bar closes above orb_high, then rally to target
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # break bar: close > orb_high
                (2701, 2705, 2700, 2703, 100),  # confirm bar 1: close > orb_high
                (2703, 2710, 2702, 2710, 100),  # rally
                (2710, 2721, 2709, 2720, 100),  # hits target 2720
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
        )

        assert result["outcome"] == "win"
        assert result["entry_price"] == orb_high
        assert result["stop_price"] == orb_low
        assert result["target_price"] == 2720.0
        assert result["pnl_r"] is not None
        assert result["pnl_r"] > 0

    def test_short_loss(self):
        """Short trade hits stop."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2692, 2695, 2688, 2689, 100),  # break bar: close < orb_low
                (2689, 2691, 2687, 2688, 100),  # confirm
                (2688, 2701, 2687, 2700, 100),  # bounces back to stop (orb_high=2700)
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
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0
        assert result["exit_price"] == orb_high  # stop for short is orb_high

    def test_no_confirm_returns_nulls(self):
        """When confirm_bars=3 but only 1 bar closes outside, no entry."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # close above orb_high
                (2701, 2702, 2694, 2695, 100),  # close INSIDE orb range — resets
                (2695, 2698, 2694, 2697, 100),  # still inside
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
        )

        assert result["outcome"] is None
        assert result["entry_ts"] is None
        assert result["entry_price"] is None

    def test_scratch_when_no_exit(self):
        """Trade that never hits target or stop = scratch."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 0, 5, tzinfo=timezone.utc)  # very short window

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm bar
                (2701, 2703, 2698, 2702, 100),  # drifts, no target/stop
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
        )

        assert result["outcome"] == "scratch"

    def test_ambiguous_bar_conservative_loss(self):
        """Bar that hits both target and stop → conservative loss."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                # Huge range: low=2685 (below stop 2690), high=2725 (above target 2720 for RR=2)
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
        )

        assert result["outcome"] == "loss"
        assert result["pnl_r"] == -1.0

    def test_mae_mfe_tracked(self):
        """MAE and MFE are computed even for losses."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),  # confirm
                (2701, 2708, 2698, 2705, 100),  # favorable excursion: +8
                (2705, 2706, 2689, 2690, 100),  # hits stop at 2690
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
        )

        assert result["outcome"] == "loss"
        assert result["mfe_r"] is not None
        assert result["mfe_r"] > 0  # had some favorable movement
        assert result["mae_r"] is not None
        assert result["mae_r"] > 0  # hit stop = max adverse

    def test_zero_risk_returns_nulls(self):
        """If entry == stop (zero risk), returns empty result."""
        orb_high, orb_low = 2700.0, 2700.0  # zero-size ORB
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2700, 2702, 2699, 2701, 100),
                (2701, 2705, 2700, 2703, 100),
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
        )

        assert result["outcome"] is None
        assert result["entry_price"] is None

    def test_rr_targets_grid(self):
        """Different RR targets produce different target prices."""
        orb_high, orb_low = 2700.0, 2690.0
        break_ts = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        td_end = datetime(2024, 1, 5, 23, 0, tzinfo=timezone.utc)

        # Big rally — hits all targets
        bars = _make_bars(
            datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc),
            [
                (2698, 2701, 2695, 2701, 100),
                (2701, 2750, 2700, 2750, 100),  # massive bar
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
            )
            assert result["target_price"] == 2700.0 + 10.0 * rr
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
                (2698, 2701, 2695, 2701, 100),  # 1st close above orb_high
                (2701, 2705, 2700, 2703, 100),  # 2nd close above orb_high → confirmed
                (2703, 2730, 2702, 2725, 100),  # rally to target
            ],
        )

        # confirm_bars=1 should trigger on first bar
        r1 = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=1,
            trading_day_end=td_end, cost_spec=_cost(),
        )
        # confirm_bars=2 should trigger on second bar (later entry_ts)
        r2 = compute_single_outcome(
            bars_df=bars, break_ts=break_ts, orb_high=orb_high, orb_low=orb_low,
            break_dir="long", rr_target=2.0, confirm_bars=2,
            trading_day_end=td_end, cost_spec=_cost(),
        )

        assert r1["entry_ts"] is not None
        assert r2["entry_ts"] is not None
        assert r2["entry_ts"] > r1["entry_ts"]


# ============================================================================
# build_outcomes integration tests (with temp DB)
# ============================================================================

class TestBuildOutcomes:
    """Integration tests using a temporary DuckDB database."""

    def _setup_db(self, tmp_path):
        """Create a temp DB with schema + minimal data for 1 trading day."""
        db_path = tmp_path / "test.db"
        con = duckdb.connect(str(db_path))

        # Create pipeline schema
        from pipeline.init_db import BARS_1M_SCHEMA, BARS_5M_SCHEMA, DAILY_FEATURES_SCHEMA
        con.execute(BARS_1M_SCHEMA)
        con.execute(BARS_5M_SCHEMA)
        con.execute(DAILY_FEATURES_SCHEMA)

        # Create trading_app schema
        from trading_app.db_manager import init_trading_app_schema
        con.close()
        init_trading_app_schema(db_path=db_path)

        con = duckdb.connect(str(db_path))

        # Insert 1 trading day of bars_1m (2024-01-05, UTC range 23:00 Jan 4 → 23:00 Jan 5)
        # ORB 0900 Brisbane = 23:00 UTC previous day. 5-min ORB = 23:00-23:05 UTC
        base_ts = datetime(2024, 1, 4, 23, 0, tzinfo=timezone.utc)
        bars = []
        price = 2700.0
        for i in range(300):  # 5 hours of bars
            ts = base_ts + timedelta(minutes=i)
            # Gentle uptrend
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

        # Insert daily_features with a break on ORB 0900
        # ORB high/low from first 5 bars
        orb_high = 2700.0 + 4 * 0.1 + 2  # ~2702.4
        orb_low = 2700.0 - 1  # 2699.0
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

        # Should produce rows for the 0900 ORB break
        # 6 RR targets × 3 confirm_bars = 18 rows for that ORB
        assert count >= 18

        # Verify DB has rows
        con = duckdb.connect(str(db_path), read_only=True)
        actual = con.execute("SELECT COUNT(*) FROM orb_outcomes").fetchone()[0]
        con.close()
        assert actual >= 18

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

        assert count > 0  # counted but not written

        con = duckdb.connect(str(db_path), read_only=True)
        # orb_outcomes table may not exist in dry_run since schema init is skipped
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

        # Second run replaces, doesn't duplicate
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

        # Insert daily_features with NO break
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
