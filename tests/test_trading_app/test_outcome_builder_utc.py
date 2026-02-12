"""Tests for outcome_builder UTC normalization of break_ts (T7)."""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta, date
from pipeline.cost_model import get_cost_spec
from trading_app.outcome_builder import compute_single_outcome


COST_SPEC = get_cost_spec("MGC")


def _make_bars(timestamps, base_price=2700.0):
    """Create simple bars at given timestamps."""
    rows = []
    for i, ts in enumerate(timestamps):
        p = base_price + i * 0.5
        rows.append({
            "ts_utc": pd.Timestamp(ts),
            "open": p,
            "high": p + 2.0,
            "low": p - 2.0,
            "close": p + 1.0,
            "volume": 100,
        })
    return pd.DataFrame(rows)


class TestBreakTsUtcNormalization:
    def test_utc_break_ts_works(self):
        break_ts = datetime(2025, 6, 15, 23, 10, tzinfo=timezone.utc)
        td_end = datetime(2025, 6, 16, 22, 59, tzinfo=timezone.utc)
        timestamps = [
            datetime(2025, 6, 15, 23, 10 + i, tzinfo=timezone.utc)
            for i in range(30)
        ]
        bars = _make_bars(timestamps)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts,
            orb_high=2705.0, orb_low=2695.0, break_dir="long",
            rr_target=1.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=COST_SPEC, entry_model="E1",
        )
        # Should produce some result (entry or no-entry)
        assert isinstance(result, dict)

    def test_brisbane_tz_break_ts_works(self):
        """DuckDB may return break_ts in Brisbane TZ. Verify it still works."""
        brisbane = timezone(timedelta(hours=10))
        # 09:10 Brisbane = 23:10 UTC
        break_ts = datetime(2025, 6, 16, 9, 10, tzinfo=brisbane)
        td_end = datetime(2025, 6, 17, 8, 59, tzinfo=brisbane)
        timestamps = [
            datetime(2025, 6, 15, 23, 10 + i, tzinfo=timezone.utc)
            for i in range(30)
        ]
        bars = _make_bars(timestamps)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts,
            orb_high=2705.0, orb_low=2695.0, break_dir="long",
            rr_target=1.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=COST_SPEC, entry_model="E1",
        )
        assert isinstance(result, dict)

    def test_naive_break_ts_works(self):
        """Naive datetime (no tzinfo) should still work."""
        break_ts = datetime(2025, 6, 15, 23, 10)
        td_end = datetime(2025, 6, 16, 22, 59)
        timestamps = [datetime(2025, 6, 15, 23, 10 + i) for i in range(30)]
        bars = _make_bars(timestamps)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts,
            orb_high=2705.0, orb_low=2695.0, break_dir="long",
            rr_target=1.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=COST_SPEC, entry_model="E1",
        )
        assert isinstance(result, dict)

    def test_break_ts_after_all_bars_returns_empty(self):
        """If break_ts is after all bars, no entry should be found."""
        break_ts = datetime(2025, 6, 16, 23, 0, tzinfo=timezone.utc)
        td_end = datetime(2025, 6, 17, 22, 59, tzinfo=timezone.utc)
        timestamps = [
            datetime(2025, 6, 15, 23, i, tzinfo=timezone.utc)
            for i in range(30)
        ]
        bars = _make_bars(timestamps)
        result = compute_single_outcome(
            bars_df=bars, break_ts=break_ts,
            orb_high=2705.0, orb_low=2695.0, break_dir="long",
            rr_target=1.0, confirm_bars=1, trading_day_end=td_end,
            cost_spec=COST_SPEC, entry_model="E1",
        )
        assert result["outcome"] is None
        assert result["entry_ts"] is None
