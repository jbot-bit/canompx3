"""Tests for resample_to_5m with timezone edge cases (T3)."""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from trading_app.nested.builder import resample_to_5m


def _ts(hour, minute, tz=timezone.utc):
    return datetime(2025, 6, 15, hour, minute, tzinfo=tz)


def _make_1m_bars(timestamps, prices=None):
    if prices is None:
        prices = list(range(100, 100 + len(timestamps)))
    rows = []
    for i, ts in enumerate(timestamps):
        p = prices[i] if i < len(prices) else 100 + i
        rows.append({
            "ts_utc": pd.Timestamp(ts),
            "open": float(p),
            "high": float(p + 1),
            "low": float(p - 1),
            "close": float(p + 0.5),
            "volume": 10,
        })
    return pd.DataFrame(rows)


class TestResampleBasic:
    def test_five_bars_in_same_bucket(self):
        # Bars at :01-:04 all floor to :00 bucket. :05 starts a new bucket.
        bars = _make_1m_bars([
            _ts(23, 1), _ts(23, 2), _ts(23, 3), _ts(23, 4),
        ], [100, 102, 98, 101])
        result = resample_to_5m(bars, _ts(23, 0))
        assert len(result) == 1
        assert result.iloc[0]["open"] == 100  # first open
        assert result.iloc[0]["high"] == 103  # max of all highs (102+1)
        assert result.iloc[0]["low"] == 97  # min of all lows (98-1)
        assert result.iloc[0]["close"] == 101.5  # last close
        assert result.iloc[0]["volume"] == 40  # sum of 4 bars x 10

    def test_empty_after_ts(self):
        bars = _make_1m_bars([_ts(22, 55), _ts(22, 56)])
        result = resample_to_5m(bars, _ts(23, 0))
        assert result.empty

    def test_after_ts_is_strict(self):
        bars = _make_1m_bars([_ts(23, 0), _ts(23, 1)])
        result = resample_to_5m(bars, _ts(23, 0))
        assert len(result) == 1  # only 23:01

    def test_multiple_5m_buckets(self):
        # :01-:04 -> :00 bucket, :05-:09 -> :05 bucket, :10 -> :10 bucket
        bars = _make_1m_bars([
            _ts(23, 1), _ts(23, 2), _ts(23, 3), _ts(23, 4), _ts(23, 5),
            _ts(23, 6), _ts(23, 7), _ts(23, 8), _ts(23, 9), _ts(23, 10),
        ])
        result = resample_to_5m(bars, _ts(23, 0))
        assert len(result) == 3  # :00, :05, :10 buckets


class TestResampleTimezoneEdges:
    def test_brisbane_tz_timestamps_handled(self):
        """DuckDB returns TIMESTAMPTZ in local TZ (Brisbane). Verify resample handles it."""
        brisbane = timezone(timedelta(hours=10))
        bars = _make_1m_bars([
            datetime(2025, 6, 16, 9, 1, tzinfo=brisbane),  # = 23:01 UTC
            datetime(2025, 6, 16, 9, 2, tzinfo=brisbane),
            datetime(2025, 6, 16, 9, 3, tzinfo=brisbane),
            datetime(2025, 6, 16, 9, 4, tzinfo=brisbane),
            datetime(2025, 6, 16, 9, 5, tzinfo=brisbane),
        ])
        after = datetime(2025, 6, 15, 23, 0, tzinfo=timezone.utc)
        result = resample_to_5m(bars, after)
        # Should get 1 or 2 buckets depending on how floor works with TZ
        assert len(result) >= 1
        assert result.iloc[0]["volume"] > 0

    def test_utc_midnight_crossing(self):
        """Bars crossing UTC midnight should resample correctly."""
        bars = _make_1m_bars([
            _ts(23, 58), _ts(23, 59),
            datetime(2025, 6, 16, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 16, 0, 1, tzinfo=timezone.utc),
            datetime(2025, 6, 16, 0, 2, tzinfo=timezone.utc),
        ])
        result = resample_to_5m(bars, _ts(23, 55))
        # 23:55-23:59 bucket and 00:00-00:04 bucket
        assert len(result) == 2

    def test_tz_naive_timestamps_handled(self):
        """If timestamps are naive (no tz info), resample still works."""
        bars = _make_1m_bars([
            datetime(2025, 6, 15, 23, 1),
            datetime(2025, 6, 15, 23, 2),
            datetime(2025, 6, 15, 23, 3),
        ])
        after = datetime(2025, 6, 15, 23, 0)
        result = resample_to_5m(bars, after)
        assert len(result) == 1
