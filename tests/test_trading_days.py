"""
Tests for pipeline.ingest_dbn_mgc trading day computation.

Tests compute_trading_days() boundary cases — pure function, no DB needed.
"""

import pytest
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo

from pipeline.ingest_dbn_mgc import compute_trading_days


class TestComputeTradingDays:
    """Tests for vectorized trading day assignment."""

    def _make_df(self, utc_timestamps):
        """Helper: create DataFrame with UTC timestamps as index."""
        utc = ZoneInfo("UTC")
        idx = pd.DatetimeIndex([
            datetime(*ts, tzinfo=utc) for ts in utc_timestamps
        ])
        return pd.DataFrame({'open': [1.0] * len(idx)}, index=idx)

    def test_bar_at_1000_brisbane_is_same_day(self):
        # 10:00 Brisbane = 00:00 UTC (UTC+10)
        # 2024-06-03 00:00 UTC = 2024-06-03 10:00 Brisbane
        # Hour >= 9, so trading day = 2024-06-03
        df = self._make_df([(2024, 6, 3, 0, 0)])
        result = compute_trading_days(df)
        assert result.iloc[0] == date(2024, 6, 3)

    def test_bar_at_0859_brisbane_is_previous_day(self):
        # 08:59 Brisbane = 22:59 UTC previous day (UTC+10)
        # 2024-06-02 22:59 UTC = 2024-06-03 08:59 Brisbane
        # Hour < 9, so trading day = 2024-06-02 (previous day)
        df = self._make_df([(2024, 6, 2, 22, 59)])
        result = compute_trading_days(df)
        assert result.iloc[0] == date(2024, 6, 2)

    def test_bar_at_0900_brisbane_is_same_day(self):
        # 09:00 Brisbane = 23:00 UTC previous day (UTC+10)
        # 2024-06-02 23:00 UTC = 2024-06-03 09:00 Brisbane
        # Hour == 9, so trading day = 2024-06-03
        df = self._make_df([(2024, 6, 2, 23, 0)])
        result = compute_trading_days(df)
        assert result.iloc[0] == date(2024, 6, 3)

    def test_bar_at_midnight_brisbane(self):
        # 00:00 Brisbane = 14:00 UTC previous day (UTC+10)
        # 2024-06-02 14:00 UTC = 2024-06-03 00:00 Brisbane
        # Hour == 0 < 9, so trading day = 2024-06-02
        df = self._make_df([(2024, 6, 2, 14, 0)])
        result = compute_trading_days(df)
        assert result.iloc[0] == date(2024, 6, 2)

    def test_multiple_bars_mixed_days(self):
        # Mix of bars that should be on different trading days
        df = self._make_df([
            (2024, 6, 2, 23, 0),   # 09:00 Brisbane Jun 3 → trading day Jun 3
            (2024, 6, 3, 0, 0),    # 10:00 Brisbane Jun 3 → trading day Jun 3
            (2024, 6, 3, 22, 59),  # 08:59 Brisbane Jun 4 → trading day Jun 3
            (2024, 6, 3, 23, 0),   # 09:00 Brisbane Jun 4 → trading day Jun 4
        ])
        result = compute_trading_days(df)
        assert result.iloc[0] == date(2024, 6, 3)
        assert result.iloc[1] == date(2024, 6, 3)
        assert result.iloc[2] == date(2024, 6, 3)
        assert result.iloc[3] == date(2024, 6, 4)

    def test_returns_series_with_same_index(self):
        df = self._make_df([(2024, 6, 3, 0, 0)])
        result = compute_trading_days(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)
        assert result.index.equals(df.index)
