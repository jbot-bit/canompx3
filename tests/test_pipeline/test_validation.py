"""
Tests for pipeline.ingest_dbn_mgc validation functions.

Tests validate_chunk() and validate_timestamp_utc() — pure functions, no DB needed.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from pipeline.ingest_dbn_mgc import validate_chunk, validate_timestamp_utc


# =============================================================================
# validate_chunk tests
# =============================================================================

class TestValidateChunk:
    """Tests for OHLCV validation (vectorized, fail-closed)."""

    def test_valid_data_passes(self, sample_bars_1m):
        valid, reason, bad = validate_chunk(sample_bars_1m)
        assert valid is True
        assert reason == ""
        assert bad is None

    def test_nan_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0, float('nan')],
            'high': [2352.0, 2353.0],
            'low': [2349.0, 2350.0],
            'close': [2351.0, 2349.0],
            'volume': [100, 150],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
            datetime(2024, 1, 1, 0, 1, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "NaN" in reason

    def test_infinite_price_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0, float('inf')],
            'high': [2352.0, float('inf')],
            'low': [2349.0, 2350.0],
            'close': [2351.0, 2349.0],
            'volume': [100, 150],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
            datetime(2024, 1, 1, 0, 1, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "Infinite" in reason or "infinite" in reason.lower()

    def test_negative_price_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0, -1.0],
            'high': [2352.0, 2353.0],
            'low': [2349.0, -1.0],
            'close': [2351.0, 2349.0],
            'volume': [100, 150],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
            datetime(2024, 1, 1, 0, 1, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "Non-positive" in reason or "positive" in reason.lower()

    def test_zero_price_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [0.0],
            'high': [0.0],
            'low': [0.0],
            'close': [0.0],
            'volume': [100],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False

    def test_high_less_than_low_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0],
            'high': [2348.0],  # high < low
            'low': [2351.0],
            'close': [2350.0],
            'volume': [100],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "high" in reason.lower()

    def test_negative_volume_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0],
            'high': [2352.0],
            'low': [2349.0],
            'close': [2351.0],
            'volume': [-1],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "volume" in reason.lower()

    def test_zero_volume_passes(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0],
            'high': [2352.0],
            'low': [2349.0],
            'close': [2351.0],
            'volume': [0],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is True

    def test_missing_column_fails(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': [2350.0],
            'high': [2352.0],
            # 'low' missing
            'close': [2351.0],
            'volume': [100],
        }, index=pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0, tzinfo=utc),
        ]))

        valid, reason, bad = validate_chunk(df)
        assert valid is False
        assert "Missing column" in reason

    def test_empty_dataframe_passes(self):
        utc = ZoneInfo("UTC")
        df = pd.DataFrame({
            'open': pd.Series(dtype='float64'),
            'high': pd.Series(dtype='float64'),
            'low': pd.Series(dtype='float64'),
            'close': pd.Series(dtype='float64'),
            'volume': pd.Series(dtype='int64'),
        }, index=pd.DatetimeIndex([], tz=utc))
        valid, reason, bad = validate_chunk(df)
        assert valid is True


# =============================================================================
# validate_timestamp_utc tests
# =============================================================================

class TestValidateTimestampUtc:
    """Tests for timezone verification gate."""

    def test_utc_passes(self, sample_bars_1m):
        valid, reason = validate_timestamp_utc(sample_bars_1m)
        assert valid is True
        assert reason == ""

    def test_naive_fails(self):
        df = pd.DataFrame(
            {'open': [1.0]},
            index=pd.DatetimeIndex([datetime(2024, 1, 1)])
        )
        valid, reason = validate_timestamp_utc(df)
        assert valid is False
        assert "None" in reason or "naive" in reason.lower()

    def test_wrong_tz_fails(self):
        ts = pd.DatetimeIndex([
            datetime(2024, 1, 1, tzinfo=ZoneInfo("US/Eastern"))
        ])
        df = pd.DataFrame({'open': [1.0]}, index=ts)
        valid, reason = validate_timestamp_utc(df)
        assert valid is False
        assert "UTC" in reason

    def test_null_timestamp_fails(self):
        ts = pd.DatetimeIndex([pd.NaT], tz="UTC")
        df = pd.DataFrame({'open': [1.0]}, index=ts)
        valid, reason = validate_timestamp_utc(df)
        assert valid is False
        assert "Null" in reason or "null" in reason.lower()
