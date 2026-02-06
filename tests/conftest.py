"""
Shared test fixtures for the MGC data pipeline.

Provides:
- tmp_db: temporary DuckDB with schema initialized
- sample_bars_1m: valid MGC 1-minute bar DataFrame
- sample_bars_1m_bad: DataFrame with deliberate violations
"""

import pytest
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import sys

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary DuckDB with bars_1m and bars_5m schema."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc        TIMESTAMPTZ NOT NULL,
            symbol        TEXT        NOT NULL,
            source_symbol TEXT        NOT NULL,
            open          DOUBLE      NOT NULL,
            high          DOUBLE      NOT NULL,
            low           DOUBLE      NOT NULL,
            close         DOUBLE      NOT NULL,
            volume        BIGINT      NOT NULL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)

    con.execute("""
        CREATE TABLE bars_5m (
            ts_utc        TIMESTAMPTZ NOT NULL,
            symbol        TEXT        NOT NULL,
            source_symbol TEXT,
            open          DOUBLE      NOT NULL,
            high          DOUBLE      NOT NULL,
            low           DOUBLE      NOT NULL,
            close         DOUBLE      NOT NULL,
            volume        BIGINT      NOT NULL,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)

    yield con
    con.close()


@pytest.fixture
def sample_bars_1m():
    """Valid MGC 1-minute bars (5 bars at 10:00-10:04 Brisbane = 00:00-00:04 UTC)."""
    utc = ZoneInfo("UTC")
    timestamps = pd.DatetimeIndex([
        datetime(2024, 6, 3, 0, 0, tzinfo=utc),
        datetime(2024, 6, 3, 0, 1, tzinfo=utc),
        datetime(2024, 6, 3, 0, 2, tzinfo=utc),
        datetime(2024, 6, 3, 0, 3, tzinfo=utc),
        datetime(2024, 6, 3, 0, 4, tzinfo=utc),
    ])

    df = pd.DataFrame({
        'symbol': ['MGCM4'] * 5,
        'open':   [2350.0, 2351.0, 2349.0, 2350.5, 2352.0],
        'high':   [2352.0, 2353.0, 2351.0, 2352.5, 2354.0],
        'low':    [2349.0, 2349.0, 2348.0, 2349.5, 2351.0],
        'close':  [2351.0, 2352.0, 2350.5, 2352.0, 2353.0],
        'volume': [100, 150, 80, 120, 200],
    }, index=timestamps)
    df.index.name = 'ts_event'

    return df


@pytest.fixture
def sample_bars_1m_bad():
    """MGC bars with deliberate validation violations."""
    utc = ZoneInfo("UTC")
    timestamps = pd.DatetimeIndex([
        datetime(2024, 6, 3, 0, 0, tzinfo=utc),
        datetime(2024, 6, 3, 0, 1, tzinfo=utc),
        datetime(2024, 6, 3, 0, 2, tzinfo=utc),
    ])

    df = pd.DataFrame({
        'symbol': ['MGCM4'] * 3,
        'open':   [2350.0, float('nan'), 2350.0],   # NaN in row 2
        'high':   [2352.0, 2353.0, 2348.0],          # high < low in row 3
        'low':    [2349.0, 2350.0, 2351.0],           # low > high in row 3
        'close':  [2351.0, 2349.0, 2350.0],
        'volume': [100, 150, -10],                    # negative volume in row 3
    }, index=timestamps)
    df.index.name = 'ts_event'

    return df
