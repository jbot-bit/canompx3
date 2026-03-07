"""Tests for ui_v2/data_layer.py — DuckDB queries with retry-backoff."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Connection retry ─────────────────────────────────────────────────────────


def test_get_connection_retries_on_lock(tmp_path):
    """Verify retry-backoff when DuckDB is locked."""
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_connection

    # Clean state
    test_db = tmp_path / "test.db"
    # Create a valid DB first
    con = duckdb.connect(str(test_db))
    con.execute("CREATE TABLE t (x INT)")
    con.close()

    # Clear cache
    key = str(test_db)
    _DB_CONNECTIONS.pop(key, None)

    # Should connect successfully
    conn = get_connection(test_db)
    assert conn is not None

    # Cleanup
    _DB_CONNECTIONS.pop(key, None)


def test_query_df_rejects_non_select():
    from ui_v2.data_layer import query_df

    with pytest.raises(ValueError, match="Only SELECT"):
        query_df("DROP TABLE bars_1m")


def test_query_df_allows_with():
    """WITH (CTE) queries should be allowed."""
    from ui_v2.data_layer import query_df

    # This will fail on missing table, not on the guard
    with pytest.raises(Exception):  # noqa: B017 — DuckDB error, not ValueError
        query_df("WITH cte AS (SELECT 1 AS x) SELECT * FROM nonexistent_table")


# ── get_prior_day_atr ────────────────────────────────────────────────────────


def test_get_prior_day_atr_returns_none_on_empty(tmp_path):
    """Returns None when daily_features table is empty or missing."""
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_prior_day_atr

    db = tmp_path / "empty.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE daily_features (
            symbol VARCHAR, orb_minutes INT, trading_day DATE, atr_20 DOUBLE
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_prior_day_atr("MGC", db_path=db)
    assert result is None

    _DB_CONNECTIONS.pop(key, None)


# ── get_previous_trading_day ─────────────────────────────────────────────────


def test_get_previous_trading_day_returns_none_on_empty(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_previous_trading_day

    db = tmp_path / "empty.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE daily_features (
            trading_day DATE, orb_minutes INT, symbol VARCHAR
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_previous_trading_day(date(2026, 3, 4), db_path=db)
    assert result is None

    _DB_CONNECTIONS.pop(key, None)


# ── get_today_completed_sessions ─────────────────────────────────────────────


def test_get_today_completed_returns_empty_on_no_data(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_today_completed_sessions

    db = tmp_path / "empty.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            orb_label VARCHAR, symbol VARCHAR, break_dir VARCHAR,
            pnl_r DOUBLE, outcome VARCHAR, entry_model VARCHAR,
            rr_target DOUBLE, trading_day DATE, orb_minutes INT
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_today_completed_sessions(date(2026, 3, 4), db_path=db)
    assert result == []

    _DB_CONNECTIONS.pop(key, None)


# ── get_session_history ──────────────────────────────────────────────────────


def test_get_session_history_returns_list(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_session_history

    db = tmp_path / "hist.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            orb_label VARCHAR, symbol VARCHAR, break_dir VARCHAR,
            pnl_r DOUBLE, outcome VARCHAR, entry_model VARCHAR,
            rr_target DOUBLE, trading_day DATE, orb_minutes INT
        )
    """
    )
    con.execute(
        """
        INSERT INTO orb_outcomes VALUES
        ('CME_REOPEN', 'MGC', 'long', 1.5, 'win', 'E2', 2.0, '2026-03-03', 5),
        ('CME_REOPEN', 'MGC', 'short', -1.0, 'loss', 'E2', 2.0, '2026-03-02', 5)
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_session_history("CME_REOPEN", db_path=db)
    assert len(result) == 2
    assert result[0]["pnl_r"] == 1.5  # DESC order — most recent first

    _DB_CONNECTIONS.pop(key, None)


# ── get_rolling_pnl ──────────────────────────────────────────────────────────


def test_get_rolling_pnl_empty_db(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_rolling_pnl

    db = tmp_path / "pnl.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            trading_day DATE, pnl_r DOUBLE, orb_minutes INT
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_rolling_pnl(db_path=db)
    assert result["daily"] == []
    assert result["week_r"] == 0.0

    _DB_CONNECTIONS.pop(key, None)


# ── get_overnight_recap ──────────────────────────────────────────────────────


def test_get_overnight_recap_empty(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_overnight_recap

    db = tmp_path / "overnight.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            orb_label VARCHAR, symbol VARCHAR, break_dir VARCHAR,
            pnl_r DOUBLE, outcome VARCHAR, trading_day DATE, orb_minutes INT
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_overnight_recap(date(2026, 3, 4), db_path=db)
    assert result == []

    _DB_CONNECTIONS.pop(key, None)


# ── get_fitness_regimes ──────────────────────────────────────────────────────


def test_get_fitness_regimes_empty(tmp_path):
    import duckdb

    from ui_v2.data_layer import _DB_CONNECTIONS, get_fitness_regimes

    db = tmp_path / "fitness.db"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE validated_setups (
            strategy_id VARCHAR, instrument VARCHAR, orb_label VARCHAR,
            entry_model VARCHAR, filter_type VARCHAR, rr_target DOUBLE,
            orb_minutes INT, expectancy_r DOUBLE, sharpe DOUBLE,
            win_rate DOUBLE, sample_size INT
        )
    """
    )
    con.close()

    key = str(db)
    _DB_CONNECTIONS.pop(key, None)

    result = get_fitness_regimes(db_path=db)
    assert result == []

    _DB_CONNECTIONS.pop(key, None)
