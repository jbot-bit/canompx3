"""Tests that run_parallel_ingest merge does NOT touch trading_app tables."""
import sys
from pathlib import Path
import inspect

import pytest
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_merge_does_not_drop_trading_tables(tmp_path):
    """merge_bars_only() must NOT drop trading_app tables."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Create bars_1m + a fake trading table
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ, symbol TEXT, source_symbol TEXT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.execute("""
        CREATE TABLE validated_setups (
            strategy_id TEXT PRIMARY KEY, instrument TEXT NOT NULL
        )
    """)
    con.execute("""
        INSERT INTO validated_setups VALUES ('test_strat', 'MGC')
    """)
    con.commit()
    con.close()

    # Import and run the safe merge
    from scripts.infra.run_parallel_ingest import merge_bars_only
    merge_bars_only(db_path=db_path, temp_dbs=[])

    # Trading table must survive
    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
    con.close()
    assert count == 1, "merge_bars_only() destroyed trading_app data!"


def test_merge_bars_only_has_no_force_rebuild_param():
    """merge_bars_only() must NOT have a force_rebuild parameter."""
    from scripts.infra.run_parallel_ingest import merge_bars_only
    sig = inspect.signature(merge_bars_only)
    assert "force_rebuild" not in sig.parameters, \
        "merge_bars_only() should not have force_rebuild — that's in main()"


def test_no_merge_all_function():
    """The old merge_all() function must be gone."""
    import scripts.infra.run_parallel_ingest as mod
    assert not hasattr(mod, "merge_all"), \
        "merge_all() still exists — must be replaced by merge_bars_only()"
