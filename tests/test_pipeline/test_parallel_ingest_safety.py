"""Tests for parallel ingest safety: merge integrity + concurrent write behavior."""
import inspect
import threading
from pathlib import Path

import pytest
import duckdb

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


def test_merge_deduplicates_overlapping_rows(tmp_path):
    """INSERT OR REPLACE in merge_bars_only() deduplicates overlapping rows."""
    db_path = tmp_path / "gold.db"
    temp1 = tmp_path / "temp_0.db"
    temp2 = tmp_path / "temp_1.db"

    # Create gold.db with bars_1m schema
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ, symbol TEXT, source_symbol TEXT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.close()

    # Create two temp DBs with overlapping rows (simulates 2-day overlap)
    for temp_db, rows in [
        (temp1, [
            ("2024-01-01 09:00:00+00", "MGC", "GCG4", 2050, 2051, 2049, 2050, 100),
            ("2024-01-01 09:01:00+00", "MGC", "GCG4", 2050, 2052, 2049, 2051, 110),
        ]),
        (temp2, [
            # Overlapping row (same PK) + new row
            ("2024-01-01 09:01:00+00", "MGC", "GCG4", 2050, 2052, 2049, 2051, 110),
            ("2024-01-02 09:00:00+00", "MGC", "GCG4", 2055, 2056, 2054, 2055, 120),
        ]),
    ]:
        tc = duckdb.connect(str(temp_db))
        tc.execute("""
            CREATE TABLE bars_1m (
                ts_utc TIMESTAMPTZ, symbol TEXT, source_symbol TEXT,
                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
                PRIMARY KEY (symbol, ts_utc)
            )
        """)
        for r in rows:
            tc.execute(
                "INSERT INTO bars_1m VALUES (?, ?, ?, ?, ?, ?, ?, ?)", r
            )
        tc.close()

    from scripts.infra.run_parallel_ingest import merge_bars_only
    merge_bars_only(db_path=db_path, temp_dbs=[temp1, temp2])

    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    con.close()

    # 3 unique rows (the overlap is deduplicated)
    assert count == 3, f"Expected 3 unique rows after merge, got {count}"


def test_concurrent_duckdb_writers_error_or_serialize(tmp_path):
    """Two threads writing to the same DuckDB: must raise error OR serialize.

    DuckDB uses a single-writer model. Concurrent writers to the same file
    should either serialize (one waits) or raise a clean error. This test
    documents which behavior DuckDB provides.

    The parallel ingest avoids this by writing to separate temp DBs, but
    this test verifies the assumption that concurrent writes are safely
    handled (not silently corrupting data).
    """
    db_path = tmp_path / "concurrent.db"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE bars_1m (
            ts_utc TIMESTAMPTZ, symbol TEXT, source_symbol TEXT,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            PRIMARY KEY (symbol, ts_utc)
        )
    """)
    con.close()

    errors = []
    rows_written = {"t1": 0, "t2": 0}

    def writer(thread_id, start_offset):
        try:
            wcon = duckdb.connect(str(db_path))
            for i in range(50):
                ts = f"2024-01-01 {9 + (start_offset + i) // 60:02d}:{(start_offset + i) % 60:02d}:00+00"
                wcon.execute(
                    "INSERT OR REPLACE INTO bars_1m VALUES (?, 'MGC', 'GCG4', 2050, 2051, 2049, 2050, ?)",
                    [ts, i],
                )
            wcon.commit()
            rows_written[thread_id] = 50
            wcon.close()
        except Exception as e:
            errors.append((thread_id, type(e).__name__, str(e)))

    t1 = threading.Thread(target=writer, args=("t1", 0))
    t2 = threading.Thread(target=writer, args=("t2", 50))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    # Acceptable outcomes:
    # 1. Both succeed (DuckDB serialized the writes) — all 100 rows present
    # 2. One fails with a clean error — the other's rows are present, no corruption
    # Unacceptable: silent data loss or corruption

    rcon = duckdb.connect(str(db_path), read_only=True)
    count = rcon.execute("SELECT COUNT(*) FROM bars_1m").fetchone()[0]
    rcon.close()

    if not errors:
        # Both succeeded — DuckDB serialized
        assert count == 100, f"Both writers succeeded but only {count}/100 rows present"
    else:
        # At least one failed — verify no data corruption
        assert count > 0, "Both writers failed — no rows written at all"
        assert count <= 100, f"More rows than expected: {count}"
