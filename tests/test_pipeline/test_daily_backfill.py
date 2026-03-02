import duckdb
import os
import tempfile
from datetime import date
from unittest.mock import patch

import pytest


def _make_db(rows=None):
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    os.unlink(f.name)  # DuckDB must create the file itself; 0-byte file causes IOException
    con = duckdb.connect(f.name)
    con.execute("CREATE TABLE bars_1m (ts_event TIMESTAMPTZ, symbol VARCHAR)")
    if rows:
        for row in rows:
            con.execute("INSERT INTO bars_1m VALUES (?, ?)", row)
    con.close()
    return f.name


# ── A1 tests ─────────────────────────────────────────────────────────────────

def test_get_last_ingested_date_empty():
    from pipeline.daily_backfill import get_last_ingested_date

    db = _make_db()
    try:
        assert get_last_ingested_date(db, "MGC") is None
    finally:
        os.unlink(db)


def test_get_last_ingested_date_with_data():
    from pipeline.daily_backfill import get_last_ingested_date

    db = _make_db([("2026-02-28 15:00:00+00", "MGC")])
    try:
        result = get_last_ingested_date(db, "MGC")
        assert result.date().isoformat() == "2026-02-28"
    finally:
        os.unlink(db)


def test_is_up_to_date_true():
    from pipeline.daily_backfill import is_up_to_date

    db = _make_db([("2026-03-01 15:00:00+00", "MGC")])
    try:
        assert is_up_to_date(db, "MGC", date(2026, 3, 1)) is True
    finally:
        os.unlink(db)


def test_is_up_to_date_false():
    from pipeline.daily_backfill import is_up_to_date

    db = _make_db([("2026-02-28 15:00:00+00", "MGC")])
    try:
        assert is_up_to_date(db, "MGC", date(2026, 3, 1)) is False
    finally:
        os.unlink(db)


# ── A2 tests ──────────────────────────────────────────────────────────────────

def test_run_backfill_skips_when_current():
    from pipeline.daily_backfill import run_backfill_for_instrument

    db = _make_db([("2026-03-01 15:00:00+00", "MGC")])
    try:
        with patch("pipeline.daily_backfill._run") as mock_run:
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
            mock_run.assert_not_called()
    finally:
        os.unlink(db)


def test_run_backfill_calls_pipeline_when_stale():
    from pipeline.daily_backfill import run_backfill_for_instrument

    db = _make_db()  # empty
    try:
        with patch("pipeline.daily_backfill._run") as mock_run:
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
            assert mock_run.call_count >= 4  # ingest, 5m, daily_features, outcomes
    finally:
        os.unlink(db)
