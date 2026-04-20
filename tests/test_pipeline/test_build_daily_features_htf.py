"""
Regression test for HTF seed-loading in build_daily_features.

Root cause (fixed in this commit): build_daily_features() called
_apply_htf_level_fields(rows) on only the newly-built rows. On narrow
incremental ranges (e.g., 2-day nightly backfill), `rows` had no prior
week/month history — so every prev_week_* / prev_month_* field was NULL.

Incident: MNQ 2026-04-17 and 2026-04-19 nightly backfills left those rows
with NULL HTF values, surfacing as Check 59 `stale_miss` in
pipeline/check_drift.py. Fix was wired via `_load_htf_seed_rows()` which
prepends the prior calendar month as seed input to
`_apply_htf_level_fields`.

These tests pin:
  1. The seed loader returns [first-of-prior-calendar-month, start_day)
     rows in trading_day order with the 5 canonical daily_* fields.
  2. Empty-seed case (fresh install) returns [] without error.
  3. orb_minutes=5 filter — HTF is orb-agnostic, seed must be single-slice.
  4. End-to-end: seed + new_rows → new_rows emerge from
     _apply_htf_level_fields with non-NULL prev_week_high /
     prev_month_high when the seed covers the required prior period.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date

import duckdb
import pytest


HTF_SCHEMA = """
CREATE TABLE daily_features (
    symbol VARCHAR,
    trading_day DATE,
    orb_minutes INTEGER,
    daily_open DOUBLE,
    daily_high DOUBLE,
    daily_low DOUBLE,
    daily_close DOUBLE,
    prev_week_high DOUBLE,
    prev_week_low DOUBLE,
    prev_week_open DOUBLE,
    prev_week_close DOUBLE,
    prev_week_range DOUBLE,
    prev_week_mid DOUBLE,
    prev_month_high DOUBLE,
    prev_month_low DOUBLE,
    prev_month_open DOUBLE,
    prev_month_close DOUBLE,
    prev_month_range DOUBLE,
    prev_month_mid DOUBLE
)
"""


def _make_db():
    """Create an isolated DuckDB with the minimal daily_features slice."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # DuckDB must create the file itself
    con = duckdb.connect(path)
    con.execute(HTF_SCHEMA)
    return con, path


def _seed_month(con, symbol: str, start: date, days: int, base_price: float = 100.0):
    """Insert `days` daily_features rows (all three orb_minutes) starting at `start`."""
    for i in range(days):
        td = date.fromordinal(start.toordinal() + i)
        o = base_price + i
        h = o + 2.0
        lo = o - 1.5
        c = o + 0.5
        for om in (5, 15, 30):
            con.execute(
                """INSERT INTO daily_features
                     (symbol, trading_day, orb_minutes,
                      daily_open, daily_high, daily_low, daily_close)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [symbol, td, om, o, h, lo, c],
            )


# ── _load_htf_seed_rows contract ─────────────────────────────────────────────


def test_load_htf_seed_returns_prior_month():
    from pipeline.build_daily_features import _load_htf_seed_rows

    con, path = _make_db()
    try:
        _seed_month(con, "MNQ", date(2026, 3, 1), days=45)

        # start_day = 2026-04-15 → prior-month key = 2026-03-01
        # Expect rows in [2026-03-01, 2026-04-15) = 45 days
        seed = _load_htf_seed_rows(con, "MNQ", date(2026, 4, 15))
        assert len(seed) == 45
        assert seed[0]["trading_day"] == date(2026, 3, 1)
        assert seed[-1]["trading_day"] == date(2026, 4, 14)
        # All five canonical daily_* fields present
        for r in seed:
            assert set(r.keys()) == {
                "trading_day",
                "daily_open",
                "daily_high",
                "daily_low",
                "daily_close",
            }
    finally:
        con.close()
        os.unlink(path)


def test_load_htf_seed_empty_on_fresh_install():
    from pipeline.build_daily_features import _load_htf_seed_rows

    con, path = _make_db()
    try:
        seed = _load_htf_seed_rows(con, "MNQ", date(2026, 4, 15))
        assert seed == []
    finally:
        con.close()
        os.unlink(path)


def test_load_htf_seed_filters_orb_minutes_5():
    """HTF is orb-agnostic; seed must be a single representative slice."""
    from pipeline.build_daily_features import _load_htf_seed_rows

    con, path = _make_db()
    try:
        _seed_month(con, "MNQ", date(2026, 3, 1), days=10)
        # seed_month inserts all 3 orb_minutes for each day; we want 10, not 30
        seed = _load_htf_seed_rows(con, "MNQ", date(2026, 4, 1))
        assert len(seed) == 10
    finally:
        con.close()
        os.unlink(path)


def test_load_htf_seed_scopes_to_symbol():
    from pipeline.build_daily_features import _load_htf_seed_rows

    con, path = _make_db()
    try:
        _seed_month(con, "MNQ", date(2026, 3, 1), days=10)
        _seed_month(con, "MES", date(2026, 3, 1), days=10)

        mnq_seed = _load_htf_seed_rows(con, "MNQ", date(2026, 4, 1))
        mes_seed = _load_htf_seed_rows(con, "MES", date(2026, 4, 1))
        assert len(mnq_seed) == 10
        assert len(mes_seed) == 10
        assert all(r["daily_open"] is not None for r in mnq_seed)
    finally:
        con.close()
        os.unlink(path)


# ── End-to-end: seed + new rows → HTF populated ──────────────────────────────


def test_seed_bridges_narrow_incremental_range():
    """The Check 59 regression guard: 2-day new-row range with seed populates HTF."""
    from pipeline.build_daily_features import (
        _apply_htf_level_fields,
        _load_htf_seed_rows,
    )

    con, path = _make_db()
    try:
        # Seed March 1 → April 14 (45 days, covers prev_week + prev_month for April 15)
        _seed_month(con, "MNQ", date(2026, 3, 1), days=45)

        # Simulate two newly-computed rows (trading_days 2026-04-15, 2026-04-16)
        new_rows = [
            {
                "trading_day": date(2026, 4, 15),
                "daily_open": 145.0,
                "daily_high": 147.0,
                "daily_low": 143.0,
                "daily_close": 145.5,
            },
            {
                "trading_day": date(2026, 4, 16),
                "daily_open": 145.5,
                "daily_high": 148.0,
                "daily_low": 144.0,
                "daily_close": 147.0,
            },
        ]

        seed = _load_htf_seed_rows(con, "MNQ", date(2026, 4, 15))
        _apply_htf_level_fields(seed + new_rows)

        # New rows must now have non-NULL HTF (the whole point of the fix)
        for r in new_rows:
            assert r["prev_week_high"] is not None, f"prev_week_high NULL on {r['trading_day']} — Check 59 regression"
            assert r["prev_week_low"] is not None
            assert r["prev_month_high"] is not None, f"prev_month_high NULL on {r['trading_day']} — Check 59 regression"
            assert r["prev_month_low"] is not None
            assert r["prev_month_range"] == round(r["prev_month_high"] - r["prev_month_low"], 4)
    finally:
        con.close()
        os.unlink(path)


def test_no_seed_reproduces_original_bug():
    """Without seed, narrow range produces NULL HTF — proves the fix is load-bearing."""
    from pipeline.build_daily_features import _apply_htf_level_fields

    new_rows = [
        {
            "trading_day": date(2026, 4, 15),
            "daily_open": 145.0,
            "daily_high": 147.0,
            "daily_low": 143.0,
            "daily_close": 145.5,
        },
        {
            "trading_day": date(2026, 4, 16),
            "daily_open": 145.5,
            "daily_high": 148.0,
            "daily_low": 144.0,
            "daily_close": 147.0,
        },
    ]

    _apply_htf_level_fields(new_rows)

    # Pre-fix behavior: NULL HTF (this is what Check 59 caught in prod)
    for r in new_rows:
        assert r["prev_week_high"] is None
        assert r["prev_month_high"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
