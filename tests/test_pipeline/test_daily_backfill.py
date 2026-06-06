import os
import tempfile
from datetime import UTC, date, datetime
from unittest.mock import patch

import duckdb
import pytest


def _make_db(rows=None):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        pass
    os.unlink(f.name)  # DuckDB must create the file itself; 0-byte file causes IOException
    con = duckdb.connect(f.name)
    con.execute("CREATE TABLE bars_1m (ts_utc TIMESTAMPTZ, symbol VARCHAR)")
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


# ── Option W tests: in-progress trading day must NOT count as up-to-date ───────
#
# Root cause (audit 2026-06-06): the live bridge writes partial bars for the
# CURRENT trading day; that advances MAX(ts_utc) so is_up_to_date returned True
# and suppressed the Databento full-day ingest. Completeness is NOT inferable
# from bars (a 60-bar partial is identical to a 60-bar quiet complete day), so
# the guard is WALL-CLOCK: a day counts as up-to-date only if its trading-day
# UTC window has fully elapsed. Past days (incl. CME half-days) are unaffected.


def test_is_up_to_date_false_for_in_progress_day():
    """A day whose trading-day window has NOT yet elapsed is never up-to-date,
    even if a (live-partial) bar already advanced MAX(ts_utc)."""
    from pipeline.daily_backfill import is_up_to_date

    # Bar exists at the start of 2026-06-05 (live partial), but "now" is during
    # that same trading day → window not elapsed → must be False.
    db = _make_db([("2026-06-04 23:30:00+00", "MGC")])  # ~09:30 Brisbane on td 2026-06-05
    now = datetime(2026, 6, 5, 2, 0, tzinfo=UTC)  # mid-trading-day 2026-06-05
    try:
        assert is_up_to_date(db, "MGC", date(2026, 6, 5), now=now) is False
    finally:
        os.unlink(db)


def test_is_up_to_date_true_for_elapsed_past_day():
    """A fully-elapsed past day with bars is up-to-date (Option W must not
    regress the normal past-day case)."""
    from pipeline.daily_backfill import is_up_to_date

    db = _make_db([("2026-03-01 15:00:00+00", "MGC")])
    now = datetime(2026, 6, 5, 2, 0, tzinfo=UTC)  # long after 2026-03-01
    try:
        assert is_up_to_date(db, "MGC", date(2026, 3, 1), now=now) is True
    finally:
        os.unlink(db)


def test_is_up_to_date_false_when_target_day_itself_has_no_bars():
    """Global MAX(ts_utc) must NOT mask a missing target day. A live partial for
    a LATER trading day advances global MAX, but the target day (here 06-04) has
    no real bars → must be False so Databento ingest runs for it. This closes the
    original Gap A (max-ts-alone) that Option W's wall-clock guard alone leaves."""
    from pipeline.daily_backfill import is_up_to_date

    # Bar maps to trading_day 2026-06-05 (~09:30 Brisbane = 23:30 UTC on 06-04),
    # but its CALENDAR date is 06-04. Target 06-04 has NO real bars of its own.
    db = _make_db([("2026-06-04 23:30:00+00", "MGC")])
    now = datetime(2026, 6, 5, 2, 0, tzinfo=UTC)  # 06-04 window has elapsed
    try:
        assert is_up_to_date(db, "MGC", date(2026, 6, 4), now=now) is False
    finally:
        os.unlink(db)


def test_is_up_to_date_halfday_not_excluded_by_wallclock():
    """A CME half-day in the past is up-to-date — Option W keys on wall-clock,
    not on whether bars reached the last-session window (which half-days don't)."""
    from pipeline.daily_backfill import is_up_to_date

    # Half-day bars stop early (18:14 UTC on the half-day), but the day is in the
    # past → window elapsed → up-to-date.
    db = _make_db([("2025-11-28 18:14:00+00", "MES")])
    now = datetime(2026, 6, 5, 2, 0, tzinfo=UTC)
    try:
        assert is_up_to_date(db, "MES", date(2025, 11, 28), now=now) is True
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
        with (
            patch("pipeline.daily_backfill._run") as mock_run,
            patch("pipeline.daily_backfill._patch_atr_percentiles"),
        ):
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
            assert mock_run.call_count >= 4  # ingest, 5m, daily_features, outcomes
    finally:
        os.unlink(db)


# ── A3 tests: subprocess invocation form (regression guard) ───────────────────
#
# All four pipeline sub-stages import via `from pipeline.X import Y` /
# `from trading_app.X import Y`. That package-import pattern requires the
# project root on sys.path, which happens when Python is launched with
# `-m <module>` but NOT when launched as `python path/to/file.py` (the
# script's directory is prepended to sys.path instead).
#
# Incident (2026-04-20): daily_backfill used `python pipeline/ingest_dbn.py`
# form, which silently failed with ModuleNotFoundError: No module named
# 'pipeline'. The failure cascaded into empty backfills and a stale
# gold.db. Fix + this regression guard landed together; do NOT revert
# to the file-path form without reviewing this test.


def test_run_backfill_uses_module_invocation_form():
    """Every subprocess.run call must use `-m <module>` form, never a path."""
    from pipeline.daily_backfill import run_backfill_for_instrument

    db = _make_db()  # empty → triggers full backfill path
    captured: list[list[str]] = []

    def _fake_run(cmd, desc):
        captured.append(list(cmd))

    try:
        with (
            patch("pipeline.daily_backfill._run", side_effect=_fake_run),
            patch("pipeline.daily_backfill._patch_atr_percentiles"),
        ):
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
    finally:
        os.unlink(db)

    assert len(captured) == 4, f"Expected 4 subprocess calls, got {len(captured)}"

    # Each command must have shape: [python, "-m", <module>, *args]
    for cmd in captured:
        assert len(cmd) >= 3, f"cmd too short: {cmd}"
        assert cmd[1] == "-m", (
            f"Subprocess invocation must use `-m <module>` form (package imports "
            f"break with `python path/to/file.py`). Got: {cmd[:3]}"
        )
        module = cmd[2]
        assert module.startswith(("pipeline.", "trading_app.")), (
            f"Module target must be pipeline.X or trading_app.X. Got: {module!r}"
        )
        # Guard against accidental path-separator leak.
        for sep in ("/", "\\"):
            assert sep not in module, f"Module target contains path separator {sep!r}: {module!r}"
        # Guard against `.py` suffix (script-path leak).
        assert not module.endswith(".py"), f"Module target ends with .py: {module!r}"

    # Assert each expected target is present exactly once.
    expected_modules = {
        "pipeline.ingest_dbn",
        "pipeline.build_bars_5m",
        "pipeline.build_daily_features",
        "trading_app.outcome_builder",
    }
    actual_modules = {cmd[2] for cmd in captured}
    assert actual_modules == expected_modules, (
        f"Missing or unexpected module targets.\n"
        f"  Expected: {sorted(expected_modules)}\n"
        f"  Got:      {sorted(actual_modules)}"
    )
