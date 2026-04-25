import os
import tempfile
from datetime import date
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
