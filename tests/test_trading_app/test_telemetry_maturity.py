"""Tests for the live-telemetry maturity gate.

Mutation proofs:
- 29 distinct trading_day SESSION_START records for MNQ -> UNVERIFIED.
- 30 distinct trading_day SESSION_START records for MNQ -> MATURE.
- Records for a different instrument do not count toward MNQ maturity.
- Empty / missing signals dir -> fail-closed UNVERIFIED.
- Malformed jsonl lines are skipped without raising.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.live.telemetry_maturity import (
    MIN_TELEMETRY_TRADING_DAYS,
    VERDICT_MATURE,
    VERDICT_UNVERIFIED,
    evaluate_telemetry_maturity,
    iter_known_uptime_record_types,
)


def _write_signals(signals_dir: Path, records_by_day: dict) -> None:
    """Write one live_signals_<day>.jsonl per (day, records) item.

    Each value is a list of dicts that become jsonl lines.
    """
    signals_dir.mkdir(parents=True, exist_ok=True)
    for day_iso, records in records_by_day.items():
        path = signals_dir / f"live_signals_{day_iso}.jsonl"
        lines = [json.dumps(r) for r in records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _session_start_record(instrument: str, day_offset: int, base_iso: str = "2026-05-01T20:00:00+00:00") -> dict:
    """Build a SESSION_START record offset N days from a base UTC date."""
    base = datetime.fromisoformat(base_iso)
    ts = (base + timedelta(days=day_offset)).isoformat()
    return {
        "ts": ts,
        "instrument": instrument,
        "type": "SESSION_START",
        "contract": f"CON.F.US.{instrument}.M26",
        "mode": "signal_only",
    }


def test_floor_constant_is_30():
    """The bright-line floor is 30 -- per Criterion 8 + RULE 3.2/3.3."""
    assert MIN_TELEMETRY_TRADING_DAYS == 30


def test_29_distinct_days_returns_unverified(tmp_path):
    """29 distinct MNQ trading_days falls one short of the gate."""
    records_by_day = {}
    for i in range(29):
        day_iso = (datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        records_by_day[day_iso] = [_session_start_record("MNQ", i)]
    _write_signals(tmp_path, records_by_day)

    report = evaluate_telemetry_maturity(tmp_path, instrument="MNQ")
    assert report.verdict == VERDICT_UNVERIFIED
    assert report.n_unique_trading_days == 29
    assert report.min_required == 30
    assert report.signal_files_scanned == 29
    assert report.records_qualifying == 29
    assert "need 1 more" in report.justification()


def test_30_distinct_days_returns_mature(tmp_path):
    """30 distinct MNQ trading_days passes the bright-line gate."""
    records_by_day = {}
    for i in range(30):
        day_iso = (datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        records_by_day[day_iso] = [_session_start_record("MNQ", i)]
    _write_signals(tmp_path, records_by_day)

    report = evaluate_telemetry_maturity(tmp_path, instrument="MNQ")
    assert report.verdict == VERDICT_MATURE
    assert report.n_unique_trading_days == 30
    assert report.is_mature
    assert "telemetry mature" in report.justification()


def test_other_instrument_records_do_not_count_for_mnq(tmp_path):
    """Per-instrument scoping: MES records do not satisfy MNQ's floor."""
    records_by_day = {}
    for i in range(30):
        day_iso = (datetime(2026, 5, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        records_by_day[day_iso] = [_session_start_record("MES", i)]
    # Add 5 real MNQ days
    for i in range(5):
        day_iso = (datetime(2026, 6, 1, tzinfo=UTC) + timedelta(days=i)).date().isoformat()
        records_by_day[day_iso] = records_by_day.get(day_iso, []) + [
            _session_start_record("MNQ", i, "2026-06-01T20:00:00+00:00")
        ]
    _write_signals(tmp_path, records_by_day)

    report = evaluate_telemetry_maturity(tmp_path, instrument="MNQ")
    assert report.verdict == VERDICT_UNVERIFIED
    assert report.n_unique_trading_days == 5, "only MNQ-instrument records count"


def test_missing_dir_fails_closed(tmp_path):
    """Pointing the gate at a nonexistent directory MUST return UNVERIFIED, not raise."""
    nonexistent = tmp_path / "does_not_exist"
    report = evaluate_telemetry_maturity(nonexistent, instrument="MNQ")
    assert report.verdict == VERDICT_UNVERIFIED
    assert report.n_unique_trading_days == 0
    assert report.signal_files_scanned == 0


def test_malformed_lines_are_skipped(tmp_path):
    """Malformed jsonl is skipped (debug log), never raises."""
    path = tmp_path / "live_signals_2026-05-15.jsonl"
    path.write_text(
        "this is not json\n"
        + json.dumps(_session_start_record("MNQ", 0))
        + "\n"
        + "{partial json...\n"
        + json.dumps(_session_start_record("MNQ", 1))
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_telemetry_maturity(tmp_path, instrument="MNQ")
    assert report.verdict == VERDICT_UNVERIFIED
    assert report.n_unique_trading_days == 2
    assert report.records_scanned == 4, "malformed lines counted in scanned total"
    assert report.records_qualifying == 2, "only valid lines qualify"


def test_min_trading_days_must_be_positive_int(tmp_path):
    """Bad override values raise -- no silent zero-floor bypass."""
    with pytest.raises(ValueError):
        evaluate_telemetry_maturity(tmp_path, instrument="MNQ", min_trading_days=0)
    with pytest.raises(ValueError):
        evaluate_telemetry_maturity(tmp_path, instrument="MNQ", min_trading_days=-5)


def test_uptime_record_types_include_session_start_and_entries():
    """Doctrine check: SESSION_START and the entry-class records are uptime evidence."""
    known = set(iter_known_uptime_record_types())
    assert "SESSION_START" in known
    assert "SIGNAL_ENTRY" in known
    assert "ORDER_ENTRY" in known
    assert "KILL_SWITCH" in known
