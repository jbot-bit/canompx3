"""Tests for scripts/reports/monitor_q4_band_shadow.py.

Focus: fail-closed guarantees and eligibility computation. The monitor is
append-only, read-only and must never regress into capital-action territory.
"""

from __future__ import annotations

import csv
import json

import pytest

from scripts.tools import monitor_q4_band_shadow as m


def _make_breakpoints_payload(*, pre_reg_sha: str = "aa9999a7", recalibration_allowed: bool = False) -> dict:
    return {
        "pre_reg_sha": pre_reg_sha,
        "instrument": "MNQ",
        "recalibration_allowed": recalibration_allowed,
        "lanes": {
            "MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long": {
                "session": "NYSE_OPEN",
                "direction": "long",
                "p60": 1.5,
                "p80": 2.5,
                "n_is_rows": 500,
            },
            "MNQ_EUROPE_FLOW_O5_E2_CB1_RR1.5_long": {
                "session": "EUROPE_FLOW",
                "direction": "long",
                "p60": 1.4,
                "p80": 2.3,
                "n_is_rows": 400,
            },
            "MNQ_CME_PRECLOSE_O5_E2_CB1_RR1.5_long": {
                "session": "CME_PRECLOSE",
                "direction": "long",
                "p60": 1.3,
                "p80": 1.9,
                "n_is_rows": 600,
            },
        },
    }


def _make_allocator_payload() -> dict:
    return {
        "lanes": [
            {
                "strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
                "instrument": "MNQ",
                "orb_label": "EUROPE_FLOW",
                "status": "DEPLOY",
            },
            {
                "strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12",
                "instrument": "MNQ",
                "orb_label": "NYSE_OPEN",
                "status": "DEPLOY",
            },
            {
                "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5",
                "instrument": "MNQ",
                "orb_label": "COMEX_SETTLE",
                "status": "MONITOR",  # not DEPLOY
            },
        ],
    }


def test_load_breakpoints_fails_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "BREAKPOINTS_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="FAIL-CLOSED: frozen breakpoints"):
        m._load_breakpoints()


def test_load_breakpoints_fails_on_sha_mismatch(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_make_breakpoints_payload(pre_reg_sha="wrong_sha")))
    monkeypatch.setattr(m, "BREAKPOINTS_PATH", bad)
    with pytest.raises(RuntimeError, match="pre_reg_sha.*does not match"):
        m._load_breakpoints()


def test_load_breakpoints_fails_on_recalibration_flag(tmp_path, monkeypatch):
    recal = tmp_path / "recal.json"
    recal.write_text(json.dumps(_make_breakpoints_payload(recalibration_allowed=True)))
    monkeypatch.setattr(m, "BREAKPOINTS_PATH", recal)
    with pytest.raises(RuntimeError, match="recalibration_allowed=True"):
        m._load_breakpoints()


def test_load_breakpoints_returns_specs_and_payload(tmp_path, monkeypatch):
    good = tmp_path / "good.json"
    payload = _make_breakpoints_payload()
    good.write_text(json.dumps(payload))
    monkeypatch.setattr(m, "BREAKPOINTS_PATH", good)
    specs, returned_payload = m._load_breakpoints()
    assert set(specs.keys()) == set(payload["lanes"].keys())
    assert returned_payload["instrument"] == "MNQ"
    nyse = specs["MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long"]
    assert nyse.session == "NYSE_OPEN"
    assert nyse.direction == "long"
    assert nyse.p60 == 1.5
    assert nyse.p80 == 2.5


def test_load_deployed_sessions_fails_on_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LANE_ALLOCATION_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        m._load_deployed_sessions()


def test_load_deployed_sessions_filters_to_deploy_status(tmp_path, monkeypatch):
    alloc = tmp_path / "alloc.json"
    alloc.write_text(json.dumps(_make_allocator_payload()))
    monkeypatch.setattr(m, "LANE_ALLOCATION_PATH", alloc)
    result = m._load_deployed_sessions()
    assert result == {("MNQ", "EUROPE_FLOW"), ("MNQ", "NYSE_OPEN")}
    # MONITOR-status lane must be excluded
    assert ("MNQ", "COMEX_SETTLE") not in result


def test_load_deployed_sessions_fails_on_missing_keys(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"lanes": [{"status": "DEPLOY"}]}))
    monkeypatch.setattr(m, "LANE_ALLOCATION_PATH", bad)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED: lane entry missing"):
        m._load_deployed_sessions()


def test_compose_row_in_q4_band():
    spec = m.LaneSpec(
        lane_id="MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long",
        session="NYSE_OPEN",
        direction="long",
        p60=1.5,
        p80=2.5,
    )
    row = m._compose_row(
        spec=spec,
        td_iso="2026-04-15",
        rel_vol=2.0,
        eligible=True,
        source_commit="abc12345",
        written_at_utc="2026-04-15T00:00:00+00:00",
    )
    assert row["in_q4_band"] == 1
    assert row["eligible_flag"] == 1
    assert row["shadow_action"] == "LOG_ONLY"
    assert row["breach_flag"] == 0
    assert row["pre_reg_sha"] == m.PREREG_SHA


def test_compose_row_outside_q4_band():
    spec = m.LaneSpec(
        lane_id="x",
        session="NYSE_OPEN",
        direction="long",
        p60=1.5,
        p80=2.5,
    )
    row = m._compose_row(
        spec=spec, td_iso="2026-04-15", rel_vol=3.0, eligible=True, source_commit="abc", written_at_utc="t"
    )
    assert row["in_q4_band"] == 0


def test_compose_row_breach_on_missing_rel_vol():
    spec = m.LaneSpec(
        lane_id="x",
        session="NYSE_OPEN",
        direction="long",
        p60=1.5,
        p80=2.5,
    )
    row = m._compose_row(
        spec=spec, td_iso="2026-04-15", rel_vol=float("nan"), eligible=True, source_commit="abc", written_at_utc="t"
    )
    assert row["breach_flag"] == 1
    assert row["breach_reason"] == "rel_vol_missing"
    assert row["in_q4_band"] == 0
    assert row["rel_vol_observed"] == ""


def test_ledger_header_mismatch_fails_closed(tmp_path, monkeypatch):
    """If an existing ledger has a wrong header, we must NOT silently overwrite."""
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("wrong,columns,here\n")
    monkeypatch.setattr(m, "LEDGER_PATH", ledger)
    with pytest.raises(RuntimeError, match="ledger header does not match schema"):
        m._ensure_ledger_header()


def test_ensure_ledger_header_creates_new_file(tmp_path, monkeypatch):
    ledger = tmp_path / "new.csv"
    monkeypatch.setattr(m, "LEDGER_PATH", ledger)
    m._ensure_ledger_header()
    assert ledger.exists()
    with ledger.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == m.CSV_COLUMNS


def test_existing_ledger_keys_reads_prior_rows(tmp_path, monkeypatch):
    ledger = tmp_path / "exist.csv"
    with ledger.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=m.CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerow(
            {k: "" for k in m.CSV_COLUMNS}
            | {
                "trading_day": "2026-04-15",
                "lane_id": "MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long",
            }
        )
    monkeypatch.setattr(m, "LEDGER_PATH", ledger)
    keys = m._existing_ledger_keys()
    assert ("2026-04-15", "MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long") in keys
    assert len(keys) == 1


def test_eligibility_uses_instrument_session_pair(tmp_path, monkeypatch):
    """The eligibility gate must compare (instrument, session) tuples, not
    bare lane_id strings — frozen lane_ids include RR/direction suffixes the
    allocator rows do not carry."""
    # Allocator has 2 DEPLOY'd pairs: (MNQ, EUROPE_FLOW) and (MNQ, NYSE_OPEN)
    alloc = tmp_path / "alloc.json"
    alloc.write_text(json.dumps(_make_allocator_payload()))
    # Breakpoints file has 3 frozen lanes: NYSE_OPEN_long, EUROPE_FLOW_long, CME_PRECLOSE_long
    bp = tmp_path / "bp.json"
    bp.write_text(json.dumps(_make_breakpoints_payload()))
    monkeypatch.setattr(m, "BREAKPOINTS_PATH", bp)
    monkeypatch.setattr(m, "LANE_ALLOCATION_PATH", alloc)
    specs, payload = m._load_breakpoints()
    deploy_set = m._load_deployed_sessions()
    instrument = payload["instrument"]
    eligible_ids = {lid for lid, spec in specs.items() if (instrument, spec.session) in deploy_set}
    # NYSE_OPEN + EUROPE_FLOW → eligible; CME_PRECLOSE is MONITOR (not DEPLOY) → excluded
    assert "MNQ_NYSE_OPEN_O5_E2_CB1_RR1.5_long" in eligible_ids
    assert "MNQ_EUROPE_FLOW_O5_E2_CB1_RR1.5_long" in eligible_ids
    assert "MNQ_CME_PRECLOSE_O5_E2_CB1_RR1.5_long" not in eligible_ids
