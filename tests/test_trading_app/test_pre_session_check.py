"""Tests for pre_session_check — CRITICAL-1: corrupt HWM file must FAIL, not pass."""

import json
from pathlib import Path
from unittest.mock import patch

from trading_app.pre_session_check import check_dd_circuit_breaker


def _make_hwm_file(tmp_path: Path, data: dict, filename: str = "account_hwm_TEST123.json") -> Path:
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    f = state_dir / filename
    f.write_text(json.dumps(data))
    return state_dir


def test_corrupt_hwm_file_returns_fail(tmp_path):
    """Corrupt/unreadable HWM state file MUST return NO-GO (fail-closed)."""
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = state_dir / "account_hwm_TEST123.json"
    corrupt_file.write_text("{{{{not valid json at all")

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is False, f"Corrupt HWM file should FAIL (got ok=True): {msg}"
    assert "BLOCKED" in msg
    assert "unreadable" in msg


def test_empty_hwm_file_returns_fail(tmp_path):
    """Empty HWM file = unreadable = FAIL."""
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "account_hwm_TEST123.json").write_text("")

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is False, f"Empty HWM file should FAIL: {msg}"
    assert "BLOCKED" in msg


def test_valid_hwm_clear_returns_pass(tmp_path):
    """Valid HWM with no halt → pass."""
    state_dir = _make_hwm_file(tmp_path, {
        "account_id": "TEST123",
        "halt_triggered": False,
        "dd_used_dollars": 500,
        "dd_limit_dollars": 2000,
    })

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is True
    assert "clear" in msg.lower()


def test_valid_hwm_halted_returns_fail(tmp_path):
    """Valid HWM with halt_triggered=True → FAIL."""
    state_dir = _make_hwm_file(tmp_path, {
        "account_id": "TEST123",
        "halt_triggered": True,
        "dd_used_dollars": 2100,
        "dd_limit_dollars": 2000,
    })

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is False
    assert "HALT" in msg


def test_no_hwm_files_returns_pass(tmp_path):
    """No HWM files at all (first session) → pass."""
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is True
    assert "first session" in msg.lower() or "No DD tracker" in msg


def test_corrupt_file_skipped_returns_pass(tmp_path):
    """Files with CORRUPT in the name are skipped (not treated as errors)."""
    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "account_hwm_TEST123.CORRUPT.json").write_text("garbage")

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    # CORRUPT file ignored, no other files → first session path
    assert ok is True
