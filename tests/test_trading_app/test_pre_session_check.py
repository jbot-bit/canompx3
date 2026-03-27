"""Tests for pre_session_check — HWM fail-closed + manual halt."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from trading_app.pre_session_check import check_dd_circuit_breaker, check_manual_halt


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
    state_dir = _make_hwm_file(
        tmp_path,
        {
            "account_id": "TEST123",
            "halt_triggered": False,
            "dd_used_dollars": 500,
            "dd_limit_dollars": 2000,
        },
    )

    with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
        ok, msg = check_dd_circuit_breaker()

    assert ok is True
    assert "clear" in msg.lower()


def test_valid_hwm_halted_returns_fail(tmp_path):
    """Valid HWM with halt_triggered=True → FAIL."""
    state_dir = _make_hwm_file(
        tmp_path,
        {
            "account_id": "TEST123",
            "halt_triggered": True,
            "dd_used_dollars": 2100,
            "dd_limit_dollars": 2000,
        },
    )

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


# ── Manual halt tests ────────────────────────────────────────────────────


class TestManualHalt:
    """Manual trading halt via state file."""

    def test_no_halt_file(self, tmp_path):
        """No halt file → pass."""
        halt = tmp_path / "halt_trading.json"
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is True
        assert "No manual halt" in msg

    def test_active_halt(self, tmp_path):
        """Active halt file → NO-GO with reason."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text(json.dumps({
            "active": True,
            "reason": "Down $900, sitting out",
            "expires": (date.today() + timedelta(days=1)).isoformat(),
        }))
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is False
        assert "MANUAL HALT" in msg
        assert "Down $900" in msg

    def test_inactive_halt(self, tmp_path):
        """Halt file with active=false → pass."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text(json.dumps({"active": False, "reason": "cleared"}))
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is True
        assert "inactive" in msg

    def test_expired_halt(self, tmp_path):
        """Halt file with past expiry → auto-resume (pass)."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text(json.dumps({
            "active": True,
            "reason": "yesterday",
            "expires": (date.today() - timedelta(days=1)).isoformat(),
        }))
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is True
        assert "expired" in msg.lower()

    def test_corrupt_halt_file(self, tmp_path):
        """Corrupt halt file → pass with warning (can't verify halt)."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text("{{{invalid json")
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is True
        assert "unreadable" in msg.lower()

    def test_halt_no_expiry(self, tmp_path):
        """Halt with no expiry field → blocks indefinitely."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text(json.dumps({"active": True, "reason": "indefinite"}))
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is False
        assert "indefinite" in msg
