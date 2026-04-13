"""Tests for pre_session_check — HWM fail-closed + manual halt."""

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.pre_session_check import (
    _resolve_session_lane,
    _resolve_session_lanes,
    check_consistency_rule,
    check_daily_equity,
    check_dd_circuit_breaker,
    check_hwm_tracker,
    check_lane_lifecycle,
    check_manual_halt,
    check_topstep_xfa_aggregate_cap,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, get_profile


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
        halt.write_text(
            json.dumps(
                {
                    "active": True,
                    "reason": "Down $900, sitting out",
                    "expires": (date.today() + timedelta(days=1)).isoformat(),
                }
            )
        )
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
        halt.write_text(
            json.dumps(
                {
                    "active": True,
                    "reason": "yesterday",
                    "expires": (date.today() - timedelta(days=1)).isoformat(),
                }
            )
        )
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is True
        assert "expired" in msg.lower()

    def test_corrupt_halt_file(self, tmp_path):
        """Corrupt halt file → BLOCKED (fail-closed: can't verify halt status)."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text("{{{invalid json")
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is False
        assert "BLOCKED" in msg

    def test_halt_no_expiry(self, tmp_path):
        """Halt with no expiry field → blocks indefinitely."""
        halt = tmp_path / "halt_trading.json"
        halt.write_text(json.dumps({"active": True, "reason": "indefinite"}))
        with patch("trading_app.pre_session_check.HALT_FILE", halt):
            ok, msg = check_manual_halt()
        assert ok is False
        assert "indefinite" in msg


# ── Daily equity / DLL fallback tests ────────────────────────────────


class TestDailyEquityDLLFallback:
    """check_daily_equity must warn on profile lookup failure and use fallback DLL."""

    def test_profile_lookup_failure_warns_and_uses_fallback(self, tmp_path, capsys):
        """When resolve_profile_id raises, function should still return a result
        using the $1000 fallback DLL and emit a warning to stderr."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        eq_file = state_dir / f"equity_{date.today()}.json"
        eq_file.write_text(json.dumps({"date": str(date.today()), "starting_equity": 50000, "current_dd": -200.0}))

        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            with patch("trading_app.prop_profiles.resolve_profile_id", side_effect=KeyError("bad_profile")):
                ok, msg = check_daily_equity(profile_id="bad_profile")

        assert ok is True
        assert "$-200" in msg or "Daily DD" in msg
        stderr = capsys.readouterr().err
        assert "WARNING" in stderr
        assert "bad_profile" in stderr
        assert "fallback" in stderr or "$1,000" in stderr

    def test_profile_lookup_success_uses_real_dll(self, tmp_path):
        """When profile lookup succeeds, function uses the real DLL from ACCOUNT_TIERS."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        eq_file = state_dir / f"equity_{date.today()}.json"
        eq_file.write_text(json.dumps({"date": str(date.today()), "starting_equity": 50000, "current_dd": -800.0}))

        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_daily_equity(profile_id="topstep_50k_mnq_auto")

        # Should pass — real DLL for topstep_50k is ~$1000, DD of $800 is under
        assert ok is True


class TestHWMTrackerFailClosed:
    """check_hwm_tracker must fail-closed on corrupt HWM files."""

    def test_corrupt_hwm_file_blocks_trading(self):
        """Corrupt HWM file → check_hwm_tracker returns False (fail-closed)."""
        from trading_app.pre_session_check import STATE_DIR

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        corrupt = STATE_DIR / "account_hwm_TEST_BAD_DATA.json"
        try:
            corrupt.write_text("{{{not valid json")
            ok, msg = check_hwm_tracker()
            assert ok is False, f"Corrupt HWM file should block trading: {msg}"
            assert "BLOCKED" in msg
        finally:
            if corrupt.exists():
                corrupt.unlink()


def test_resolve_session_lane_ambiguous_profile_requires_strategy_specific_tool():
    # NYSE_OPEN has both MNQ and MES lanes in type_a profiles
    with patch.dict(
        ACCOUNT_PROFILES,
        {"topstep_50k_type_a": replace(get_profile("topstep_50k_type_a"), active=True)},
        clear=False,
    ):
        with pytest.raises(ValueError, match="multiple lanes"):
            _resolve_session_lane("NYSE_OPEN", "topstep_50k_type_a")


def test_resolve_session_lanes_returns_all_shared_session_lanes():
    with patch("trading_app.prop_profiles.resolve_profile_id", return_value="topstep_50k_type_a"):
        profile_id, lanes = _resolve_session_lanes("NYSE_OPEN", profile_id="topstep_50k_type_a")

    assert profile_id == "topstep_50k_type_a"
    assert len(lanes) == 2
    assert {lane["instrument"] for lane in lanes} == {"MES", "MNQ"}


def test_check_consistency_rule_fails_closed_on_ambiguous_active_profiles(monkeypatch):
    monkeypatch.setitem(
        ACCOUNT_PROFILES,
        "tradeify_50k",
        replace(get_profile("tradeify_50k"), active=True),
    )
    ok, msg = check_consistency_rule()
    assert ok is False
    assert "Multiple active execution profiles" in msg


class TestLaneLifecycle:
    def test_blocks_paused_strategy(self):
        lifecycle = {
            "criterion12": {"available": True, "valid": True},
            "strategy_states": {
                "SID_A": {
                    "blocked": True,
                    "block_source": "pause",
                    "block_reason": "Manual pause",
                }
            },
        }
        with patch("trading_app.pre_session_check.read_lifecycle_state", return_value=lifecycle):
            ok, msg = check_lane_lifecycle("SID_A", "topstep_50k_mnq_auto")
        assert ok is False
        assert "Manual pause" in msg

    def test_blocks_sr_alarm_strategy(self):
        lifecycle = {
            "criterion12": {"available": True, "valid": True},
            "strategy_states": {
                "SID_A": {
                    "blocked": True,
                    "block_source": "sr_alarm",
                    "block_reason": "Criterion 12 SR ALARM — manual review required",
                }
            },
        }
        with patch("trading_app.pre_session_check.read_lifecycle_state", return_value=lifecycle):
            ok, msg = check_lane_lifecycle("SID_A", "topstep_50k_mnq_auto")
        assert ok is False
        assert "SR ALARM" in msg

    def test_warns_when_sr_state_is_stale(self):
        lifecycle = {
            "criterion12": {"available": True, "valid": False, "reason": "stale state: 3d old > 2d"},
            "strategy_states": {"SID_A": {"blocked": False, "sr_status": None}},
        }
        with patch("trading_app.pre_session_check.read_lifecycle_state", return_value=lifecycle):
            ok, msg = check_lane_lifecycle("SID_A", "topstep_50k_mnq_auto")
        assert ok is True
        assert "stale/mismatched" in msg

    def test_reviewed_watch_alarm_is_not_blocked(self):
        lifecycle = {
            "criterion12": {"available": True, "valid": True},
            "strategy_states": {
                "SID_A": {
                    "blocked": False,
                    "sr_status": "ALARM",
                    "sr_review_outcome": "watch",
                }
            },
        }
        with patch("trading_app.pre_session_check.read_lifecycle_state", return_value=lifecycle):
            ok, msg = check_lane_lifecycle("SID_A", "topstep_50k_mnq_auto")
        assert ok is True
        assert "reviewed WATCH" in msg

    def test_blocks_when_lifecycle_state_unreadable(self):
        """Fail-closed: unreadable lifecycle state must block trading, not permit it."""
        with patch(
            "trading_app.pre_session_check.read_lifecycle_state",
            side_effect=OSError("state file missing"),
        ):
            ok, msg = check_lane_lifecycle("SID_A", "topstep_50k_mnq_auto")
        assert ok is False
        assert "BLOCKED" in msg
        assert "unavailable" in msg


# ─── F-6: TopStep 5-XFA aggregate cap ────────────────────────────────────
# @canonical-source docs/research-input/topstep/topstep_xfa_parameters.txt
# @verbatim "You can have up to 5 active Express Funded Accounts at the same time."


class TestTopstepXfaAggregateCap:
    """check_topstep_xfa_aggregate_cap (F-6) — sums copies across active TopStep profiles."""

    def _patch_profiles(self, monkeypatch, fake_profiles: dict) -> None:
        """Replace ACCOUNT_PROFILES with a controlled fake set for the duration of one test."""
        from trading_app import prop_profiles

        monkeypatch.setattr(prop_profiles, "ACCOUNT_PROFILES", fake_profiles)

    def _make_profile(self, profile_id: str, firm: str, copies: int, active: bool):
        """Build a real AccountProfile from an existing one and override fields."""
        base = get_profile("topstep_50k_mnq_auto")
        return replace(base, profile_id=profile_id, firm=firm, copies=copies, active=active)

    def test_under_cap_ok(self, monkeypatch):
        fake = {
            "p1": self._make_profile("p1", "topstep", copies=2, active=True),
            "p2": self._make_profile("p2", "topstep", copies=1, active=False),  # inactive ignored
        }
        self._patch_profiles(monkeypatch, fake)
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True
        assert "2/5" in msg

    def test_at_cap_warns(self, monkeypatch):
        fake = {
            "p1": self._make_profile("p1", "topstep", copies=3, active=True),
            "p2": self._make_profile("p2", "topstep", copies=2, active=True),
        }
        self._patch_profiles(monkeypatch, fake)
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True  # at cap = warning, not block
        assert "WARNING" in msg
        assert "5/5" in msg

    def test_over_cap_blocks(self, monkeypatch):
        fake = {
            "p1": self._make_profile("p1", "topstep", copies=5, active=True),
            "p2": self._make_profile("p2", "topstep", copies=2, active=True),
        }
        self._patch_profiles(monkeypatch, fake)
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is False
        assert "BLOCKED" in msg
        assert "7" in msg  # total

    def test_ignores_non_topstep_profiles(self, monkeypatch):
        fake = {
            "ts": self._make_profile("ts", "topstep", copies=3, active=True),
            "tradeify": self._make_profile("tradeify", "tradeify", copies=10, active=True),  # ignored
            "bulenox": self._make_profile("bulenox", "bulenox", copies=10, active=True),  # ignored
        }
        self._patch_profiles(monkeypatch, fake)
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True
        assert "3/5" in msg

    def test_ignores_inactive_topstep_profiles(self, monkeypatch):
        fake = {
            "active": self._make_profile("active", "topstep", copies=2, active=True),
            "inactive_big": self._make_profile("inactive_big", "topstep", copies=10, active=False),
        }
        self._patch_profiles(monkeypatch, fake)
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True
        assert "2/5" in msg

    def test_empty_profiles_ok(self, monkeypatch):
        self._patch_profiles(monkeypatch, {})
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True
        assert "0/5" in msg

    def test_current_repo_profiles_within_cap(self):
        """Sanity check on the actual ACCOUNT_PROFILES — must be ≤ 5 right now."""
        ok, msg = check_topstep_xfa_aggregate_cap()
        assert ok is True, f"Repo state breaches 5-XFA cap: {msg}"
