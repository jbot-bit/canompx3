"""Tests for pre_session_check — HWM fail-closed + manual halt."""

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.pre_session_check import (
    _conditional_overlay_from_lifecycle,
    _resolve_session_lane,
    _resolve_session_lanes,
    check_consistency_rule,
    check_daily_equity,
    check_dd_circuit_breaker,
    check_hwm_tracker,
    check_lane_lifecycle,
    check_manual_halt,
    check_topstep_inactivity_window,
    check_topstep_xfa_aggregate_cap,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, DailyLaneSpec, get_profile


def _shared_nyse_open_profile(profile_id: str):
    """Return a copy of profile_id with MNQ+MES lanes on NYSE_OPEN + active=True.

    2026-04-19 rebuild of topstep_50k_type_a removed the shared-session lanes
    these tests exercise. Inject them synthetically so shared-session logic
    is still testable after the profile rebuild.
    """
    base = ACCOUNT_PROFILES[profile_id]
    non_nyse = tuple(lane for lane in base.daily_lanes if lane.orb_label != "NYSE_OPEN")
    shared = (
        DailyLaneSpec(
            "MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50",
            "MNQ",
            "NYSE_OPEN",
            max_orb_size_pts=117.8,
        ),
        DailyLaneSpec(
            "MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12",
            "MES",
            "NYSE_OPEN",
            max_orb_size_pts=60.0,
        ),
    )
    return replace(
        base,
        active=True,
        daily_lanes=non_nyse + shared,
        allowed_instruments=frozenset(base.allowed_instruments | {"MNQ", "MES"}),
        allowed_sessions=frozenset(base.allowed_sessions | {"NYSE_OPEN"}),
    )


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

    def test_corrupt_hwm_file_blocks_trading(self, tmp_path):
        """Corrupt HWM file → check_hwm_tracker returns False (fail-closed)."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        corrupt = state_dir / "account_hwm_TEST_BAD_DATA.json"
        corrupt.write_text("{{{not valid json")

        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_hwm_tracker()

        assert ok is False, f"Corrupt HWM file should block trading: {msg}"
        assert "BLOCKED" in msg


def test_resolve_session_lane_ambiguous_profile_requires_strategy_specific_tool():
    # Synthetic: NYSE_OPEN shared between MNQ + MES lanes — must raise.
    with patch.dict(
        ACCOUNT_PROFILES,
        {"topstep_50k_type_a": _shared_nyse_open_profile("topstep_50k_type_a")},
        clear=False,
    ):
        with pytest.raises(ValueError, match="multiple lanes"):
            _resolve_session_lane("NYSE_OPEN", "topstep_50k_type_a")


def test_resolve_session_lanes_returns_all_shared_session_lanes():
    with (
        patch.dict(
            ACCOUNT_PROFILES,
            {"topstep_50k_type_a": _shared_nyse_open_profile("topstep_50k_type_a")},
            clear=False,
        ),
        patch("trading_app.prop_profiles.resolve_profile_id", return_value="topstep_50k_type_a"),
    ):
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


def test_conditional_overlay_from_lifecycle_reports_ready_overlay():
    ok, msg = _conditional_overlay_from_lifecycle(
        {
            "conditional_overlays": {
                "available": True,
                "overlays": [
                    {
                        "overlay_id": "pr48_mgc_cont_exec_v1",
                        "valid": True,
                        "status": "ready",
                        "summary": {"ready_count": 3, "row_count": 18},
                    }
                ],
            }
        }
    )

    assert ok is True
    assert "Shadow overlay" in msg
    assert "3/18" in msg


def test_conditional_overlay_from_lifecycle_warns_on_invalid_state():
    ok, msg = _conditional_overlay_from_lifecycle(
        {
            "conditional_overlays": {
                "available": True,
                "overlays": [
                    {
                        "overlay_id": "pr48_mgc_cont_exec_v1",
                        "valid": False,
                        "reason": "missing breakpoint row",
                        "status": "invalid",
                    }
                ],
            }
        }
    )

    assert ok is True
    assert "WARN" in msg
    assert "missing breakpoint row" in msg


def test_conditional_overlay_from_lifecycle_warns_on_invalid_status_even_when_envelope_valid():
    ok, msg = _conditional_overlay_from_lifecycle(
        {
            "conditional_overlays": {
                "available": True,
                "overlays": [
                    {
                        "overlay_id": "pr48_mgc_cont_exec_v1",
                        "valid": True,
                        "reason": "missing breakpoint row",
                        "status": "invalid",
                    }
                ],
            }
        }
    )

    assert ok is True
    assert "WARN" in msg
    assert "missing breakpoint row" in msg


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


# ── Stage 3 — shared-reader delegation + message-format unification ───────


class TestStage3SharedReaderDelegation:
    """Stage 3 of HWM persistence integrity hardening.

    Pins:
      - both pre-session HWM functions delegate to account_hwm_tracker.read_state_file
      - corrupt-state message format unified to BLOCKED <filename>: <reason>
      - boolean behavior on corrupt is unchanged (False — already True before Stage 3)
      - no inline json.loads against account_hwm_*.json paths in pre_session_check
        or weekly_review (canonical owner is account_hwm_tracker.py)
    """

    def test_check_dd_circuit_breaker_calls_shared_reader(self, tmp_path):
        """Mutation guard: re-introducing inline json.loads flips this test."""
        state_dir = _make_hwm_file(
            tmp_path,
            {"account_id": "DELEG", "halt_triggered": False, "dd_used_dollars": 100, "dd_limit_dollars": 2000},
        )
        with (
            patch("trading_app.pre_session_check.STATE_DIR", state_dir),
            patch("trading_app.account_hwm_tracker.read_state_file") as mock_reader,
        ):
            mock_reader.return_value = {"account_id": "DELEG", "halt_triggered": False}
            check_dd_circuit_breaker()
        assert mock_reader.call_count >= 1, "check_dd_circuit_breaker must delegate to read_state_file"

    def test_check_hwm_tracker_calls_shared_reader(self, tmp_path):
        """Mutation guard: re-introducing inline json.loads flips this test."""
        state_dir = _make_hwm_file(
            tmp_path,
            {
                "account_id": "DELEG",
                "halt_triggered": False,
                "dd_used_dollars": 100,
                "dd_limit_dollars": 2000,
                "dd_pct_used": 0.05,
                "hwm_dollars": 50000,
                "hwm_timestamp": "2026-04-26T00:00:00",
            },
        )
        with (
            patch("trading_app.pre_session_check.STATE_DIR", state_dir),
            patch("trading_app.account_hwm_tracker.read_state_file") as mock_reader,
        ):
            mock_reader.return_value = {
                "account_id": "DELEG",
                "halt_triggered": False,
                "dd_used_dollars": 100,
                "dd_limit_dollars": 2000,
                "dd_pct_used": 0.05,
                "hwm_dollars": 50000,
                "hwm_timestamp": "2026-04-26T00:00:00",
            }
            check_hwm_tracker()
        assert mock_reader.call_count >= 1, "check_hwm_tracker must delegate to read_state_file"

    def test_corrupt_state_returns_blocked_filename_format_dd_circuit_breaker(self, tmp_path):
        """Both functions return (False, 'BLOCKED <filename>...') on corrupt input."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_BAD.json").write_text("{not json")
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_dd_circuit_breaker()
        assert ok is False
        assert msg.startswith("BLOCKED "), f"Message must start with 'BLOCKED '; got {msg!r}"
        assert "account_hwm_BAD.json" in msg

    def test_corrupt_state_returns_blocked_filename_format_hwm_tracker(self, tmp_path):
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "account_hwm_BAD2.json").write_text("{not json")
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_hwm_tracker()
        assert ok is False
        assert "BLOCKED account_hwm_BAD2.json" in msg, (
            f"check_hwm_tracker must produce unified BLOCKED <filename>: format; got {msg!r}"
        )

    def test_no_inline_json_loads_against_account_hwm_in_pre_session(self):
        """Stage 3 AC 10: no `json.loads` co-occurring with `account_hwm` token in
        pre_session_check.py or weekly_review.py source. Mutation guard against
        any future re-introduction of inline JSON parsing for tracker state files.
        """
        from trading_app import pre_session_check as ps_mod
        from trading_app import weekly_review as wr_mod

        for mod in (ps_mod, wr_mod):
            mod_file = mod.__file__
            assert mod_file is not None, f"{mod} has no __file__ — cannot scan source"
            src = Path(mod_file).read_text(encoding="utf-8")
            for line in src.splitlines():
                if "json.loads" in line and "account_hwm" in line:
                    raise AssertionError(
                        f"{mod_file}: inline json.loads against account_hwm — "
                        f"must delegate to read_state_file. Offending line: {line.strip()!r}"
                    )

    def test_clean_state_unchanged_behavior(self, tmp_path):
        """Backward compat: clean state file → both functions return True/passing."""
        state_dir = _make_hwm_file(
            tmp_path,
            {
                "account_id": "CLEAN",
                "halt_triggered": False,
                "dd_used_dollars": 100,
                "dd_limit_dollars": 2000,
                "dd_pct_used": 0.05,
                "hwm_dollars": 50000,
                "hwm_timestamp": "2026-04-26T00:00:00",
            },
        )
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok1, _ = check_dd_circuit_breaker()
            ok2, _ = check_hwm_tracker()
        assert ok1 is True
        assert ok2 is True


# ── Stage 4 — TopStep inactivity-window pre-flight check ──────────────────


def _seed_aged_state_file(tmp_path: Path, account_id: str, *, age_days: float) -> Path:
    """Helper: write an account_hwm_*.json with last_equity_timestamp aged
    exactly age_days from now. Returns the STATE_DIR for use with patch.
    """
    from datetime import UTC, datetime, timedelta

    state_dir = tmp_path / "data" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    last_ts = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    data = {
        "account_id": account_id,
        "firm": "topstep",
        "hwm_dollars": 50000.0,
        "hwm_timestamp": last_ts,
        "last_equity": 50000.0,
        "last_equity_timestamp": last_ts,
        "halt_triggered": False,
        "halt_timestamp": None,
        "halt_reason": "",
        "consecutive_poll_failures": 0,
        "hwm_frozen": False,
        "session_log": [],
    }
    f = state_dir / f"account_hwm_{account_id}.json"
    f.write_text(json.dumps(data))
    return state_dir


class TestStage4InactivityWindow:
    """Stage 4 of HWM persistence integrity hardening — pre-session
    pre-flight surfacing the TopStep XFA inactivity-closure boundary.

    Pins:
      - <25 days OK; >=25 and <30 WARN (still proceed); >=30 BLOCK.
      - state_file_age_days returning None blocks (fail-closed).
      - canonical-source annotation, UNGROUNDED+Rationale on the buffer.
      - delegation to state_file_age_days (no inline age computation).

    Boundary direction: `>= 25` and `>= 30` (not `>` either) — matches
    Stage 2's `_STATE_STALENESS_FAIL_DAYS` convention. Mutation guards on
    each boundary direction.
    """

    def test_under_25_days_returns_ok(self, tmp_path):
        state_dir = _seed_aged_state_file(tmp_path, "U25", age_days=10.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True
        assert "OK" in msg
        assert "BLOCKED" not in msg
        assert "WARN" not in msg

    def test_25_days_plus_1s_warns_continues(self, tmp_path):
        """Lower-boundary direction (>= 25). Mutation: > 25 flips."""
        state_dir = _seed_aged_state_file(tmp_path, "W25P", age_days=25.0 + 1 / 86400.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True, f"Should still proceed at 25d+1s; got blocked: {msg!r}"
        assert "WARN" in msg, f"Expected WARN; got {msg!r}"

    def test_25_days_minus_1s_silent_ok(self, tmp_path):
        """Boundary direction reverse: 25 - 1s is still OK band."""
        state_dir = _seed_aged_state_file(tmp_path, "W25M", age_days=25.0 - 1 / 86400.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True
        assert "WARN" not in msg
        assert "OK" in msg

    def test_29_days_warns(self, tmp_path):
        state_dir = _seed_aged_state_file(tmp_path, "W29", age_days=29.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True
        assert "WARN" in msg
        assert "29d" in msg

    def test_30_days_plus_1s_blocks(self, tmp_path):
        """Upper-boundary direction (>= 30). Mutation: > 30 flips.

        Stage 4 audit-gate SG-1 closure: BLOCK message must include the
        analogy disclaimer to prevent operators from mistakenly believing
        TopStep enforces account closure at exactly the 30-day bot-poll
        boundary. Mutation: removing the analogy text flips this test.
        """
        state_dir = _seed_aged_state_file(tmp_path, "B30P", age_days=30.0 + 1 / 86400.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is False, f"Should block at 30d+1s; got OK: {msg!r}"
        assert "BLOCKED" in msg
        assert "30d inactivity boundary" in msg
        assert "archive or delete" in msg
        # SG-1 audit-gate fix-up: disclaimer must be in the runtime message,
        # not only in the docstring.
        assert "borrowed by analogy" in msg, (
            f"Stage 4 audit-gate SG-1: BLOCK message must surface the analogy disclaimer; got {msg!r}"
        )

    def test_30_days_minus_1s_warns_does_not_block(self, tmp_path):
        """Boundary direction reverse: 30 - 1s is WARN, not BLOCK."""
        state_dir = _seed_aged_state_file(tmp_path, "B30M", age_days=30.0 - 1 / 86400.0)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True, f"Should not block at 30d-1s; got blocked: {msg!r}"
        assert "WARN" in msg

    def test_state_unreadable_blocks(self, tmp_path):
        """state_file_age_days returns None → BLOCKED (fail-closed)."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        # Create a file that exists but is unreadable enough that
        # state_file_age_days returns None (e.g. mock the helper).
        (state_dir / "account_hwm_NONE.json").write_text("garbage")
        with (
            patch("trading_app.pre_session_check.STATE_DIR", state_dir),
            patch("trading_app.account_hwm_tracker.state_file_age_days", return_value=None),
        ):
            ok, msg = check_topstep_inactivity_window()
        assert ok is False
        assert "BLOCKED" in msg
        assert "state file unreadable (cannot compute age)" in msg

    def test_no_files_returns_ok(self, tmp_path):
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True
        assert "first session" in msg.lower() or "no account state" in msg.lower()

    def test_skips_corrupt_named_files(self, tmp_path):
        """CORRUPT-named files are skipped (consistent with the rest of pre_session_check)."""
        state_dir = tmp_path / "data" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        # Write only a CORRUPT-named file → loop skips it → no other files → OK
        (state_dir / "account_hwm_X_CORRUPT_20260101.json").write_text("garbage")
        with patch("trading_app.pre_session_check.STATE_DIR", state_dir):
            ok, msg = check_topstep_inactivity_window()
        assert ok is True
        # Should hit the no-files branch (CORRUPT filtered out)
        assert "first session" in msg.lower() or "no account state" in msg.lower()

    def test_carries_canonical_source_annotation(self):
        """Greppable: docstring contains @canonical-source, @verbatim,
        @audit-finding, AND topstep_xfa_parameters.txt:351.
        Mutation: dropping any token flips this test.
        """
        doc = check_topstep_inactivity_window.__doc__ or ""
        assert "@canonical-source" in doc, "Missing @canonical-source token"
        assert "@verbatim" in doc, "Missing @verbatim token"
        assert "@audit-finding" in doc, "Missing @audit-finding token"
        assert "topstep_xfa_parameters.txt:351" in doc, "Missing canonical citation to topstep_xfa_parameters.txt:351"

    def test_25_day_buffer_has_ungrounded_rationale_comment(self):
        """Source-file scan: UNGROUNDED + Rationale: tokens in the function
        body, near the 25-day buffer constant.
        """
        from trading_app import pre_session_check as ps_mod

        mod_file = ps_mod.__file__
        assert mod_file is not None
        src = Path(mod_file).read_text(encoding="utf-8")
        # Locate the function body
        assert "def check_topstep_inactivity_window" in src
        idx = src.index("def check_topstep_inactivity_window")
        # Grab the next 3000 chars (function body)
        body = src[idx : idx + 3000]
        assert "UNGROUNDED" in body, "Missing UNGROUNDED label on the buffer constant"
        assert "Rationale:" in body, "Missing Rationale: block on the buffer constant"
        assert "_INACTIVITY_WARN_DAYS" in body or "25.0" in body, "Missing buffer constant"

    def test_delegates_to_state_file_age_days(self, tmp_path):
        """Mutation guard: re-introducing inline mtime/timestamp parsing flips this test."""
        state_dir = _seed_aged_state_file(tmp_path, "DELEG_S4", age_days=10.0)
        with (
            patch("trading_app.pre_session_check.STATE_DIR", state_dir),
            patch(
                "trading_app.account_hwm_tracker.state_file_age_days",
                return_value=10.0,
            ) as mock_helper,
        ):
            check_topstep_inactivity_window()
        assert mock_helper.call_count >= 1, "check_topstep_inactivity_window must delegate to state_file_age_days"
