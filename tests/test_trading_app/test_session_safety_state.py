"""Tests for SessionSafetyState crash-recovery persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_app.live.session_safety_state import SessionSafetyState


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override _STATE_DIR to use tmp_path."""
    import trading_app.live.session_safety_state as mod

    monkeypatch.setattr(mod, "_STATE_DIR", tmp_path)
    return tmp_path


class TestSessionSafetyState:
    def test_clean_start_no_file(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        assert state.kill_switch_fired is False
        assert state.close_time_forced is False
        assert state.blocked_strategies == {}
        assert state.shadow_failures == {}

    def test_save_creates_file(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        state.kill_switch_fired = True
        state.save()
        f = state_dir / "session_safety_profile_test_MNQ.json"
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["kill_switch_fired"] is True

    def test_load_restores_state(self, state_dir: Path) -> None:
        # Save state
        state1 = SessionSafetyState("profile_test", "MNQ")
        state1.kill_switch_fired = True
        state1.close_time_forced = True
        state1.blocked_strategies = {"STRAT_1": "orphan", "STRAT_2": "stuck exit"}
        state1.shadow_failures = {"12345": "submit: TimeoutError"}
        state1.save()

        # Load in new instance (simulates crash recovery)
        state2 = SessionSafetyState("profile_test", "MNQ")
        assert state2.kill_switch_fired is True
        assert state2.close_time_forced is True
        assert state2.blocked_strategies == {"STRAT_1": "orphan", "STRAT_2": "stuck exit"}
        assert state2.shadow_failures == {"12345": "submit: TimeoutError"}

    def test_clear_removes_file(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        state.kill_switch_fired = True
        state.save()
        f = state_dir / "session_safety_profile_test_MNQ.json"
        assert f.exists()

        state.clear()
        assert not f.exists()

    def test_clear_idempotent_no_file(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        state.clear()  # Should not raise

    def test_atomic_write_no_tmp_left(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        state.kill_switch_fired = True
        state.save()
        # No .tmp file should remain
        tmp = state_dir / "session_safety_profile_test_MNQ.tmp"
        assert not tmp.exists()

    def test_corrupt_file_treated_as_clean(self, state_dir: Path) -> None:
        f = state_dir / "session_safety_profile_test_MNQ.json"
        f.write_text("NOT VALID JSON {{{")
        state = SessionSafetyState("profile_test", "MNQ")
        # Should not crash — treated as clean start
        assert state.kill_switch_fired is False

    def test_different_instruments_isolated(self, state_dir: Path) -> None:
        s1 = SessionSafetyState("profile_test", "MNQ")
        s1.kill_switch_fired = True
        s1.save()

        s2 = SessionSafetyState("profile_test", "MGC")
        assert s2.kill_switch_fired is False  # Separate file

    def test_daily_pnl_r_persists_same_day(self, state_dir: Path) -> None:
        """Daily PnL is restored when trading_day matches."""
        state = SessionSafetyState("profile_test", "MNQ")
        state.daily_pnl_r = -4.5
        state.trading_day = "2026-04-10"
        state.save()

        loaded = SessionSafetyState("profile_test", "MNQ")
        assert loaded.daily_pnl_r == -4.5
        assert loaded.trading_day == "2026-04-10"

    def test_daily_pnl_r_stale_day_ignored_by_caller(self, state_dir: Path) -> None:
        """daily_pnl_r is loaded regardless — caller checks trading_day match."""
        state = SessionSafetyState("profile_test", "MNQ")
        state.daily_pnl_r = -6.0
        state.trading_day = "2026-04-09"  # yesterday
        state.save()

        loaded = SessionSafetyState("profile_test", "MNQ")
        # Value is loaded (state doesn't know what day it is)
        assert loaded.daily_pnl_r == -6.0
        # But the orchestrator checks: if trading_day != str(self.trading_day), skip
        assert loaded.trading_day == "2026-04-09"

    def test_blocked_strategies_incremental(self, state_dir: Path) -> None:
        state = SessionSafetyState("profile_test", "MNQ")
        state.blocked_strategies["STRAT_A"] = "orphan"
        state.save()
        state.blocked_strategies["STRAT_B"] = "stuck exit"
        state.save()

        loaded = SessionSafetyState("profile_test", "MNQ")
        assert len(loaded.blocked_strategies) == 2
        assert "STRAT_A" in loaded.blocked_strategies
        assert "STRAT_B" in loaded.blocked_strategies
