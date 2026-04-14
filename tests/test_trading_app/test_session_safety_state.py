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

    def test_load_drops_legacy_sr_alarm_blocks(self, state_dir: Path) -> None:
        """Pre-2026-04-14 files persisted SR-ALARM blocks. Loading such a
        file must drop those entries (they're re-derived from the SR review
        registry at session start). Real crash-recovery blocks (orphan,
        stuck exit) stay. The cleaned file is re-saved so the migration
        happens once."""
        f = state_dir / "session_safety_profile_test_MNQ.json"
        f.write_text(
            json.dumps(
                {
                    "portfolio": "profile_test",
                    "instrument": "MNQ",
                    "kill_switch_fired": False,
                    "close_time_forced": False,
                    "blocked_strategies": {
                        "STRAT_ORPHAN": "Orphaned broker position — manual resolution required",
                        "STRAT_STUCK": "Stuck exit retry failed — manual close required",
                        "STRAT_SR_STALE": "Criterion 12 SR ALARM — manual review required",
                        "STRAT_REGIME_STALE": "Criterion 11 regime fail — paused",
                        "STRAT_LIFECYCLE_STALE": "Paused pending manual review",
                    },
                    "shadow_failures": {},
                    "daily_pnl_r": 0.0,
                    "trading_day": "",
                    "cooldown_until": "",
                }
            )
        )

        state = SessionSafetyState("profile_test", "MNQ")

        # Legacy lifecycle-sourced entries dropped
        assert "STRAT_SR_STALE" not in state.blocked_strategies
        assert "STRAT_REGIME_STALE" not in state.blocked_strategies
        assert "STRAT_LIFECYCLE_STALE" not in state.blocked_strategies
        # Crash-recovery entries preserved
        assert "STRAT_ORPHAN" in state.blocked_strategies
        assert "STRAT_STUCK" in state.blocked_strategies

        # Disk file rewritten without the legacy entries (subsequent loads
        # are clean — migration is idempotent).
        data = json.loads(f.read_text())
        assert set(data["blocked_strategies"].keys()) == {"STRAT_ORPHAN", "STRAT_STUCK"}

    def test_load_no_migration_when_all_entries_are_legitimate(self, state_dir: Path) -> None:
        """Pure crash-recovery file (no legacy lifecycle entries) loads
        verbatim with no migration side effects."""
        f = state_dir / "session_safety_profile_test_MNQ.json"
        original = {
            "portfolio": "profile_test",
            "instrument": "MNQ",
            "kill_switch_fired": True,
            "close_time_forced": False,
            "blocked_strategies": {
                "STRAT_ORPHAN": "Orphaned broker position — manual resolution required",
            },
            "shadow_failures": {"123": "submit: TimeoutError"},
            "daily_pnl_r": -1.5,
            "trading_day": "2026-04-14",
            "cooldown_until": "",
        }
        f.write_text(json.dumps(original))
        mtime_before = f.stat().st_mtime_ns

        state = SessionSafetyState("profile_test", "MNQ")
        assert state.blocked_strategies == {
            "STRAT_ORPHAN": "Orphaned broker position — manual resolution required",
        }
        # File not rewritten (migration only triggers when legacy entries exist)
        assert f.stat().st_mtime_ns == mtime_before

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
