"""Tests for trading_app.lane_ctl — lane pause/resume/list."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.prop_profiles import ACCOUNT_PROFILES, effective_daily_lanes
from trading_app.lane_ctl import (
    _load_overrides,
    _save_overrides,
    get_lane_override,
    get_paused_strategy_ids,
    list_overrides,
    pause_lane,
    pause_strategy_id,
    resume_lane,
)


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "data" / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def mock_state(state_dir):
    """Patch STATE_DIR to tmp_path."""
    with patch("trading_app.lane_ctl.STATE_DIR", state_dir):
        yield state_dir


PROFILE = "topstep_50k_mnq_auto"


def _first_lane(profile_id: str):
    """Return the first deployed lane for the profile.

    Picks dynamically rather than hardcoding a session: which sessions are
    in the profile depends on the live `lane_allocation.json` post-gate
    (e.g. the chordia gate, allocator_chordia_gate stage 2026-05-01,
    refuses lanes failing Chordia 2018 t-stat — so the surviving lane set
    can shrink between rebalances). This test only needs any valid lane
    to exercise the lane_ctl CRUD paths; pin to the first lane present.
    """
    profile = ACCOUNT_PROFILES[profile_id]
    lanes = list(effective_daily_lanes(profile))
    if not lanes:
        raise AssertionError(
            f"Profile {profile_id} has no deployed lanes — "
            f"check docs/runtime/lane_allocation.json and the chordia gate output."
        )
    return lanes[0]


_FIRST_LANE = _first_lane(PROFILE)
TEST_SID = _FIRST_LANE.strategy_id
TEST_SESSION = _FIRST_LANE.orb_label


class TestLoadSaveOverrides:
    def test_empty_when_no_file(self, mock_state):
        assert _load_overrides(PROFILE) == {}

    def test_roundtrip(self, mock_state):
        data = {TEST_SID: {"active": False, "reason": "test"}}
        _save_overrides(PROFILE, data)
        assert _load_overrides(PROFILE) == data

    def test_corrupt_file_returns_empty(self, mock_state):
        path = mock_state / f"lane_overrides_{PROFILE}.json"
        path.write_text("{{{not json")
        assert _load_overrides(PROFILE) == {}


class TestGetLaneOverride:
    def test_no_override(self, mock_state):
        assert get_lane_override(PROFILE, TEST_SID) is None

    def test_paused_lane(self, mock_state):
        _save_overrides(PROFILE, {TEST_SID: {"active": False, "reason": "cold"}})
        result = get_lane_override(PROFILE, TEST_SID)
        assert result is not None
        assert result["reason"] == "cold"

    def test_expired_override_returns_none(self, mock_state):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        _save_overrides(PROFILE, {TEST_SID: {"active": False, "expires": yesterday}})
        assert get_lane_override(PROFILE, TEST_SID) is None

    def test_future_expiry_still_paused(self, mock_state):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        _save_overrides(PROFILE, {TEST_SID: {"active": False, "expires": tomorrow}})
        assert get_lane_override(PROFILE, TEST_SID) is not None

    def test_active_true_returns_none(self, mock_state):
        """active=True means lane is NOT paused."""
        _save_overrides(PROFILE, {TEST_SID: {"active": True}})
        assert get_lane_override(PROFILE, TEST_SID) is None


class TestPauseResume:
    def test_pause_creates_override(self, mock_state):
        pause_lane(PROFILE, TEST_SESSION, reason="losing streak")
        result = get_lane_override(PROFILE, TEST_SID)
        assert result is not None
        assert "losing streak" in result["reason"]

    def test_resume_removes_override(self, mock_state):
        pause_lane(PROFILE, TEST_SESSION, reason="test")
        resume_lane(PROFILE, TEST_SESSION)
        assert get_lane_override(PROFILE, TEST_SID) is None

    def test_pause_with_expiry(self, mock_state):
        pause_lane(PROFILE, TEST_SESSION, expires="2026-04-15")
        overrides = _load_overrides(PROFILE)
        assert overrides[TEST_SID]["expires"] == "2026-04-15"

    def test_pause_unknown_session_exits(self, mock_state):
        with pytest.raises(SystemExit):
            pause_lane(PROFILE, "NONEXISTENT_SESSION")

    def test_pause_strategy_id_creates_override(self, mock_state):
        created = pause_strategy_id(PROFILE, TEST_SID, reason="sr alarm", source="sr_monitor")
        assert created is True
        result = get_lane_override(PROFILE, TEST_SID)
        assert result is not None
        assert result["source"] == "sr_monitor"

    def test_pause_strategy_id_idempotent_when_already_paused(self, mock_state):
        assert pause_strategy_id(PROFILE, TEST_SID, reason="first") is True
        assert pause_strategy_id(PROFILE, TEST_SID, reason="second") is False


class TestPausedIds:
    def test_get_paused_strategy_ids_empty(self, mock_state):
        assert get_paused_strategy_ids(PROFILE) == set()

    def test_get_paused_strategy_ids_filters_expired(self, mock_state):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        _save_overrides(
            PROFILE,
            {
                TEST_SID: {"active": False, "expires": tomorrow},
                "OLD_SID": {"active": False, "expires": yesterday},
                "ACTIVE_SID": {"active": True},
            },
        )
        assert get_paused_strategy_ids(PROFILE) == {TEST_SID}


class TestListOverrides:
    def test_empty_list(self, mock_state, capsys):
        list_overrides(PROFILE)
        assert "No lane overrides" in capsys.readouterr().out

    def test_shows_paused_lane(self, mock_state, capsys):
        pause_lane(PROFILE, TEST_SESSION, reason="cold streak")
        list_overrides(PROFILE)
        out = capsys.readouterr().out
        assert "PAUSED" in out
        assert "cold streak" in out

    def test_detects_orphan(self, mock_state, capsys):
        _save_overrides(PROFILE, {"NONEXISTENT_STRATEGY": {"active": False, "since": "2026-01-01"}})
        list_overrides(PROFILE)
        out = capsys.readouterr().out
        assert "ORPHANED" in out
