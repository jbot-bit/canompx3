"""Tests for trading_app.lane_ctl — lane pause/resume/list."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_app.lane_ctl import (
    _load_overrides,
    _save_overrides,
    get_lane_override,
    list_overrides,
    pause_lane,
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


PROFILE = "apex_50k_manual"
TEST_SID = "MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15"
TEST_SESSION = "SINGAPORE_OPEN"


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
