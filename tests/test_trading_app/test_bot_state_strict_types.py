"""Strict-type contamination guard for bot_state.write_state.

Mutation proof for the 2026-05-17 live-throughput triage finding: the canonical
bot-state runtime file was polluted with literal "<MagicMock name='...'>"
strings because json.dumps(default=str, ...) silently coerced mock objects. The
fix (_sanitize_for_state + _json_default) must refuse such writes.
"""

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading_app.live import bot_state


@pytest.fixture
def isolated_state_file(tmp_path, monkeypatch):
    """Redirect STATE_FILE to a tempdir so tests cannot touch the canonical path."""
    target = tmp_path / "bot_state.json"
    monkeypatch.setattr(bot_state, "STATE_FILE", target)
    return target


def test_write_state_refuses_top_level_magicmock(isolated_state_file, caplog):
    """A MagicMock at the root payload must abort the write and leave no file."""
    payload = {
        "mode": "live",
        "daily_pnl_r": MagicMock(name="mock.daily_pnl_r"),
    }
    with caplog.at_level(logging.CRITICAL, logger="trading_app.live.bot_state"):
        bot_state.write_state(payload)

    assert not isolated_state_file.exists(), "contaminated state file must not be created"
    assert any("bot_state contamination" in r.message for r in caplog.records), (
        "operator-visible CRITICAL log must name the contamination"
    )
    assert any("daily_pnl_r" in r.message for r in caplog.records), (
        "log must include the dotted field path so operators can locate the contamination source"
    )


def test_write_state_refuses_nested_magicmock(isolated_state_file, caplog):
    """A MagicMock nested inside a lane dict must abort the write with full dotted path."""
    payload = {
        "mode": "live",
        "lanes": {
            "TEST_STRAT_001": {
                "instrument": "MGC",
                "orb_break_direction": MagicMock(name="mock.orbs.get().break_dir"),
            }
        },
    }
    with caplog.at_level(logging.CRITICAL, logger="trading_app.live.bot_state"):
        bot_state.write_state(payload)

    assert not isolated_state_file.exists()
    msgs = " | ".join(r.message for r in caplog.records)
    assert "bot_state contamination" in msgs
    assert "lanes.TEST_STRAT_001.orb_break_direction" in msgs, f"dotted path must locate the nested mock; got: {msgs}"


def test_write_state_accepts_clean_payload_with_datetime_date_path(isolated_state_file):
    """Clean payload with datetime/date/Path round-trips correctly through strict serializer."""
    payload = {
        "mode": "live",
        "instrument": "MNQ",
        "trading_day": date(2026, 5, 17),
        "session_start_utc": datetime(2026, 5, 17, 20, 30, 0, tzinfo=UTC),
        "config_path": Path("/tmp/cfg.yaml"),
        "lanes": {
            "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12": {
                "instrument": "MNQ",
                "rr_target": 1.0,
                "current_pnl_r": None,
                "active": True,
            }
        },
        "lane_cards": [
            {"strategy_id": "MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12", "status": "WAITING"},
        ],
    }
    bot_state.write_state(payload)

    assert isolated_state_file.exists(), "clean payload must produce a file"
    content = json.loads(isolated_state_file.read_text(encoding="utf-8"))
    assert content["mode"] == "live"
    assert content["trading_day"] == "2026-05-17", "date must serialize to ISO string"
    assert content["session_start_utc"].startswith("2026-05-17T20:30:00")
    assert "heartbeat_utc" in content, "heartbeat_utc must be auto-stamped on write"
    assert "MagicMock" not in isolated_state_file.read_text(encoding="utf-8"), (
        "clean payload must never produce MagicMock literals in the file"
    )
