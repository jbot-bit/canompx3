"""Stage 3 of HWM persistence integrity hardening — weekly_review delegation.

Pins:
  - section_0_account_health calls account_hwm_tracker.read_state_file
  - corrupt state files silently skip (no `ERROR:` print to stdout); the
    granular reason is captured in operator logs by read_state_file.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d


def _write_valid(state_dir: Path, account_id: str = "WR1") -> Path:
    f = state_dir / f"account_hwm_{account_id}.json"
    f.write_text(
        json.dumps(
            {
                "account_id": account_id,
                "firm": "topstep",
                "last_equity": 50000,
                "hwm_dollars": 50000,
                "dd_used_dollars": 100,
                "dd_limit_dollars": 2000,
                "dd_pct_used": 0.05,
                "halt_triggered": False,
            }
        )
    )
    return f


def test_section_0_calls_shared_reader(state_dir):
    """Mutation guard: re-introducing inline json.loads flips this test."""
    _write_valid(state_dir, "DELEG")
    with (
        patch("trading_app.weekly_review.STATE_DIR", state_dir),
        patch("trading_app.account_hwm_tracker.read_state_file") as mock_reader,
    ):
        mock_reader.return_value = {
            "account_id": "DELEG",
            "firm": "topstep",
            "last_equity": 50000,
            "hwm_dollars": 50000,
            "dd_used_dollars": 100,
            "dd_limit_dollars": 2000,
            "dd_pct_used": 0.05,
            "halt_triggered": False,
        }
        from trading_app.weekly_review import section_0_account_health

        section_0_account_health()
    assert mock_reader.call_count >= 1, "section_0_account_health must delegate to read_state_file"


def test_section_0_corrupt_state_silently_skips(state_dir, capsys):
    """Corrupt files must NOT print 'ERROR:' to stdout; granular reason
    captured by read_state_file's log.warning instead.
    """
    (state_dir / "account_hwm_BAD.json").write_text("{not json")
    with patch("trading_app.weekly_review.STATE_DIR", state_dir):
        from trading_app.weekly_review import section_0_account_health

        section_0_account_health()
    out = capsys.readouterr().out
    assert "ERROR:" not in out, f"Stage 3: corrupt files must skip silently (no stdout ERROR); got: {out!r}"


def test_section_0_no_inline_json_loads_against_account_hwm():
    """Static-source assertion: weekly_review.py must not call json.loads
    against account_hwm_*.json paths inline. Same constraint as
    pre_session_check (Stage 3 AC 10). Mutation: re-introducing inline
    parsing flips this test.
    """
    from trading_app import weekly_review as wr_mod

    mod_file = wr_mod.__file__
    assert mod_file is not None
    src = Path(mod_file).read_text(encoding="utf-8")
    for line in src.splitlines():
        if "json.loads" in line and "account_hwm" in line:
            raise AssertionError(
                f"weekly_review.py: inline json.loads against account_hwm — "
                f"must delegate to read_state_file. Offending line: {line.strip()!r}"
            )
