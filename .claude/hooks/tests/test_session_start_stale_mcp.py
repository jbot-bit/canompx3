"""Tests for stale-MCP-server detection in `.claude/hooks/session-start.py`.

When an MCP server's source is committed AFTER its running process started, the
live server is serving stale code (the operator's "I edited the MCP but it didn't
take effect" footgun). The session-start hook auto-warns so nothing must be
remembered. Pure decision logic is unit-tested here; the OS/git boundary
(_stale_mcp_server_lines) is fail-open and not exercised in-process.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mod():
    hook_path = Path(__file__).resolve().parents[1] / "session-start.py"
    spec = importlib.util.spec_from_file_location("session_start_hook", hook_path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["session_start_hook"] = m
    spec.loader.exec_module(m)
    return m


def _dt(day: int) -> datetime:
    return datetime(2026, 5, day, 12, 0, 0, tzinfo=timezone.utc)


def test_code_committed_after_process_start_is_stale(mod):
    """Process started day 28; code committed day 31 -> stale (restart needed)."""
    stale = mod._stale_mcp_servers(
        [("research-catalog", _dt(28), _dt(31))]
    )
    assert stale == ["research-catalog"]


def test_code_committed_before_process_start_is_fresh(mod):
    """Process started day 31; code committed day 28 -> server has the code."""
    stale = mod._stale_mcp_servers(
        [("research-catalog", _dt(31), _dt(28))]
    )
    assert stale == []


def test_no_running_process_is_not_stale(mod):
    """No running process (started is None) -> nothing to restart, not stale."""
    stale = mod._stale_mcp_servers(
        [("research-catalog", None, _dt(31))]
    )
    assert stale == []


def test_unknown_commit_time_is_not_stale(mod):
    """No commit time resolvable (None) -> fail-open, not flagged."""
    stale = mod._stale_mcp_servers(
        [("research-catalog", _dt(28), None)]
    )
    assert stale == []


def test_multiple_servers_only_stale_ones_returned(mod):
    stale = mod._stale_mcp_servers(
        [
            ("research-catalog", _dt(28), _dt(31)),  # stale
            ("gold-db", _dt(31), _dt(28)),           # fresh
            ("repo-state", _dt(20), _dt(30)),        # stale
        ]
    )
    assert stale == ["research-catalog", "repo-state"]


def test_lines_warn_format_when_stale(mod, monkeypatch):
    """The line wrapper emits one warn line per stale server with the restart cue."""
    monkeypatch.setattr(mod, "_collect_mcp_server_states", lambda: [("research-catalog", _dt(28), _dt(31))])
    lines = mod._stale_mcp_server_lines()
    assert any("research-catalog" in ln and "restart" in ln.lower() for ln in lines), lines


def test_lines_empty_when_all_fresh(mod, monkeypatch):
    monkeypatch.setattr(mod, "_collect_mcp_server_states", lambda: [("gold-db", _dt(31), _dt(28))])
    assert mod._stale_mcp_server_lines() == []


def test_lines_fail_open_on_collector_error(mod, monkeypatch):
    def _boom():
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(mod, "_collect_mcp_server_states", _boom)
    assert mod._stale_mcp_server_lines() == []
