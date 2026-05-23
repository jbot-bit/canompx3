"""Tests for `.claude/hooks/worktree_guard.py` (PreToolUse hook).

We do NOT acquire a real OS-level FileLock in these tests — the WORKTREE_GUARD_BYPASS
env var short-circuits the hook for tests that only need to validate event-handling
and matcher behaviour. For the actual blocking path we run the hook subprocess with
a synthetic acquire() that returns "blocked" via monkeypatching.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "worktree_guard.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("worktree_guard_hook", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_hook(event: dict, env: dict | None = None) -> tuple[int, str, str]:
    """Run the hook in a subprocess feeding `event` on stdin."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=full_env,
        timeout=10,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_bypass_env_var_allows_unconditionally():
    rc, _, stderr = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "echo hi"}},
        env={"WORKTREE_GUARD_BYPASS": "1"},
    )
    assert rc == 0, stderr


def test_unmatched_tool_passes_through():
    """The settings.json matcher already filters, but the hook is defensive."""
    rc, _, _ = _run_hook(
        {"tool_name": "Read", "tool_input": {"file_path": "foo"}},
    )
    assert rc == 0  # Read isn't in {Edit, Write, MultiEdit, Bash}


def test_malformed_event_fails_open():
    """JSON parse error -> exit 0 (fail-open)."""
    proc = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input="this is not json",
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0


def test_blocks_when_acquire_returns_blocked(monkeypatch: pytest.MonkeyPatch):
    """Directly invoke main() with a monkeypatched canonical module."""
    mod = _load_hook()

    fake_lease = {
        "pid": 99999,
        "worktree": "/tmp/peer",
        "branch": "feature/x",
        "iso_started": "2026-05-23T00:00:00+00:00",
        "iso_heartbeat": "2026-05-23T00:25:00+00:00",
    }

    class _FakeCanonical:
        @staticmethod
        def acquire(_cwd):  # noqa: ARG004
            del _cwd
            return "blocked", fake_lease, "peer holds OS lock"

        @staticmethod
        def lease_path(_cwd):  # noqa: ARG004
            del _cwd
            return Path("/tmp/peer/.git/.claude.worktree.lease.json")

        @staticmethod
        def lock_path(_cwd):  # noqa: ARG004
            del _cwd
            return Path("/tmp/peer/.git/.claude.worktree.lock")

    monkeypatch.setattr(mod, "_load_canonical", lambda: _FakeCanonical)
    monkeypatch.setattr(
        "sys.stdin",
        _StdinReplay(json.dumps({"tool_name": "Edit", "tool_input": {}})),
    )

    rc = mod.main()
    assert rc == 2


def test_allows_when_acquire_returns_acquired(monkeypatch: pytest.MonkeyPatch):
    mod = _load_hook()

    class _FakeCanonical:
        @staticmethod
        def acquire(_cwd):  # noqa: ARG004
            del _cwd
            return "acquired", {"pid": os.getpid()}, "acquired"

        @staticmethod
        def lease_path(_cwd):  # noqa: ARG004
            del _cwd
            return Path("/tmp/x/.git/.claude.worktree.lease.json")

        @staticmethod
        def lock_path(_cwd):  # noqa: ARG004
            del _cwd
            return Path("/tmp/x/.git/.claude.worktree.lock")

    monkeypatch.setattr(mod, "_load_canonical", lambda: _FakeCanonical)
    monkeypatch.setattr(
        "sys.stdin",
        _StdinReplay(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}})),
    )

    rc = mod.main()
    assert rc == 0


def test_allows_when_canonical_module_unavailable(monkeypatch: pytest.MonkeyPatch):
    """If _load_canonical returns None, fail-open."""
    mod = _load_hook()
    monkeypatch.setattr(mod, "_load_canonical", lambda: None)
    monkeypatch.setattr(
        "sys.stdin",
        _StdinReplay(json.dumps({"tool_name": "Edit", "tool_input": {}})),
    )
    rc = mod.main()
    assert rc == 0


def test_allows_when_acquire_raises(monkeypatch: pytest.MonkeyPatch):
    mod = _load_hook()

    class _BoomCanonical:
        @staticmethod
        def acquire(_cwd):  # noqa: ARG004
            del _cwd
            raise RuntimeError("synthetic failure")

        @staticmethod
        def lease_path(_cwd):  # noqa: ARG004
            del _cwd
            return None

        @staticmethod
        def lock_path(_cwd):  # noqa: ARG004
            del _cwd
            return None

    monkeypatch.setattr(mod, "_load_canonical", lambda: _BoomCanonical)
    monkeypatch.setattr(
        "sys.stdin",
        _StdinReplay(json.dumps({"tool_name": "Edit", "tool_input": {}})),
    )
    rc = mod.main()
    assert rc == 0


class _StdinReplay:
    """Minimal stdin replacement that satisfies json.load + isatty False."""

    def __init__(self, payload: str):
        self._payload = payload
        self._consumed = False

    def read(self) -> str:
        if self._consumed:
            return ""
        self._consumed = True
        return self._payload

    def isatty(self) -> bool:
        return False
