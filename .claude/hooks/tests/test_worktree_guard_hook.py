"""End-to-end regression tests for the worktree-guard PreToolUse hook.

These exercise the ACTUAL hook as Claude Code invokes it — a fresh `python`
subprocess with the event JSON on stdin and cwd set to the repo. This is the
path that failed twice (n=2, 2026-05-29/30): the old OS-lock model never blocked
a second session because the lock holder was the ephemeral hook subprocess
itself. The new (session_id, ppid)+heartbeat model blocks on pure state
inspection, which survives the subprocess boundary.

Strategy: every test runs against a synthetic git worktree under tmp_path so the
developer's real lease is never touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOOK = PROJECT_ROOT / ".claude" / "hooks" / "worktree_guard.py"
CANONICAL_DIR = PROJECT_ROOT / "scripts" / "tools"
sys.path.insert(0, str(CANONICAL_DIR))

import worktree_guard as wg  # noqa: E402  # type: ignore[import-not-found]


def _python() -> str:
    """The repo venv python if present, else the current interpreter."""
    if os.name == "nt":
        cand = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        cand = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


@pytest.fixture
def repo(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    yield tmp_path
    try:
        wg.release(tmp_path, force=True)
    except OSError:
        pass


def _run_hook(repo: Path, session_id: str, tool: str = "Bash") -> subprocess.CompletedProcess:
    event = json.dumps({"tool_name": tool, "session_id": session_id})
    # Scrub the bypass seam from the child env so these tests are deterministic
    # regardless of an ambient WORKTREE_GUARD_BYPASS set by the dev shell or CI
    # (the bypass would make the hook return 0 unconditionally, masking a real
    # block-path regression — caught 2026-05-30 when the verification shell had
    # it exported globally).
    env = {k: v for k, v in os.environ.items() if k != "WORKTREE_GUARD_BYPASS"}
    return subprocess.run(
        [_python(), str(HOOK)],
        input=event,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
    )


def test_hook_blocks_second_live_session(repo: Path):
    # Session A holds a live lease (ppid pinned to this live test process).
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    # Session B's hook must BLOCK (exit 2) and name the peer.
    r = _run_hook(repo, session_id="B")
    assert r.returncode == 2, r.stderr
    assert "BLOCKED" in r.stderr
    assert "A" in r.stderr  # peer session id surfaced


def test_hook_allows_same_session_refresh(repo: Path):
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    # The SAME session re-acting must be allowed (refresh, exit 0).
    r = _run_hook(repo, session_id="A")
    assert r.returncode == 0, r.stderr


def test_hook_allows_when_no_peer(repo: Path):
    r = _run_hook(repo, session_id="solo")
    assert r.returncode == 0, r.stderr


def test_hook_reclaims_dead_peer(repo: Path):
    # A peer whose ppid is dead must be reclaimable → second session allowed.
    wg.acquire(repo, pid=1001, session_id="deadA", ppid=os.getpid())
    ls = wg.lease_path(repo)
    assert ls is not None
    data = json.loads(ls.read_text(encoding="utf-8"))
    data["ppid"] = 2147480000  # no real process holds this
    ls.write_text(json.dumps(data), encoding="utf-8")
    r = _run_hook(repo, session_id="fresh")
    assert r.returncode == 0, r.stderr


def test_hook_fails_open_on_non_git(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    event = json.dumps({"tool_name": "Bash", "session_id": "x"})
    r = subprocess.run(
        [_python(), str(HOOK)], input=event, capture_output=True, text=True, cwd=str(plain)
    )
    assert r.returncode == 0  # not a git repo → fail-open (allow)


def test_hook_bypass_env_seam(repo: Path):
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    env = dict(os.environ, WORKTREE_GUARD_BYPASS="1")
    event = json.dumps({"tool_name": "Bash", "session_id": "B"})
    r = subprocess.run(
        [_python(), str(HOOK)], input=event, capture_output=True, text=True, cwd=str(repo), env=env
    )
    assert r.returncode == 0  # bypass seam → allow even with a live peer
