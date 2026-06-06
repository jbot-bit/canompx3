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


def _run_hook(
    repo: Path,
    session_id: str,
    tool: str = "Bash",
    *,
    command: str | None = "git commit -m x",
    file_path: str | None = None,
    event_cwd: str | None = None,
    proc_cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Invoke the hook subprocess as Claude Code does.

    Defaults to an index-MUTATING Bash command (`git commit`) so the legacy
    block-path tests still exercise the block branch under the post-2026-06-03
    op-classification gate (read-only Bash no longer blocks). Pass
    `command`/`file_path`/`event_cwd`/`proc_cwd` to exercise the new gates.
    """
    payload: dict = {"tool_name": tool, "session_id": session_id}
    if tool == "Bash" and command is not None:
        payload["tool_input"] = {"command": command}
    if tool in {"Edit", "Write", "MultiEdit"} and file_path is not None:
        payload["tool_input"] = {"file_path": file_path}
    if event_cwd is not None:
        payload["cwd"] = event_cwd
    event = json.dumps(payload)
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
        cwd=str(proc_cwd if proc_cwd is not None else repo),
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
    event = json.dumps({"tool_name": "Bash", "session_id": "B", "tool_input": {"command": "git commit -m x"}})
    r = subprocess.run(
        [_python(), str(HOOK)], input=event, capture_output=True, text=True, cwd=str(repo), env=env
    )
    assert r.returncode == 0  # bypass seam → allow even with a live peer


# ---------------------------------------------------------------------------
# 2026-06-03 fix: F1 (cwd scoping), F2 (read-only Bash), F4 (block message),
# F5 (outside-repo writes). The prior hook blocked EVERY Bash command and read
# the lease against the hook-process cwd (always the main checkout), so a
# correctly-isolated worktree got false-blocked by the main checkout's peer.
# ---------------------------------------------------------------------------


def test_f2_readonly_bash_allowed_under_live_peer(repo: Path):
    """A read-only Bash command must ALLOW even when a live peer holds the lease.

    This is the dominant false-block class: the old hook blocked ALL Bash, so a
    harmless `git status` / `pwd` was killed while a peer held the lease.
    """
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    for cmd in ("git status", "pwd", "git rev-parse --git-dir", "ls -la", "python -c 'print(1)'", "git log --oneline"):
        r = _run_hook(repo, session_id="B", command=cmd)
        assert r.returncode == 0, f"read-only {cmd!r} should ALLOW; stderr={r.stderr}"


def test_f2_mutating_git_bash_blocked_under_live_peer(repo: Path):
    """A mutating git Bash command must still BLOCK under a live peer."""
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    for cmd in ("git commit -m x", "git add .", "cd sub && git merge feature", "git -C . reset --hard"):
        r = _run_hook(repo, session_id="B", command=cmd)
        assert r.returncode == 2, f"mutating {cmd!r} should BLOCK; stderr={r.stderr}"


def test_f1_event_cwd_scopes_to_isolated_worktree(repo: Path):
    """An Edit/Bash whose event cwd is a DIFFERENT git tree must not read THIS
    tree's lease. Mirrors the real bug: hook process cwd = main checkout, but
    the tool ran in an isolated worktree with no peer.

    We point the hook process cwd at `repo` (which has a live peer lease) but
    set event["cwd"] to a SEPARATE git repo with no lease. The fix must scope to
    the event cwd → ALLOW.
    """
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    other = repo.parent / "other_tree"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=other, check=True)
    try:
        r = _run_hook(
            repo,
            session_id="B",
            command="git commit -m x",
            event_cwd=str(other),
            proc_cwd=repo,  # hook process cwd = the tree WITH the peer
        )
        assert r.returncode == 0, f"event cwd in a peerless tree should ALLOW; stderr={r.stderr}"
    finally:
        wg.release(other, force=True)


def test_f1_missing_event_cwd_falls_back_to_proc_cwd(repo: Path):
    """No event cwd → fall back to the hook-process cwd (historical behaviour).

    A mutating Bash with a live peer and no event cwd must still BLOCK.
    """
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    r = _run_hook(repo, session_id="B", command="git commit -m x", event_cwd=None, proc_cwd=repo)
    assert r.returncode == 2, r.stderr


def test_f5_outside_repo_write_allowed_under_live_peer(repo: Path):
    """An Edit/Write to a path OUTSIDE the repo must ALLOW under a live peer.

    The user's `memory/` dir lives under ~/.claude/…, not the repo, so such a
    write cannot corrupt the git index.
    """
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    outside = repo.parent / "outside_memory.md"
    r = _run_hook(repo, session_id="B", tool="Write", file_path=str(outside), event_cwd=str(repo))
    assert r.returncode == 0, f"outside-repo write should ALLOW; stderr={r.stderr}"


def test_f5_inside_repo_write_blocked_under_live_peer(repo: Path):
    """An Edit/Write to a path INSIDE the repo must still BLOCK under a peer."""
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    inside = repo / "pipeline" / "x.py"
    r = _run_hook(repo, session_id="B", tool="Edit", file_path=str(inside), event_cwd=str(repo))
    assert r.returncode == 2, r.stderr


def test_f4_block_message_leads_with_liveness_and_advisory_pid(repo: Path):
    """The BLOCK message must surface heartbeat age + a live verdict and mark the
    PID advisory (not authoritative)."""
    wg.acquire(repo, pid=1001, session_id="A", ppid=os.getpid())
    r = _run_hook(repo, session_id="B", command="git commit -m x")
    assert r.returncode == 2, r.stderr
    assert "Liveness:" in r.stderr
    assert "heartbeat" in r.stderr.lower()
    assert "advisory" in r.stderr.lower()  # PID flagged advisory
    assert "new_session.sh" in r.stderr  # exact safe next action
