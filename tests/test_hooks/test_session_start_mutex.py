"""Tests for `.claude/hooks/session-start.py`.

Covers:
- `_session_lock_lines()` mutex (PR #138 — was shipped without committed tests).
- `_action_queue_ready_lines()` surfacer (introduced alongside this test file).

Loads the hook via `importlib.util` because the file name uses a hyphen
(`session-start.py`) and cannot be imported as a normal module.

Subprocess-free by design: `_load_hook` monkeypatches the hook's `_git`
seam so no test in this module spawns `git`. Earlier revisions called
`subprocess.run(["git","init"])` once per test plus the hook's own
`git rev-parse` calls (~20 total) — on a GH-hosted Windows runner under
coverage instrumentation this reliably triggered a KeyboardInterrupt at
`threading.py:359` at ~10s elapsed, mid-`test_oserror_on_write_warns_…`.
Local Windows runs always passed. Eliminating the subprocess load removes
the trigger without weakening contract coverage: the rev-parse code path
is trivial subprocess plumbing already exercised by integration tests.
See `memory/feedback_ci_keyboardinterrupt_test_session_start_mutex_external_kill.md`.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

# Disable pytest-timeout for this module. Two prior fixes (subprocess elimination
# in 2ea20ee9, `timeout_func_only=true` in 85ae67fa) reduced but did not
# eliminate the GH-hosted-Windows-runner crash where pytest-timeout's
# `thread` method fires `_thread.interrupt_main()` during the inter-test
# teardown gap, raising KeyboardInterrupt at threading.py:359 and then
# crashing in `_pytest/capture.py:802` on `assert self._global_capturing
# is not None`. Observed again 2026-05-24 (run 26355837431) after the SHA
# manifest fix unmasked it. These tests are subprocess-free and complete
# in <1s collectively; the watchdog has no signal to detect here, only
# false positives from the Windows thread-interrupt race.
pytestmark = pytest.mark.timeout(0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "session-start.py"


def _load_hook(
    monkeypatch: pytest.MonkeyPatch,
    fake_root: Path,
    *,
    branch: str = "main",
    head_sha: str = "0" * 40,
) -> ModuleType:
    """Load session-start.py with `PROJECT_ROOT` pointed at `fake_root` and
    every git subprocess stubbed out.

    The hook's `_git([...])` is the single seam used by `_git_dir`,
    `_session_lock_lines` (branch + HEAD lookups), and several status
    surfacers. Stubbing it here means tests run with zero `subprocess.run`
    calls — see module docstring for the CI rationale.

    The stub creates `fake_root/.git/` so `_git_dir`'s return value is a
    real directory the test can write to.
    """
    spec = importlib.util.spec_from_file_location("session_start_hook", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PROJECT_ROOT", fake_root, raising=True)

    # `.git/` is expected to exist (real worktrees always have it). Tests
    # that pre-write the lock file MUST also pre-create the dir via
    # `_setup_git_dir(tmp_path)` — see the helper below.
    git_dir = fake_root / ".git"
    git_dir.mkdir(exist_ok=True)

    def _fake_git(args: list[str], timeout: int = 5) -> tuple[int, str]:
        # Mirror the small subset of `git` calls the mutex code actually makes.
        # Anything outside this set is a test bug — fail loudly so we don't
        # silently regress contract coverage.
        if args[:2] == ["rev-parse", "--git-dir"]:
            return 0, str(git_dir)
        if args[:2] == ["rev-parse", "--git-common-dir"]:
            return 0, str(git_dir)
        if args[:1] == ["branch"] and "--show-current" in args:
            return 0, branch
        if args[:1] == ["rev-parse"] and "HEAD" in args:
            return 0, head_sha
        return 1, ""

    monkeypatch.setattr(mod, "_git", _fake_git, raising=True)
    return mod


def _setup_git_dir(repo: Path) -> None:
    """Create `repo/.git/` so tests pre-writing the lock file have a parent dir.

    Tests that call `_load_hook(monkeypatch, tmp_path)` AFTER pre-writing
    the lock must call this helper first; `_load_hook` is what otherwise
    materialises the dir.
    """
    (repo / ".git").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# _session_lock_lines — 4 scenarios PR #138 claimed but did not commit
# ---------------------------------------------------------------------------


class TestSessionLockMutex:
    def test_clean_creates_lock_and_does_not_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook(monkeypatch, tmp_path)

        lines, should_block = hook._session_lock_lines()

        assert should_block is False
        assert lines == []
        lock = tmp_path / ".git" / ".claude.pid"
        assert lock.exists()
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["worktree"] == str(tmp_path)
        assert "iso_started" in payload

    def test_stale_pre_phase1_lock_warns_guard_inactive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A lock for THIS worktree without branch_at_start = pre-Phase-1 stale.
        The branch-flip-guard would silently exit 0 — surface the warning so
        the user knows to rm the lock and restart.
        """
        _setup_git_dir(tmp_path)
        lock = tmp_path / ".git" / ".claude.pid"
        # Pre-Phase-1 lock: no branch_at_start field, but THIS worktree.
        lock.write_text(
            json.dumps(
                {
                    "pid": 12345,
                    "ppid": 1,
                    "iso_started": "2026-04-26T00:00:00+00:00",
                    "worktree": str(tmp_path),
                }
            ),
            encoding="utf-8",
        )
        hook = _load_hook(monkeypatch, tmp_path)

        lines, should_block = hook._session_lock_lines()

        # Warning, not block — user can keep working but must restart to enable guard.
        assert should_block is False
        joined = "\n".join(lines)
        assert "branch-flip-guard inactive" in joined
        assert f"rm '{lock}'" in joined
        # Stale lock unchanged — must not overwrite.
        assert "12345" in lock.read_text(encoding="utf-8")

    def test_held_lock_blocks_with_diagnostic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pre-write a lock as if another live session held it.
        _setup_git_dir(tmp_path)
        lock = tmp_path / ".git" / ".claude.pid"
        lock.write_text(
            json.dumps(
                {
                    "pid": 99999,
                    "ppid": 1,
                    "iso_started": "2026-04-26T00:00:00+00:00",
                    "worktree": "/other/worktree/path",
                }
            ),
            encoding="utf-8",
        )
        hook = _load_hook(monkeypatch, tmp_path)
        monkeypatch.setattr(hook, "_pid_is_alive", lambda pid: True, raising=True)

        lines, should_block = hook._session_lock_lines()

        assert should_block is True
        joined = "\n".join(lines)
        assert "BLOCKED" in joined
        assert "99999" in joined  # holder PID surfaced
        assert "/other/worktree/path" in joined  # holder worktree surfaced
        assert f"rm '{lock}'" in joined  # exact cleanup command surfaced
        # Lock content unchanged — must not overwrite a held lock.
        assert "99999" in lock.read_text(encoding="utf-8")

    def test_corrupted_lock_still_blocks_and_degrades_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _setup_git_dir(tmp_path)
        lock = tmp_path / ".git" / ".claude.pid"
        lock.write_text("this is not json {{{ broken", encoding="utf-8")
        hook = _load_hook(monkeypatch, tmp_path)

        lines, should_block = hook._session_lock_lines()

        # Conservative: a corrupted lock means an unknown other session may
        # be live, so we still BLOCK rather than auto-clean.
        assert should_block is True
        joined = "\n".join(lines)
        assert "BLOCKED" in joined
        assert "(corrupted lock file)" in joined

    def test_stale_dead_pid_lock_auto_recovers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lock from a dead PID older than STALE_LOCK_RECLAIM_HOURS must be
        atomically replaced — never leaves the user with a sticky lock that
        silently disables protection.

        Reproduces the 2026-05-14 incident: lock from 2026-05-11 sat on disk
        for 3 days while two concurrent sessions wrote to the same .git/index
        and one nearly clobbered the other's staged commit. The original
        `_session_lock_lines()` had no liveness check; both sessions exited
        the function without re-acquiring the lock.
        """
        _setup_git_dir(tmp_path)
        lock = tmp_path / ".git" / ".claude.pid"
        held_pid = 999_999_001  # arbitrary; we mock _pid_is_alive below
        lock.write_text(
            json.dumps(
                {
                    "pid": held_pid,
                    "ppid": 1,
                    "iso_started": "2026-05-11T05:43:18.252397+00:00",
                    "worktree": str(tmp_path),
                    "branch_at_start": "main",
                }
            ),
            encoding="utf-8",
        )
        hook = _load_hook(monkeypatch, tmp_path)
        # Pin "now" so the lock age check is deterministic across CI clocks.
        from datetime import UTC, datetime

        fixed_now = datetime(2026, 5, 14, 5, 43, 18, tzinfo=UTC)

        class _PinnedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now if tz is None else fixed_now.astimezone(tz)

        monkeypatch.setattr(hook, "datetime", _PinnedDateTime, raising=True)
        # Stub PID liveness — production behaviour is OS-dependent (POSIX
        # uses os.kill(pid,0); Windows conservatively treats unknown errors
        # as alive). The contract under test is the recovery branch itself.
        monkeypatch.setattr(hook, "_pid_is_alive", lambda pid: pid != held_pid, raising=True)

        lines, should_block = hook._session_lock_lines()

        assert should_block is False, "stale dead-PID lock must NOT block — auto-recover"
        # New lock must reflect THIS session, not the stale one.
        new_payload = json.loads(lock.read_text(encoding="utf-8"))
        assert new_payload["pid"] == os.getpid()
        assert new_payload["worktree"] == str(tmp_path)
        # Surface the recovery so the user knows a stale lock was cleaned.
        joined = "\n".join(lines)
        assert "stale" in joined.lower()

    def test_held_lock_with_live_pid_in_other_worktree_still_blocks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A live PID in another worktree must always BLOCK — auto-recovery
        only applies to dead PIDs. This guards against a too-aggressive
        reclaim that would clobber a real concurrent session.
        """
        _setup_git_dir(tmp_path)
        lock = tmp_path / ".git" / ".claude.pid"
        # Use os.getpid() — guaranteed live (it's us). Different worktree.
        lock.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "ppid": 1,
                    "iso_started": "2026-05-11T05:43:18.252397+00:00",
                    "worktree": "/some/other/worktree",
                    "branch_at_start": "main",
                }
            ),
            encoding="utf-8",
        )
        hook = _load_hook(monkeypatch, tmp_path)

        lines, should_block = hook._session_lock_lines()

        assert should_block is True, "live PID in other worktree must BLOCK even if old"
        joined = "\n".join(lines)
        assert "BLOCKED" in joined

    def test_oserror_on_write_warns_but_does_not_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook(monkeypatch, tmp_path)

        def raise_enospc(_lock_path: Path) -> int:
            raise OSError(28, "No space left on device")

        # Patch the lock-acquire seam directly rather than the singleton
        # ``os.open``. Patching ``hook.os.open`` mutates the global ``os``
        # module for the test's duration, which can interact unpredictably
        # with pytest/coverage teardown machinery on Windows CI runners.
        monkeypatch.setattr(hook, "_acquire_lock_fd", raise_enospc)

        lines, should_block = hook._session_lock_lines()

        # Transient FS issues must not block — locking out every future
        # session is a worse failure mode than the contention we prevent.
        assert should_block is False
        joined = "\n".join(lines)
        assert "WARNING" in joined
        assert "No space left on device" in joined
        # Lock file must NOT exist after a write failure.
        assert not (tmp_path / ".git" / ".claude.pid").exists()

    def test_acquire_lock_fd_seam_is_not_globally_patched_after_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: after the OSError test runs, ``os.open`` must
        not remain shadowed at the singleton level. Confirms the seam stays
        scoped to ``hook._acquire_lock_fd``.
        """
        hook = _load_hook(monkeypatch, tmp_path)

        # Sentinel: capture the unpatched os.open before the seam swap.
        baseline_open = os.open
        baseline_hook_open = hook.os.open

        def raise_enospc(_lock_path: Path) -> int:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(hook, "_acquire_lock_fd", raise_enospc)
        hook._session_lock_lines()

        # ``os.open`` (both views) must be the same callable we captured.
        assert os.open is baseline_open
        assert hook.os.open is baseline_hook_open


# ---------------------------------------------------------------------------
# _action_queue_ready_lines — surfacer for stale `status: ready` items
# ---------------------------------------------------------------------------


class TestActionQueueReadySurfacer:
    def test_missing_file_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook(monkeypatch, tmp_path)
        # docs/runtime/action-queue.yaml does not exist under tmp_path.
        assert hook._action_queue_ready_lines() == []

    def test_ready_item_surfaced_by_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        queue_dir = tmp_path / "docs" / "runtime"
        queue_dir.mkdir(parents=True)
        (queue_dir / "action-queue.yaml").write_text(
            "\n".join(
                [
                    "schema_version: 1",
                    "updated_at: 2026-04-26T00:00:00+00:00",
                    "items:",
                    "  - id: only-ready-item",
                    "    title: A test item",
                    "    class: research",
                    "    status: ready",
                    "    priority: P1",
                    "    close_before_new_work: true",
                    "    owner_hint: codex",
                    "    last_verified_at: 2026-04-26",
                    "    freshness_sla_days: 3",
                    "    next_action: Do the thing",
                    "    exit_criteria: Thing is done",
                    "    blocked_by: []",
                    "    decision_refs: []",
                    "    evidence_refs: []",
                    "    notes_ref:",
                    "    override_note:",
                    "  - id: closed-item-should-not-show",
                    "    title: A closed item",
                    "    class: research",
                    "    status: closed",
                    "    priority: P2",
                    "    close_before_new_work: false",
                    "    owner_hint: codex",
                    "    last_verified_at: 2026-04-26",
                    "    freshness_sla_days: 3",
                    "    next_action: Closed",
                    "    exit_criteria: Closed",
                    "    blocked_by: []",
                    "    decision_refs: []",
                    "    evidence_refs: []",
                    "    notes_ref:",
                    "    override_note:",
                ]
            ),
            encoding="utf-8",
        )
        hook = _load_hook(monkeypatch, tmp_path)

        lines = hook._action_queue_ready_lines()

        assert len(lines) == 1
        assert "Action queue READY" in lines[0]
        assert "only-ready-item" in lines[0]
        assert "closed-item-should-not-show" not in lines[0]
