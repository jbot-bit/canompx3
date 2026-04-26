"""Tests for `.claude/hooks/session-start.py`.

Covers:
- `_session_lock_lines()` mutex (PR #138 — was shipped without committed tests).
- `_action_queue_ready_lines()` surfacer (introduced alongside this test file).

Loads the hook via `importlib.util` because the file name uses a hyphen
(`session-start.py`) and cannot be imported as a normal module.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "session-start.py"


def _load_hook(monkeypatch: pytest.MonkeyPatch, fake_root: Path) -> ModuleType:
    """Load session-start.py with PROJECT_ROOT pointed at `fake_root`.

    The module's PROJECT_ROOT is computed at import time from `__file__`,
    so we monkeypatch the constant after exec. Per blast-radius analysis,
    the constant is consumed only via `subprocess.run(cwd=...)` calls
    evaluated at call time, so this is safe.
    """
    spec = importlib.util.spec_from_file_location("session_start_hook", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PROJECT_ROOT", fake_root, raising=True)
    return mod


def _init_git(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# _session_lock_lines — 4 scenarios PR #138 claimed but did not commit
# ---------------------------------------------------------------------------


class TestSessionLockMutex:
    def test_clean_creates_lock_and_does_not_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _init_git(tmp_path)
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

    def test_held_lock_blocks_with_diagnostic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _init_git(tmp_path)
        # Pre-write a lock as if another live session held it.
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
        _init_git(tmp_path)
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

    def test_oserror_on_write_warns_but_does_not_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _init_git(tmp_path)
        hook = _load_hook(monkeypatch, tmp_path)

        original_open = os.open

        def selective_open(path, flags, *args, **kwargs):
            # Break only the lock-write path (which uses O_EXCL).
            if flags & os.O_EXCL:
                raise OSError(28, "No space left on device")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(hook.os, "open", selective_open)

        lines, should_block = hook._session_lock_lines()

        # Transient FS issues must not block — locking out every future
        # session is a worse failure mode than the contention we prevent.
        assert should_block is False
        joined = "\n".join(lines)
        assert "WARNING" in joined
        assert "No space left on device" in joined
        # Lock file must NOT exist after a write failure.
        assert not (tmp_path / ".git" / ".claude.pid").exists()


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
