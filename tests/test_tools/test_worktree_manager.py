"""Tests for scripts.tools.worktree_manager."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tools import worktree_manager


class TestHelpers:
    def test_slugify(self) -> None:
        assert worktree_manager.slugify("My Task / Thing") == "my-task-thing"

    def test_branch_name(self) -> None:
        assert worktree_manager.build_branch_name("codex", "Bug Fix") == "wt-codex-bug-fix"

    def test_worktree_path(self) -> None:
        path = worktree_manager.build_worktree_path("claude", "Feature A")
        assert path.parts[-4:] == (".worktrees", "tasks", "claude", "feature-a")


class TestParseWorktreeList:
    def test_parses_porcelain_output(self) -> None:
        output = "\n".join(
            [
                "worktree /repo",
                "HEAD abc123",
                "branch refs/heads/main",
                "",
                "worktree /repo/.worktrees/codex/foo",
                "HEAD def456",
                "branch refs/heads/wt/codex/foo",
                "prunable gitdir file points to non-existent location",
                "",
            ]
        )
        items = worktree_manager.parse_worktree_list(output)
        assert len(items) == 2
        assert items[0].path == "/repo"
        assert items[1].prunable is not None


class TestManagedWorktrees:
    def test_list_managed_worktrees_filters_to_active_managed_paths(self, tmp_path: Path) -> None:
        worktree_root = tmp_path / ".worktrees"
        managed_path = worktree_root / "codex" / "task-a"
        managed_path.mkdir(parents=True)
        (managed_path / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"task-a","branch":"wt-codex-task-a","created_at":"2026-03-17T00:00:00+00:00","last_opened_at":"2026-03-17T01:00:00+00:00","purpose":"Build / edit"}',
            encoding="utf-8",
        )
        stale_path = worktree_root / "claude" / "old-task"
        stale_path.mkdir(parents=True)
        (stale_path / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"claude","name":"old-task","branch":"wt-claude-old-task"}',
            encoding="utf-8",
        )

        active = [
            worktree_manager.WorktreeInfo(
                path=str(managed_path),
                head="abc123",
                branch="refs/heads/wt-codex-task-a",
            )
        ]

        with (
            patch.object(worktree_manager, "WORKTREE_ROOT", worktree_root),
            patch.object(worktree_manager, "list_worktrees", return_value=active),
            patch.object(worktree_manager, "worktree_status", return_value=[]),
        ):
            items = worktree_manager.list_managed_worktrees(tmp_path)

        assert len(items) == 1
        assert items[0].tool == "codex"
        assert items[0].name == "task-a"
        assert items[0].purpose == "Build / edit"
        assert items[0].dirty is False

    def test_read_metadata_for_returns_existing_metadata(self, tmp_path: Path) -> None:
        worktree_root = tmp_path / ".worktrees"
        managed_path = worktree_root / "tasks" / "codex" / "task-a"
        managed_path.mkdir(parents=True)
        (managed_path / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"task-a","purpose":"Review / verify"}',
            encoding="utf-8",
        )
        with patch.object(worktree_manager, "WORKTREE_ROOT", worktree_root):
            meta = worktree_manager.read_metadata_for("codex", "task-a")
        assert meta is not None
        assert meta["purpose"] == "Review / verify"

    def test_read_metadata_for_does_not_cross_tools_on_same_name(self, tmp_path: Path) -> None:
        worktree_root = tmp_path / ".worktrees"
        managed_path = worktree_root / "tasks" / "task-a"
        managed_path.mkdir(parents=True)
        (managed_path / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"claude","name":"task-a","purpose":"Review / verify"}',
            encoding="utf-8",
        )
        with patch.object(worktree_manager, "WORKTREE_ROOT", worktree_root):
            meta = worktree_manager.read_metadata_for("codex", "task-a")
        assert meta is None

    def test_list_managed_worktrees_sorts_by_last_opened_desc(self, tmp_path: Path) -> None:
        worktree_root = tmp_path / ".worktrees"
        first = worktree_root / "claude" / "older"
        second = worktree_root / "codex" / "newer"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"claude","name":"older","branch":"wt-claude-older","created_at":"2026-03-17T00:00:00+00:00","last_opened_at":"2026-03-17T01:00:00+00:00"}',
            encoding="utf-8",
        )
        (second / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"newer","branch":"wt-codex-newer","created_at":"2026-03-17T00:00:00+00:00","last_opened_at":"2026-03-17T02:00:00+00:00"}',
            encoding="utf-8",
        )

        active = [
            worktree_manager.WorktreeInfo(path=str(first), head="aaa", branch="refs/heads/wt-claude-older"),
            worktree_manager.WorktreeInfo(path=str(second), head="bbb", branch="refs/heads/wt-codex-newer"),
        ]
        with (
            patch.object(worktree_manager, "WORKTREE_ROOT", worktree_root),
            patch.object(worktree_manager, "list_worktrees", return_value=active),
            patch.object(worktree_manager, "worktree_status", return_value=[]),
        ):
            items = worktree_manager.list_managed_worktrees(tmp_path)

        assert [item.name for item in items] == ["newer", "older"]


class TestCreateClose:
    def test_create_worktree_reuses_existing_path(self, tmp_path: Path) -> None:
        existing = tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        existing.mkdir(parents=True)
        active = [worktree_manager.WorktreeInfo(path=str(existing))]
        with (
            patch.object(worktree_manager, "WORKTREE_ROOT", tmp_path / ".worktrees"),
            patch.object(worktree_manager, "list_worktrees", return_value=active),
        ):
            path = worktree_manager.create_worktree("codex", "foo", purpose="Build / edit")
        assert path == existing
        meta = json.loads((existing / worktree_manager.WORKTREE_META).read_text(encoding="utf-8"))
        assert meta["purpose"] == "Build / edit"
        assert meta["tool"] == "codex"
        assert meta["state"] == "active"

    def test_create_worktree_same_name_other_tool_gets_separate_path(self, tmp_path: Path) -> None:
        existing = tmp_path / ".worktrees" / "tasks" / "claude" / "foo"
        existing.mkdir(parents=True)
        (existing / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"claude","name":"foo","branch":"wt-claude-foo","base_ref":"HEAD"}',
            encoding="utf-8",
        )
        active = [worktree_manager.WorktreeInfo(path=str(existing))]

        def fake_run_git(*args: str, cwd: Path = worktree_manager.PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
            if args == ("worktree", "prune"):
                return subprocess.CompletedProcess(args, 0, "", "")
            if args[:3] == ("worktree", "add", "-b"):
                Path(args[4]).mkdir(parents=True, exist_ok=True)
                return subprocess.CompletedProcess(args, 0, "", "")
            raise AssertionError(f"Unexpected git call: {args} cwd={cwd}")

        with (
            patch.object(worktree_manager, "WORKTREE_ROOT", tmp_path / ".worktrees"),
            patch.object(worktree_manager, "list_worktrees", return_value=active),
            patch.object(worktree_manager, "_run_git", side_effect=fake_run_git),
        ):
            path = worktree_manager.create_worktree("codex", "foo", purpose="Build / edit")

        assert path == tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        assert path != existing
        meta = json.loads((path / worktree_manager.WORKTREE_META).read_text(encoding="utf-8"))
        assert meta["tool"] == "codex"
        assert meta["name"] == "foo"

    def test_create_worktree_rejects_stale_existing_path(self, tmp_path: Path) -> None:
        existing = tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        existing.mkdir(parents=True)
        with (
            patch.object(worktree_manager, "WORKTREE_ROOT", tmp_path / ".worktrees"),
            patch.object(worktree_manager, "list_worktrees", return_value=[]),
        ):
            with pytest.raises(RuntimeError, match="not active"):
                worktree_manager.create_worktree("codex", "foo")

    def test_close_worktree_rejects_main_repo(self) -> None:
        try:
            worktree_manager.close_worktree(worktree_manager.PROJECT_ROOT)
        except RuntimeError as exc:
            assert "main repo worktree" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError")

    def test_verify_dirty_close_requires_force(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        with patch.object(worktree_manager, "worktree_status", return_value=[" M file.py"]):
            try:
                worktree_manager.close_worktree(wt)
            except RuntimeError as exc:
                assert "uncommitted changes" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError")


class TestEnsureSymlinkCrossDrive:
    """Regression — os.path.relpath raises ValueError on Windows when the
    target and link_path.parent are on different drives (CI workspace on D:,
    pytest tmp_path on C:). ensure_symlink must fall back to an absolute
    target rather than propagating the ValueError.
    """

    def test_cross_drive_target_falls_back_to_absolute(self, tmp_path: Path) -> None:
        """When os.path.relpath raises ValueError, ensure_symlink must not crash."""
        target = tmp_path / "real_target"
        target.mkdir()
        link = tmp_path / "link_point"

        original_relpath = worktree_manager.os.path.relpath

        def relpath_cross_drive(path: str, start: str | None = None) -> str:
            raise ValueError("path is on mount 'D:', start on mount 'C:'")

        with patch.object(worktree_manager.os.path, "relpath", side_effect=relpath_cross_drive):
            # Must not raise. symlink creation itself may still silently fail
            # on OSes without symlink perms — that path is covered by the
            # existing except OSError, which is the intentional fail-open.
            worktree_manager.ensure_symlink(target, link)
        # relpath was restored
        assert worktree_manager.os.path.relpath is original_relpath


class TestWorkflowOperations:
    def test_handoff_worktree_updates_owner_and_state(self, tmp_path: Path) -> None:
        wt = tmp_path / ".worktrees" / "tasks" / "claude" / "foo"
        wt.mkdir(parents=True)
        (wt / worktree_manager.WORKTREE_META).write_text(
            json.dumps(
                {
                    "tool": "claude",
                    "name": "foo",
                    "branch": "wt-claude-foo",
                    "base_ref": "HEAD",
                    "created_at": "2026-03-17T00:00:00+00:00",
                    "last_opened_at": "2026-03-17T01:00:00+00:00",
                    "purpose": "Review / verify",
                }
            ),
            encoding="utf-8",
        )

        meta = worktree_manager.handoff_worktree(wt, target_tool="codex", purpose="Build / edit", note="pick up build")

        assert meta["tool"] == "codex"
        assert meta["state"] == "handoff"
        assert meta["handoff_note"] == "pick up build"
        assert meta["last_actor_tool"] == "claude"

    def test_ship_worktree_requires_commit_message_when_dirty(self, tmp_path: Path) -> None:
        wt = tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        wt.mkdir(parents=True)
        (wt / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"foo","branch":"wt-codex-foo","base_ref":"HEAD"}',
            encoding="utf-8",
        )

        with patch.object(worktree_manager, "worktree_status", return_value=[" M file.py"]):
            with pytest.raises(RuntimeError, match="Provide --commit-message"):
                worktree_manager.ship_worktree(wt)

    def test_ship_worktree_commits_merges_and_closes(self, tmp_path: Path) -> None:
        wt = tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        wt.mkdir(parents=True)
        (wt / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"foo","branch":"wt-codex-foo","base_ref":"HEAD"}',
            encoding="utf-8",
        )

        def fake_run_git(*args: str, cwd: Path = worktree_manager.PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
            if args == ("add", "-A"):
                return subprocess.CompletedProcess(args, 0, "", "")
            if args[:2] == ("commit", "-m"):
                return subprocess.CompletedProcess(args, 0, "committed", "")
            if args == ("branch", "--show-current"):
                return subprocess.CompletedProcess(args, 0, "wt-codex-foo\n", "")
            if args[:3] == ("merge", "--no-ff", "--no-edit"):
                return subprocess.CompletedProcess(args, 0, "merged", "")
            raise AssertionError(f"Unexpected git call: {args} cwd={cwd}")

        def fake_status(path: Path) -> list[str]:
            if path == wt:
                return [" M file.py"]
            if path == worktree_manager.PROJECT_ROOT:
                return []
            raise AssertionError(f"Unexpected status path: {path}")

        with (
            patch.object(worktree_manager, "worktree_status", side_effect=fake_status),
            patch.object(worktree_manager, "current_branch", return_value="main"),
            patch.object(worktree_manager, "_run_git", side_effect=fake_run_git),
            patch.object(worktree_manager, "close_worktree") as close_mock,
        ):
            result = worktree_manager.ship_worktree(wt, commit_message="workstream: foo")

        assert result["name"] == "foo"
        assert result["branch"] == "wt-codex-foo"
        close_mock.assert_called_once_with(wt, force=False, drop_branch=True)
