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

    def test_opencode_worktree_path(self) -> None:
        path = worktree_manager.build_worktree_path("opencode", "Third POV")
        assert path.parts[-4:] == (".worktrees", "tasks", "opencode", "third-pov")

    def test_parser_accepts_opencode_tool(self) -> None:
        parser = worktree_manager.build_parser()
        args = parser.parse_args(["create", "--tool", "opencode", "--name", "review-a"])
        assert args.tool == "opencode"


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

    def test_create_opencode_worktree_same_name_gets_separate_path(self, tmp_path: Path) -> None:
        existing = tmp_path / ".worktrees" / "tasks" / "codex" / "foo"
        existing.mkdir(parents=True)
        (existing / worktree_manager.WORKTREE_META).write_text(
            '{"tool":"codex","name":"foo","branch":"wt-codex-foo","base_ref":"HEAD"}',
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
            path = worktree_manager.create_worktree("opencode", "foo", purpose="Third POV build")

        assert path == tmp_path / ".worktrees" / "tasks" / "opencode" / "foo"
        assert path != existing
        meta = json.loads((path / worktree_manager.WORKTREE_META).read_text(encoding="utf-8"))
        assert meta["tool"] == "opencode"
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


class TestPhantomCwdReconcile:
    """Husk (force-removed worktree dir) detection + auto-heal.

    All filesystem work is under tmp_path; both list_worktrees and
    _canonical_main_root are patched so tests never touch the real lease or the
    developer's real sibling worktrees.
    """

    @staticmethod
    def _scratch_husk(path: Path) -> Path:
        """A scaffold/scratch-only husk: meta + .claude + a chart PNG, NO .git."""
        path.mkdir(parents=True)
        (path / worktree_manager.WORKTREE_META).write_text("{}", encoding="utf-8")
        (path / ".claude").mkdir()
        (path / "chart.png").write_bytes(b"\x89PNG")
        return path

    @staticmethod
    def _source_husk(path: Path) -> Path:
        """A full-tree husk that may hold uncommitted work: real source, NO .git."""
        path.mkdir(parents=True)
        (path / "foo.py").write_text("x = 1\n", encoding="utf-8")
        (path / "pipeline").mkdir()
        return path

    # -- _canonical_main_root ------------------------------------------------ #
    def test_canonical_main_root_is_git_common_dir_parent(self, tmp_path: Path) -> None:
        common = tmp_path / "canompx3" / ".git"
        common.mkdir(parents=True)

        def fake_run_git(*args: str, cwd: Path = worktree_manager.PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
            assert args == ("rev-parse", "--git-common-dir")
            return subprocess.CompletedProcess(args, 0, str(common) + "\n", "")

        with patch.object(worktree_manager, "_run_git", side_effect=fake_run_git):
            assert worktree_manager._canonical_main_root() == common.parent

    # -- is_registered_worktree ---------------------------------------------- #
    def test_is_registered_true_when_in_list(self, tmp_path: Path) -> None:
        wt = tmp_path / "canompx3-foo"
        wt.mkdir()
        active = [worktree_manager.WorktreeInfo(path=str(wt))]
        with patch.object(worktree_manager, "list_worktrees", return_value=active):
            assert worktree_manager.is_registered_worktree(wt, root=tmp_path) is True

    def test_is_registered_false_when_absent_from_list(self, tmp_path: Path) -> None:
        wt = tmp_path / "canompx3-husk"
        wt.mkdir()
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            assert worktree_manager.is_registered_worktree(wt, root=tmp_path) is False

    def test_is_registered_failclosed_on_runtimeerror(self, tmp_path: Path) -> None:
        wt = tmp_path / "canompx3-x"
        wt.mkdir()
        with patch.object(worktree_manager, "list_worktrees", side_effect=RuntimeError("git boom")):
            assert worktree_manager.is_registered_worktree(wt, root=tmp_path) is False

    def test_precomputed_registered_set_skips_git(self, tmp_path: Path) -> None:
        """Passing a precomputed `registered` set makes NO git call — this is the
        cheap-sweep path that turns ~2N `git worktree list` spawns into 1."""
        wt = tmp_path / "canompx3-foo"
        wt.mkdir()
        reg = {worktree_manager._norm(wt)}
        with patch.object(worktree_manager, "list_worktrees", side_effect=AssertionError("git called")):
            assert worktree_manager.is_registered_worktree(wt, registered=reg) is True
            assert worktree_manager.is_registered_worktree(tmp_path / "canompx3-bar", registered=reg) is False

    def test_reap_resolves_registry_once(self, tmp_path: Path) -> None:
        """reap_graveyards must call list_worktrees exactly ONCE regardless of how
        many siblings it scans (the per-sibling-git regression guard)."""
        base = tmp_path / "canompx3"
        base.mkdir()
        for i in range(4):
            self._scratch_husk(tmp_path / f"canompx3-husk{i}")
        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return []

        with patch.object(worktree_manager, "list_worktrees", side_effect=counting):
            worktree_manager.reap_graveyards(root=base, execute=False)
        assert calls["n"] == 1, f"expected 1 git call, got {calls['n']}"

    # -- _is_scratch_only ---------------------------------------------------- #
    def test_is_scratch_only_true_for_scaffold(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "husk")
        assert worktree_manager._is_scratch_only(husk) is True

    def test_is_scratch_only_false_for_source_file(self, tmp_path: Path) -> None:
        husk = tmp_path / "husk"
        husk.mkdir()
        (husk / "foo.py").write_text("x = 1\n", encoding="utf-8")
        assert worktree_manager._is_scratch_only(husk) is False

    def test_is_scratch_only_false_for_source_dir(self, tmp_path: Path) -> None:
        husk = tmp_path / "husk"
        husk.mkdir()
        (husk / "pipeline").mkdir()
        assert worktree_manager._is_scratch_only(husk) is False

    # -- is_safe_graveyard --------------------------------------------------- #
    def test_safe_graveyard_true_for_unregistered_scratch_husk(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "canompx3-husk")
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            assert worktree_manager.is_safe_graveyard(husk, root=tmp_path) is True

    def test_safe_graveyard_false_when_dotgit_file_present(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "canompx3-live")
        (husk / ".git").write_text("gitdir: ...\n", encoding="utf-8")  # worktree .git is a FILE
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            assert worktree_manager.is_safe_graveyard(husk, root=tmp_path) is False

    def test_safe_graveyard_false_when_registered(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "canompx3-reg")
        active = [worktree_manager.WorktreeInfo(path=str(husk))]
        with patch.object(worktree_manager, "list_worktrees", return_value=active):
            assert worktree_manager.is_safe_graveyard(husk, root=tmp_path) is False

    def test_safe_graveyard_false_when_has_source(self, tmp_path: Path) -> None:
        husk = self._source_husk(tmp_path / "canompx3-work")
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            assert worktree_manager.is_safe_graveyard(husk, root=tmp_path) is False

    # -- reconcile_launch_path ----------------------------------------------- #
    def test_reconcile_registered(self, tmp_path: Path) -> None:
        wt = tmp_path / "canompx3-foo"
        wt.mkdir()
        active = [worktree_manager.WorktreeInfo(path=str(wt))]
        with patch.object(worktree_manager, "list_worktrees", return_value=active):
            final, action = worktree_manager.reconcile_launch_path(wt, root=tmp_path)
        assert action == "REGISTERED"
        assert final == wt

    def test_reconcile_absent(self, tmp_path: Path) -> None:
        wt = tmp_path / "canompx3-new"
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            final, action = worktree_manager.reconcile_launch_path(wt, root=tmp_path)
        assert action == "ABSENT"
        assert final == wt

    def test_reconcile_cleaned_for_scratch_husk(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "canompx3-husk")
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            final, action = worktree_manager.reconcile_launch_path(husk, root=tmp_path)
        assert action == "CLEANED"
        assert final == husk
        assert not husk.exists()  # safe husk removed; path reusable

    def test_reconcile_repathed_preserves_source_husk(self, tmp_path: Path) -> None:
        husk = self._source_husk(tmp_path / "canompx3-work")
        with patch.object(worktree_manager, "list_worktrees", return_value=[]):
            final, action = worktree_manager.reconcile_launch_path(husk, root=tmp_path)
        assert action == "REPATHED"
        assert final != husk
        assert husk.exists()  # work-loss guard: original untouched
        assert (husk / "foo.py").exists()

    def test_reconcile_repathed_when_rmtree_fails(self, tmp_path: Path) -> None:
        husk = self._scratch_husk(tmp_path / "canompx3-locked")
        with (
            patch.object(worktree_manager, "list_worktrees", return_value=[]),
            patch.object(worktree_manager.shutil, "rmtree", side_effect=OSError("locked")),
        ):
            final, action = worktree_manager.reconcile_launch_path(husk, root=tmp_path)
        assert action == "REPATHED"
        assert final != husk
        assert husk.exists()  # rmtree failed -> original preserved

    # -- reap_graveyards ----------------------------------------------------- #
    def test_reap_preview_removes_nothing(self, tmp_path: Path) -> None:
        main = tmp_path / "canompx3"
        main.mkdir()
        safe = self._scratch_husk(tmp_path / "canompx3-husk")
        work = self._source_husk(tmp_path / "canompx3-work")
        with (
            patch.object(worktree_manager, "_canonical_main_root", return_value=main),
            patch.object(worktree_manager, "list_worktrees", return_value=[]),
        ):
            acted, skipped = worktree_manager.reap_graveyards(execute=False)
        assert safe in acted
        assert (work, "MANUAL") in skipped
        assert safe.exists() and work.exists()  # preview deletes NOTHING

    def test_reap_execute_removes_only_safe_husk(self, tmp_path: Path) -> None:
        main = tmp_path / "canompx3"
        main.mkdir()
        safe = self._scratch_husk(tmp_path / "canompx3-husk")
        work = self._source_husk(tmp_path / "canompx3-work")
        registered = self._scratch_husk(tmp_path / "canompx3-reg")
        active = [worktree_manager.WorktreeInfo(path=str(registered))]
        with (
            patch.object(worktree_manager, "_canonical_main_root", return_value=main),
            patch.object(worktree_manager, "list_worktrees", return_value=active),
        ):
            acted, skipped = worktree_manager.reap_graveyards(execute=True)
        assert safe in acted and not safe.exists()  # safe husk removed
        assert (work, "MANUAL") in skipped and work.exists()  # work preserved
        assert (registered, "REGISTERED") in skipped and registered.exists()
        assert main.exists()  # never touches the canonical root itself
