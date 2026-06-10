"""Tests for .claude/hooks/worktree-destroy-guard.py.

Stage 2a of the fleet-state-brain plan: the destructive-op brain guard.

This hook closes the TARGET-BLINDNESS gap in `worktree_guard.py`: that hook
resolves the lease on the INVOKING tree (`acquire(cwd, ...)`) and never inspects
the tree that `git worktree remove <victim>` actually destroys. This guard fires
on `git worktree remove` / `git branch -D` and consults the canonical brain
(`fleet_state`) about the TARGET — blocking when the target carries work-at-risk
(NEEDS_FINISH) or a live peer (LIVE), or when a decision-time unpushed re-check
finds unpushed commits.

Polarity note (load-bearing): this is a DESTRUCTION guard, so fail-open is
INVERTED vs the mutex guards. A parse error / unknown shape must CONSULT the
brain anyway (conservative), never silently allow. Only a genuine brain
unavailability (import failure) exits 0.

Mirrors the in-process + subprocess two-layer style of test_mcp_git_guard.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HOOK_PATH = PROJECT_ROOT / ".claude" / "hooks" / "worktree-destroy-guard.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("worktree_destroy_guard", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _event(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command}}


def _state(
    *,
    path: str = "/c/victim",
    branch: str | None = "feature/x",
    classification: str = "HEALTHY",
    unpushed: int = 0,
    reasons: list[str] | None = None,
) -> SimpleNamespace:
    """A duck-typed stand-in for fleet_state.WorktreeState (only the fields the
    hook reads). Avoids importing the real dataclass into every test."""
    return SimpleNamespace(
        path=path,
        branch=branch,
        classification=classification,
        unpushed=unpushed,
        reasons=reasons or [],
    )


# ── Command-target parsing ──────────────────────────────────────────────────


class TestTargetParsing:
    def test_non_destruction_bash_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git status"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_non_bash_tool_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        event = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}}
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(event)))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_parse_worktree_remove_target(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git worktree remove /c/victim")
        assert op == "worktree"
        assert target == "/c/victim"

    def test_parse_worktree_remove_force_target(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git worktree remove --force /c/victim")
        assert op == "worktree"
        assert target == "/c/victim"

    def test_parse_worktree_remove_quoted_target_with_spaces(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target('git worktree remove "C:/tmp/my tree"')
        assert op == "worktree"
        assert target == "C:/tmp/my tree"

    def test_parse_branch_delete_target(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git branch -D feature/x")
        assert op == "branch"
        assert target == "feature/x"

    def test_parse_branch_delete_quoted_target_with_spaces(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target('git branch -D "feature with space"')
        assert op == "branch"
        assert target == "feature with space"

    def test_parse_branch_force_delete_capital_d(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git branch --delete --force feature/x")
        assert op == "branch"
        assert target == "feature/x"

    def test_parse_ignores_lowercase_branch_d_safe_delete(self) -> None:
        # `git branch -d` (safe delete) refuses on unmerged work itself; we still
        # guard it the same way — it is a branch destruction.
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git branch -d feature/x")
        assert op == "branch"
        assert target == "feature/x"

    def test_parse_non_destroy_worktree_subcommand(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git worktree list")
        assert op is None

    def test_parse_branch_list_is_not_destroy(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git branch --list")
        assert op is None

    def test_parse_handles_git_C_prefix(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target("git -C /c/repo worktree remove /c/victim")
        assert op == "worktree"
        assert target == "/c/victim"

    def test_parse_handles_quoted_worktree_target_with_spaces(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target('git worktree remove --force "C:/tmp/tree with spaces"')
        assert op == "worktree"
        assert target == "C:/tmp/tree with spaces"

    def test_parse_handles_quoted_git_c_prefix_and_target(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target('Git -C "C:/repo root" worktree remove --force "C:/tmp/tree with spaces"')
        assert op == "worktree"
        assert target == "C:/tmp/tree with spaces"

    def test_parse_handles_quoted_branch_name_with_spaces(self) -> None:
        hook = _load_hook()
        op, target = hook._parse_destroy_target('git branch -D "feature/branch with spaces"')
        assert op == "branch"
        assert target == "feature/branch with spaces"

    @pytest.mark.parametrize("binary", ["GIT", "Git", "git.EXE", "GIT.exe"])
    def test_parse_case_insensitive_git_binary(self, binary: str) -> None:
        """Windows resolves GIT/Git/git.EXE to the same binary — a case-exact
        match silently bypassed the guard (audit finding 2026-06-06)."""
        hook = _load_hook()
        op, target = hook._parse_destroy_target(f"{binary} worktree remove /c/victim")
        assert op == "worktree"
        assert target == "/c/victim"


# ── Verdict matrix (BLOCK vs ALLOW on the TARGET) ───────────────────────────


class TestVerdict:
    def test_block_needs_finish(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        hook = _load_hook()
        target = _state(
            path="/c/victim",
            classification="NEEDS_FINISH",
            reasons=["3 unpushed commit(s)"],
        )
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: target)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 0)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git worktree remove --force /c/victim"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "BLOCKED" in err
        assert "NEEDS_FINISH" in err
        assert "unpushed" in err

    def test_block_live(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        hook = _load_hook()
        target = _state(path="/c/victim", classification="LIVE", reasons=["live peer heartbeat in this tree"])
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: target)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 0)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git worktree remove /c/victim"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        assert "LIVE" in capsys.readouterr().err

    def test_block_on_live_unpushed_even_if_class_clean(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Decision-time re-check OUTRANKS a clean fleet_state classification:
        fleet_state.unpushed can be a swallowed-except 0 / stale origin/main."""
        hook = _load_hook()
        target = _state(path="/c/victim", classification="HEALTHY", unpushed=0)
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: target)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 2)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git worktree remove /c/victim"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        assert "unpushed" in capsys.readouterr().err.lower()

    @pytest.mark.parametrize("cls", ["HOLLOW", "MERGED", "HEALTHY", "STALE"])
    def test_allow_reapable_classes(self, cls: str, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        target = _state(path="/c/victim", classification=cls, unpushed=0)
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: target)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 0)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git worktree remove /c/victim"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0, f"{cls} should be reapable"


# ── Fail-open polarity (INVERTED for a destruction guard) ───────────────────


class TestFailOpenPolarity:
    def test_brain_import_failure_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Genuine brain unavailability is the ONLY exit-0-on-uncertainty path."""
        hook = _load_hook()
        monkeypatch.setattr(hook, "_load_fleet_state", lambda: None)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 0)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git worktree remove /c/victim"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_unresolvable_target_consults_unpushed_not_silent_allow(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """If fleet_state can't classify the target (None), the guard does NOT
        silently allow — it falls through to the direct unpushed probe. If that
        finds unpushed commits, BLOCK."""
        hook = _load_hook()
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: None)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 1)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git branch -D feature/x"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 2
        assert "unpushed" in capsys.readouterr().err.lower()

    def test_unresolvable_target_no_unpushed_allows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr(hook, "_resolve_target_state", lambda op, t: None)
        monkeypatch.setattr(hook, "_live_unpushed_count", lambda op, t, st: 0)
        monkeypatch.setattr("sys.stdin", StringIO(json.dumps(_event("git branch -D feature/x"))))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0

    def test_malformed_event_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hook = _load_hook()
        monkeypatch.setattr("sys.stdin", StringIO("not json"))
        with pytest.raises(SystemExit) as exc:
            hook.main()
        assert exc.value.code == 0


# ── Live end-to-end repro (real git, real fleet_state) ──────────────────────


class TestLiveRepro:
    """Stage a real worktree with an unpushed commit, attempt to remove it via
    the real hook subprocess, assert BLOCK. This exercises the actual
    fleet_state import + git -C probe path end-to-end (no monkeypatch)."""

    def _run(self, command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        event = _event(command)
        return subprocess.run(
            [sys.executable, str(HOOK_PATH)],
            input=json.dumps(event),
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )

    def test_subprocess_blocks_remove_of_tree_with_unpushed_commit(self, tmp_path: Path) -> None:
        # A bare "remote" + a clone with a worktree holding an unpushed commit.
        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True, capture_output=True)
        main_wt = tmp_path / "main"
        subprocess.run(
            ["git", "clone", "-q", str(remote), str(main_wt)], check=True, capture_output=True
        )
        subprocess.run(["git", "-C", str(main_wt), "config", "user.email", "t@t"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(main_wt), "config", "user.name", "t"], check=True, capture_output=True)
        (main_wt / "f.txt").write_text("base", encoding="utf-8")
        subprocess.run(["git", "-C", str(main_wt), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(main_wt), "commit", "-qm", "base"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(main_wt), "push", "-q", "origin", "HEAD"], check=True, capture_output=True)
        # A worktree on a new branch with a commit NOT pushed.
        victim = tmp_path / "victim"
        subprocess.run(
            ["git", "-C", str(main_wt), "worktree", "add", "-q", "-b", "feat", str(victim)],
            check=True,
            capture_output=True,
        )
        (victim / "g.txt").write_text("unpushed work", encoding="utf-8")
        subprocess.run(["git", "-C", str(victim), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(victim), "commit", "-qm", "unpushed"], check=True, capture_output=True)

        result = self._run(f"git -C {main_wt} worktree remove --force {victim}", cwd=main_wt)
        # The unpushed-commit re-check must fire regardless of how fleet_state
        # classified the tree against this throwaway repo's base.
        assert result.returncode == 2, f"expected BLOCK; got {result.returncode}\n{result.stderr}"
        assert "unpushed" in result.stderr.lower()

    def test_subprocess_allows_non_destroy_command(self, tmp_path: Path) -> None:
        result = self._run("git status", cwd=tmp_path)
        assert result.returncode == 0
