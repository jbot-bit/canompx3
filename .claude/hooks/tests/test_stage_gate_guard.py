"""Tests for `.claude/hooks/stage-gate-guard.py` worktree-aware path resolution.

The hook is shell-invoked; tests drive it via subprocess + stdin JSON, the same
way Claude Code invokes it. Each test sets up a synthetic worktree on disk and
asserts the hook reads stage files from the correct one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


HOOK = Path(__file__).resolve().parents[1] / "stage-gate-guard.py"


def _run_hook(file_path: str, cwd: Path) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_input": {"file_path": file_path}})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        timeout=10,
        check=False,
    )


def _write_stage(stages_dir: Path, slug: str, mode: str, scope: list[str], blast: str) -> Path:
    stages_dir.mkdir(parents=True, exist_ok=True)
    f = stages_dir / f"{slug}.md"
    scope_block = "\n".join(f"  - {p}" for p in scope)
    f.write_text(
        f"---\n"
        f"task: {slug}\n"
        f"mode: {mode}\n"
        f"scope_lock:\n{scope_block}\n"
        f"blast_radius: {blast}\n"
        f"---\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def fake_worktree(tmp_path: Path) -> Path:
    """Create a fake worktree: directory tree with `.git` as a file (worktree marker)."""
    wt = tmp_path / "wt-feature"
    (wt / "pipeline").mkdir(parents=True)
    (wt / ".git").write_text("gitdir: /fake/path/to/main/.git/worktrees/wt-feature\n")
    return wt


@pytest.fixture
def fake_main(tmp_path: Path) -> Path:
    """Create a fake main checkout: directory tree with `.git` as a directory."""
    main = tmp_path / "main"
    (main / "pipeline").mkdir(parents=True)
    (main / ".git").mkdir()
    return main


def test_resolves_to_worktree_with_git_file(fake_worktree: Path, fake_main: Path) -> None:
    """Edit in worktree-X resolves to worktree-X's stages, not main's."""
    _write_stage(
        fake_worktree / "docs" / "runtime" / "stages",
        "feature-task",
        "IMPLEMENTATION",
        ["pipeline/dst.py"],
        "blast: pipeline/dst.py and tests/test_dst.py — covers session catalog reads",
    )
    edited = fake_worktree / "pipeline" / "dst.py"
    edited.write_text("# touch\n")

    # Run from main's CWD — proves CWD is irrelevant, file_path drives resolution
    result = _run_hook(str(edited), cwd=fake_main)

    assert result.returncode == 0, f"expected pass, got rc={result.returncode}, stderr={result.stderr!r}"


def test_resolves_to_main_with_git_dir(fake_main: Path, fake_worktree: Path) -> None:
    """Edit in main resolves to main's stages, not any sibling worktree's."""
    _write_stage(
        fake_main / "docs" / "runtime" / "stages",
        "main-task",
        "IMPLEMENTATION",
        ["pipeline/dst.py"],
        "blast: pipeline/dst.py and tests/test_dst.py — covers session catalog reads",
    )
    edited = fake_main / "pipeline" / "dst.py"
    edited.write_text("# touch\n")

    # Worktree exists but does NOT have a stage file for this scope — guard
    # must read main's stage file, not be confused by the sibling worktree.
    result = _run_hook(str(edited), cwd=fake_worktree)

    assert result.returncode == 0, f"expected pass, got rc={result.returncode}, stderr={result.stderr!r}"


def test_blocks_when_worktree_has_no_stage_for_path(fake_worktree: Path, fake_main: Path) -> None:
    """Worktree with no covering stage blocks edits to NEVER_TRIVIAL files,
    even if main has a covering stage. This is the GLOBAL-mode-rule retirement
    in action: each worktree gates independently.

    NOTE: pipeline/dst.py is in NEVER_TRIVIAL — auto-trivial cannot help, so
    the absence of a covering stage in the editing worktree must hard-block.
    """
    # Main has a covering stage, but the EDIT happens in fake_worktree
    _write_stage(
        fake_main / "docs" / "runtime" / "stages",
        "main-task",
        "IMPLEMENTATION",
        ["pipeline/dst.py"],
        "blast: pipeline/dst.py and tests/test_dst.py — covers session catalog reads",
    )
    edited = fake_worktree / "pipeline" / "dst.py"
    edited.write_text("# touch\n")

    result = _run_hook(str(edited), cwd=fake_main)

    assert result.returncode == 2, (
        f"expected hard-block, got rc={result.returncode}, stderr={result.stderr!r}"
    )


def test_auto_trivial_writes_to_editing_worktree(fake_worktree: Path, fake_main: Path) -> None:
    """When no stage exists and file is non-core, auto_trivial.md must be
    written to the editing worktree's stages dir, not main's. Otherwise the
    main worktree accumulates auto_trivial cruft from sibling edits."""
    edited = fake_worktree / "pipeline" / "non_core_helper.py"
    edited.write_text("# touch\n")

    result = _run_hook(str(edited), cwd=fake_main)

    # rc=0 means hook passed; auto_trivial should be in fake_worktree's stages
    assert result.returncode == 0, f"expected pass, got rc={result.returncode}, stderr={result.stderr!r}"
    assert (fake_worktree / "docs" / "runtime" / "stages" / "auto_trivial.md").exists()
    assert not (fake_main / "docs" / "runtime" / "stages" / "auto_trivial.md").exists()


def test_falls_back_when_no_git_found(tmp_path: Path) -> None:
    """File outside any git checkout — fall back to CWD-relative paths.
    Hook must not crash; behavior matches pre-fix CWD-relative read."""
    # No .git anywhere in tmp_path tree
    edited = tmp_path / "stray" / "pipeline" / "thing.py"
    edited.parent.mkdir(parents=True)
    edited.write_text("# touch\n")

    # CWD also has no stage state — guard auto-creates trivial there
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    result = _run_hook(str(edited), cwd=cwd)

    # Behavior: non-core file + no stages → auto-trivial. Either way, no crash.
    assert result.returncode in (0, 2), (
        f"unexpected rc={result.returncode}, stderr={result.stderr!r}"
    )


def test_empty_file_path_does_not_crash(tmp_path: Path) -> None:
    """Defensive: empty/missing file_path must not crash resolution."""
    result = _run_hook("", cwd=tmp_path)
    # Empty path is normalized to '' which won't match any PROD_PATHS → pass
    assert result.returncode == 0, f"unexpected rc={result.returncode}, stderr={result.stderr!r}"
