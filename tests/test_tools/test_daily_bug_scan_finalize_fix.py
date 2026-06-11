from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.tools import daily_bug_scan_finalize_fix as finalize


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("# repo\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "init")
    return root


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (" M HANDOFF.md", "HANDOFF.md"),
        ("M  scripts/tools/x.py", "scripts/tools/x.py"),
        ("?? scripts/tools/new.py", "scripts/tools/new.py"),
        ("R  old.py -> scripts/tools/new.py", "scripts/tools/new.py"),
        (" M path\\with\\slashes.py", "path/with/slashes.py"),
    ],
)
def test_path_from_porcelain_line(line: str, expected: str) -> None:
    assert finalize.path_from_porcelain_line(line) == expected


def test_split_dirty_paths_keeps_allowed_baton_out_of_package() -> None:
    package, allowed = finalize.split_dirty_paths(
        ["HANDOFF.md", "pipeline/check_drift.py", "tests/test_pipeline/test_x.py"],
        {"HANDOFF.md"},
    )

    assert package == ["pipeline/check_drift.py", "tests/test_pipeline/test_x.py"]
    assert allowed == ["HANDOFF.md"]


def test_ensure_fix_branch_creates_branch_from_current_head(git_repo: Path) -> None:
    finalize.ensure_fix_branch(git_repo, "codex/daily-bug-scan-test")

    assert _git(git_repo, "branch", "--show-current") == "codex/daily-bug-scan-test"


def test_ensure_fix_branch_refuses_existing_different_branch(git_repo: Path) -> None:
    _git(git_repo, "branch", "codex/existing")

    with pytest.raises(RuntimeError, match="already exists"):
        finalize.ensure_fix_branch(git_repo, "codex/existing")


def test_finalize_dry_run_reports_package_without_commit(git_repo: Path) -> None:
    (git_repo / "scripts" / "tools").mkdir(parents=True)
    (git_repo / "scripts" / "tools" / "fix.py").write_text("print('fix')\n", encoding="utf-8")
    (git_repo / "HANDOFF.md").write_text("baton\n", encoding="utf-8")

    result = finalize.finalize_fix(
        root=git_repo,
        branch="codex/daily-bug-scan-test",
        message="fix(test): package",
        verify=["python -c \"print('verified')\""],
        allowed_dirty=["HANDOFF.md"],
        main_repo=git_repo,
        first_run_minutes=1,
        interval_minutes=10,
        no_register=True,
        dry_run=True,
    )

    assert result.committed is False
    assert result.fix_sha is None
    assert result.registered_auto_merge is False
    assert result.committed_paths == ["scripts/tools/fix.py"]
    assert result.allowed_dirty_paths == ["HANDOFF.md"]
    assert _git(git_repo, "branch", "--show-current") == "codex/daily-bug-scan-test"
