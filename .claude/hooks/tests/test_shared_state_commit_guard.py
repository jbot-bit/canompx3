"""Tests for `.claude/hooks/shared-state-commit-guard.py`.

The hook is a PreToolUse(Bash) guard. Each test drives it via subprocess + stdin
JSON, the same way Claude Code invokes it. Tests assert:

  - Non-shared-state paths exit 0 (pass)
  - Non-git-add/commit commands exit 0 (pass)
  - Safe subpaths (stages/, drafts/) exit 0 (pass)
  - Shared-state path with active peer stage_lock exits 2 (BLOCK)
  - Shared-state path with only CLOSED stage_lock exits 0 (pass)
  - --shared-state-ack flag bypasses BLOCK
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "shared-state-commit-guard.py"


def _run_hook(command: str, cwd: Path) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        timeout=10,
        check=False,
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Create a real git repo with a single commit so HEAD resolves."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        env={**os.environ, "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z"},
        check=True,
    )
    return repo


def _write_stage(repo: Path, slug: str, mode: str, scope_path: str) -> Path:
    stages = repo / "docs" / "runtime" / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    f = stages / f"{slug}.md"
    f.write_text(
        f"---\n"
        f"task: {slug}\n"
        f"mode: {mode}\n"
        f"scope_lock:\n  - {scope_path}\n"
        f"blast_radius: |\n  test stage claiming {scope_path} for unit-test coverage of guard\n"
        f"---\n",
        encoding="utf-8",
    )
    return f


def test_non_git_command_passes(fake_repo: Path):
    """Non-git commands should exit 0 without inspection."""
    r = _run_hook("ls -la", fake_repo)
    assert r.returncode == 0


def test_git_log_passes(fake_repo: Path):
    """git log / status / diff are not add/commit — pass."""
    r = _run_hook("git log --oneline -5", fake_repo)
    assert r.returncode == 0


def test_git_add_non_shared_state_passes(fake_repo: Path):
    """git add on a file outside SHARED_STATE_DIRS — pass."""
    r = _run_hook("git add pipeline/foo.py", fake_repo)
    assert r.returncode == 0


def test_git_add_safe_subpath_stages_passes(fake_repo: Path):
    """git add on docs/runtime/stages/... is safe — pass."""
    r = _run_hook("git add docs/runtime/stages/foo.md", fake_repo)
    assert r.returncode == 0


def test_git_add_safe_subpath_drafts_passes(fake_repo: Path):
    """git add on docs/audit/hypotheses/drafts/... is safe — pass."""
    r = _run_hook("git add docs/audit/hypotheses/drafts/foo.draft.yaml", fake_repo)
    assert r.returncode == 0


def test_git_add_shared_state_no_active_stage_passes(fake_repo: Path):
    """Shared-state path with no active claimant — pass (fail-safe; check 1+3 also clean)."""
    r = _run_hook("git add docs/runtime/fast_lane_trial_ledger.yaml", fake_repo)
    # Fresh repo, fresh lock absent → Check 1 silent (no lock), Check 2 silent
    # (no stage file), Check 3 silent (no sibling worktrees). Pass.
    assert r.returncode == 0


def test_git_add_shared_state_with_active_peer_stage_blocks(fake_repo: Path):
    """Active IMPLEMENTATION stage claiming the same path → BLOCK."""
    _write_stage(fake_repo, "peer-stage", "IMPLEMENTATION", "docs/runtime/fast_lane_trial_ledger.yaml")
    r = _run_hook("git add docs/runtime/fast_lane_trial_ledger.yaml", fake_repo)
    assert r.returncode == 2
    assert "Active stage(s) claim scope_lock" in r.stderr


def test_git_add_shared_state_with_closed_stage_passes(fake_repo: Path):
    """CLOSED stage claiming the same path is a canonical-source retainer, not in-flight work — pass."""
    _write_stage(fake_repo, "closed-stage", "CLOSED", "docs/runtime/fast_lane_trial_ledger.yaml")
    r = _run_hook("git add docs/runtime/fast_lane_trial_ledger.yaml", fake_repo)
    assert r.returncode == 0


def test_ack_flag_bypasses_block(fake_repo: Path):
    """--shared-state-ack in the command bypasses the guard."""
    _write_stage(fake_repo, "peer-stage", "IMPLEMENTATION", "docs/runtime/fast_lane_trial_ledger.yaml")
    r = _run_hook(
        "git add docs/runtime/fast_lane_trial_ledger.yaml  # --shared-state-ack",
        fake_repo,
    )
    assert r.returncode == 0


def test_git_commit_with_staged_shared_state_blocks(fake_repo: Path):
    """git commit (no path args) inspects --cached for shared-state files; with peer stage → BLOCK."""
    _write_stage(fake_repo, "peer-stage", "IMPLEMENTATION", "docs/runtime/fast_lane_trial_ledger.yaml")
    target = fake_repo / "docs" / "runtime" / "fast_lane_trial_ledger.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("entries: []\n", encoding="utf-8")
    subprocess.run(["git", "add", str(target)], cwd=fake_repo, check=True)
    r = _run_hook("git commit -m 'test'", fake_repo)
    assert r.returncode == 2
    assert "fast_lane_trial_ledger" in r.stderr


def test_audit_dir_also_guarded(fake_repo: Path):
    """docs/audit/ paths (non-drafts subdir) also fire the guard when claimed."""
    _write_stage(fake_repo, "audit-stage", "IMPLEMENTATION", "docs/audit/results/foo.md")
    r = _run_hook("git add docs/audit/results/foo.md", fake_repo)
    assert r.returncode == 2
