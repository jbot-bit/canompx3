"""Tests for `.claude/hooks/judgment-review-soft-block.py`.

PreToolUse(Bash) guard. Drives the hook via subprocess + stdin JSON, the
same way Claude Code invokes it. Each test stages a fixture file in a
tempdir-scoped git repo so `git diff --cached --name-only` produces
deterministic output. Asserts on (returncode, stderr) pairs.

Reconciled scope path: the stage file's literal `scope_lock` line names
`tests/test_hooks/test_judgment_review_soft_block.py`, but every existing
hook test lives under `.claude/hooks/tests/` per the established repo
convention (`test_shared_state_commit_guard.py`, `test_stage_gate_guard.py`,
etc.). Placing this file here is the right move; the stage file is updated
in the same commit to match.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "judgment-review-soft-block.py"


def _run_hook(command: str, cwd: Path, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    env = {**os.environ}
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(cwd),
        timeout=10,
        check=False,
        env=env,
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Real git repo with a seed commit so HEAD resolves and staging works."""
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
        env={
            **os.environ,
            "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z",
        },
        check=True,
    )
    return repo


def _stage_capital_file(repo: Path, relpath: str = "trading_app/live/foo.py") -> Path:
    """Create + stage a file under a capital-class prefix."""
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# new\n", encoding="utf-8")
    subprocess.run(["git", "add", relpath], cwd=repo, check=True)
    return target


def _stage_noncapital_file(repo: Path, relpath: str = "docs/notes.md") -> Path:
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# new\n", encoding="utf-8")
    subprocess.run(["git", "add", relpath], cwd=repo, check=True)
    return target


def test_judgment_capital_clean_body_blocks(fake_repo: Path):
    """[judgment] + capital-class staged path + clean body → BLOCK (exit 2)."""
    _stage_capital_file(fake_repo)
    r = _run_hook('git commit -m "[judgment] HIGH: touch live/foo"', fake_repo)
    assert r.returncode == 2, f"expected BLOCK; got rc={r.returncode}, stderr={r.stderr!r}"
    assert "/code-review" in r.stderr
    assert "/capital-review" in r.stderr  # live-trading path also names capital-review


def test_judgment_capital_with_review_mention_passes(fake_repo: Path):
    """Body mentions code-review → ALLOW (exit 0)."""
    _stage_capital_file(fake_repo)
    r = _run_hook(
        'git commit -m "[judgment] HIGH: touch live/foo, includes code-review pass"',
        fake_repo,
    )
    assert r.returncode == 0, f"expected ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_override_flag_bypasses_block(fake_repo: Path):
    """Trailing `# --audit-acknowledged` → ALLOW."""
    _stage_capital_file(fake_repo)
    r = _run_hook(
        'git commit -m "[judgment] HIGH: touch live/foo"  # --audit-acknowledged',
        fake_repo,
    )
    assert r.returncode == 0, f"expected override ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_fresh_marker_file_suppresses(fake_repo: Path, tmp_path: Path):
    """Recent `.judgment-review-ts` marker → ALLOW."""
    _stage_capital_file(fake_repo)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = scratch / ".judgment-review-ts"
    marker.write_text("", encoding="utf-8")
    # Ensure mtime is fresh (within 60min window).
    now = time.time()
    os.utime(marker, (now, now))
    r = _run_hook(
        'git commit -m "[judgment] HIGH: touch live/foo"',
        fake_repo,
        env_overrides={"JUDGMENT_REVIEW_SCRATCH_DIR": str(scratch)},
    )
    assert r.returncode == 0, f"expected marker ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_non_capital_path_passes(fake_repo: Path):
    """[judgment] but only docs/ staged → ALLOW."""
    _stage_noncapital_file(fake_repo)
    r = _run_hook('git commit -m "[judgment] MEDIUM: doc tweak"', fake_repo)
    assert r.returncode == 0, f"expected non-capital ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_no_judgment_tag_passes(fake_repo: Path):
    """No [judgment] tag → ALLOW even on capital-class paths (nudge handles this case post-commit)."""
    _stage_capital_file(fake_repo)
    r = _run_hook('git commit -m "feat: touch live/foo"', fake_repo)
    assert r.returncode == 0, f"expected no-tag ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_non_commit_bash_passes(fake_repo: Path):
    """Command without `git commit` → ALLOW."""
    r = _run_hook("ls -la", fake_repo)
    assert r.returncode == 0
    r2 = _run_hook("git log --oneline -5", fake_repo)
    assert r2.returncode == 0
    r3 = _run_hook("git status", fake_repo)
    assert r3.returncode == 0


def test_subprocess_failure_fails_open(tmp_path: Path):
    """Running outside a git repo → `git diff --cached` fails → fail-open ALLOW."""
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    r = _run_hook('git commit -m "[judgment] HIGH: touch live/foo"', not_a_repo)
    assert r.returncode == 0, f"expected fail-open ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_override_flag_stripped_before_predicates(fake_repo: Path):
    """The override token must be stripped before any predicate runs.

    Verify by staging a capital file + tagging [judgment] (would normally
    block), then appending the override — if the override isn't stripped
    early, the block fires. We check stdout/stderr have NO block banner.
    """
    _stage_capital_file(fake_repo)
    cmd = 'git commit -m "[judgment] HIGH: touch live/foo"   # --audit-acknowledged'
    r = _run_hook(cmd, fake_repo)
    assert r.returncode == 0
    assert "JUDGMENT-REVIEW SOFT-BLOCK" not in r.stderr
    assert "/code-review" not in r.stderr


def test_amend_skipped(fake_repo: Path):
    """`git commit --amend` is skipped (consistent with the nudge's _looks_like_commit)."""
    _stage_capital_file(fake_repo)
    r = _run_hook('git commit --amend -m "[judgment] HIGH: touch live/foo"', fake_repo)
    assert r.returncode == 0, f"expected amend ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_message_file_unreadable_fails_open(fake_repo: Path):
    """`-F nonexistent.txt` → unreadable message → fail-open ALLOW."""
    _stage_capital_file(fake_repo)
    r = _run_hook("git commit -F /nonexistent/path/that/does/not/exist.txt", fake_repo)
    assert r.returncode == 0, f"expected -F fail-open ALLOW; got rc={r.returncode}, stderr={r.stderr!r}"


def test_pipeline_path_also_capital_class(fake_repo: Path):
    """`pipeline/` paths are capital-class per the nudge's prefix list — should BLOCK."""
    _stage_capital_file(fake_repo, relpath="pipeline/some_module.py")
    r = _run_hook('git commit -m "[judgment] CRITICAL: pipeline change"', fake_repo)
    assert r.returncode == 2
    # For pipeline-only paths (no live/), primary skill is /code-review
    assert "/code-review" in r.stderr
