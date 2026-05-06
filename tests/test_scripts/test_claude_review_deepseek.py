"""Tests for `scripts/tools/claude_review_deepseek.py`.

Live Claude calls are NOT exercised in tests (cost + flakiness). Mock mode
produces deterministic verdicts; integration is verified manually via
Stage B criterion 6 in the Phase 3 stage doc.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWER = PROJECT_ROOT / "scripts" / "tools" / "claude_review_deepseek.py"


def _run(
    args: list[str],
    *,
    env_overrides: dict[str, str | None] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key, val in (env_overrides or {}).items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    return subprocess.run(
        [sys.executable, str(REVIEWER), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
    )


@pytest.fixture
def staged_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with a staged ~10-line code diff."""
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
    f = tmp_path / "thing.py"
    f.write_text("def thing():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "thing.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init", "--no-verify"], cwd=str(tmp_path), check=True)
    f.write_text(
        "def thing():\n    return 1\n\n\ndef other():\n    x = 1\n    y = 2\n    return x + y\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "thing.py"], cwd=str(tmp_path), check=True)
    return tmp_path


class TestClaudeReviewDeepseek:
    def test_skips_when_env_inactive(self, staged_repo: Path) -> None:
        result = _run(
            ["--mock", "--rubric-fail"],
            env_overrides={"OPENCODE_AGENT_ACTIVE": None},
            cwd=staged_repo,
        )
        assert result.returncode == 0
        # Inactive path should not even mention BLOCK or print findings.
        assert "BLOCK" not in result.stderr

    def test_mock_rubric_pass_returns_zero(self, staged_repo: Path) -> None:
        result = _run(
            ["--mock", "--rubric-pass"],
            env_overrides={"OPENCODE_AGENT_ACTIVE": "1"},
            cwd=staged_repo,
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"

    def test_mock_rubric_fail_returns_one_with_findings(self, staged_repo: Path) -> None:
        result = _run(
            ["--mock", "--rubric-fail"],
            env_overrides={"OPENCODE_AGENT_ACTIVE": "1"},
            cwd=staged_repo,
        )
        assert result.returncode == 1
        assert "BLOCK" in result.stderr
        assert "[HIGH]" in result.stderr or "[CRITICAL]" in result.stderr

    def test_skips_when_diff_is_empty(self, tmp_path: Path) -> None:
        # Empty repo, no staged diff.
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        result = _run(
            ["--mock", "--rubric-fail"],
            env_overrides={"OPENCODE_AGENT_ACTIVE": "1"},
            cwd=tmp_path,
        )
        assert result.returncode == 0

    def test_skips_when_diff_is_doc_only(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)
        f = tmp_path / "README.md"
        f.write_text("# init\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init", "--no-verify"], cwd=str(tmp_path), check=True)
        # Make the doc change >= 5 lines so the size filter doesn't shadow the doc-only filter.
        f.write_text("# init\n\n## section A\n\n## section B\n\n## section C\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True)
        result = _run(
            ["--mock", "--rubric-fail"],
            env_overrides={"OPENCODE_AGENT_ACTIVE": "1"},
            cwd=tmp_path,
        )
        # rubric-fail would normally exit 1; doc-only filter should short-circuit to 0.
        assert result.returncode == 0, f"stderr={result.stderr!r}"

    def test_threshold_logic_via_diff_helpers(self) -> None:
        # Direct unit test on the diff filters — robust to git diff header
        # noise across platforms. < 5 line diff and doc-only diff each
        # short-circuit the reviewer.
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
        try:
            import claude_review_deepseek as crd  # type: ignore
        finally:
            sys.path.pop(0)

        doc_only_diff = (
            "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n hello\n+world\n"
        )
        assert crd._diff_is_doc_only(doc_only_diff) is True

        code_only_diff = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n"
        assert crd._diff_is_doc_only(code_only_diff) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
