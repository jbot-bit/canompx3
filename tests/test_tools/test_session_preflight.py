"""Tests for scripts.tools.session_preflight."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.system_context import PolicyDecision, PolicyIssue
from scripts.tools import session_preflight


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _decision(*, blockers: list[PolicyIssue] | None = None, warnings: list[PolicyIssue] | None = None) -> PolicyDecision:
    blockers = blockers or []
    warnings = warnings or []
    return PolicyDecision(
        decision_id="decision-1",
        action="session_start_read_only",
        allowed=not blockers,
        blockers=blockers,
        warnings=warnings,
        infos=[],
        applicable_authorities=[],
        applicable_controls=[],
    )


class TestExtractHandoffSnapshot:
    def test_extracts_tool_date_summary(self, tmp_path: Path) -> None:
        handoff = tmp_path / "HANDOFF.md"
        handoff.write_text(
            "\n".join(
                [
                    "# HANDOFF",
                    "- **Tool:** Claude Code",
                    "- **Date:** 2026-03-17",
                    "- **Summary:** Did work",
                ]
            ),
            encoding="utf-8",
        )

        snap = session_preflight.extract_handoff_snapshot(handoff)

        assert snap.tool == "Claude Code"
        assert snap.date == "2026-03-17"
        assert snap.summary == "Did work"

    def test_missing_handoff_returns_empty_snapshot(self, tmp_path: Path) -> None:
        snap = session_preflight.extract_handoff_snapshot(tmp_path / "HANDOFF.md")
        assert snap.tool is None
        assert snap.date is None
        assert snap.summary is None


class TestBuildWarnings:
    def test_warns_when_handoff_missing(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(warnings=[PolicyIssue(level="warning", code="handoff_missing", message="HANDOFF.md missing.")]),
            ),
        ):
            warnings = session_preflight.build_warnings(tmp_path, context="generic")
        assert "HANDOFF.md missing." in warnings

    def test_warns_when_dirty(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(
                    warnings=[
                        PolicyIssue(
                            level="warning",
                            code="dirty_worktree",
                            message="Working tree is dirty. Re-read changed files before editing.",
                        )
                    ]
                ),
            ),
        ):
            warnings = session_preflight.build_warnings(tmp_path, context="generic")
        assert "Working tree is dirty. Re-read changed files before editing." in warnings

    def test_warns_when_wsl_context_missing_venv(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(
                    warnings=[
                        PolicyIssue(
                            level="warning",
                            code="expected_interpreter_missing",
                            message="The repo-managed interpreter for this context is missing.",
                            detail="/repo/.venv-wsl/bin/python",
                        )
                    ]
                ),
            ),
        ):
            warnings = session_preflight.build_warnings(tmp_path, context="codex-wsl")
        assert any("interpreter" in warning.lower() and "missing" in warning.lower() for warning in warnings)

    def test_warns_on_same_branch_parallel_read_only_context(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(
                    warnings=[
                        PolicyIssue(
                            level="warning",
                            code="parallel_session_present",
                            message="Parallel session present on this branch.",
                        )
                    ]
                ),
            ),
        ):
            warnings = session_preflight.build_warnings(
                tmp_path,
                context="generic",
                active_tool="codex-search",
                active_mode="read-only",
            )
        assert any("Parallel session present" in warning for warning in warnings)

    def test_warns_when_wsl_uses_wrong_interpreter(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(
                    warnings=[
                        PolicyIssue(
                            level="warning",
                            code="wrong_interpreter",
                            message="This context is using the wrong interpreter for the repo-managed environment.",
                            detail="current=/usr/bin/python3 expected=/repo/.venv-wsl/bin/python",
                        )
                    ]
                ),
            ),
        ):
            warnings = session_preflight.build_warnings(tmp_path, context="codex-wsl")

        assert any("wrong interpreter" in warning for warning in warnings)

    def test_no_wrong_interpreter_warning_when_wsl_uses_repo_python(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(session_preflight, "evaluate_system_policy", return_value=_decision()),
        ):
            warnings = session_preflight.build_warnings(tmp_path, context="codex-wsl")

        assert not any("wrong interpreter" in warning for warning in warnings)


class TestSessionClaims:
    def test_write_and_read_claim_roundtrip(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)

        written = session_preflight.write_claim(claim_path, tool="codex", branch="main", head="abc123")
        loaded = session_preflight.read_claim(claim_path)

        assert loaded is not None
        assert loaded.tool == written.tool
        assert loaded.branch == "main"
        assert loaded.head_sha == "abc123"

    def test_verify_claim_passes_when_head_matches(self, tmp_path: Path) -> None:
        with patch.object(session_preflight, "system_verify_claim", return_value=(True, [])):
            ok, warnings = session_preflight.verify_claim(tmp_path, active_tool="codex", claim_path=tmp_path / "claim.json")
        assert ok is True
        assert warnings == []

    def test_verify_claim_fails_when_head_changes(self, tmp_path: Path) -> None:
        with patch.object(session_preflight, "system_verify_claim", return_value=(False, ["HEAD mismatch"])):
            ok, warnings = session_preflight.verify_claim(tmp_path, active_tool="codex", claim_path=tmp_path / "claim.json")
        assert ok is False
        assert any("HEAD mismatch" in warning for warning in warnings)

    def test_verify_claim_fails_when_tool_changes(self, tmp_path: Path) -> None:
        with patch.object(session_preflight, "system_verify_claim", return_value=(False, ["tool mismatch"])):
            ok, warnings = session_preflight.verify_claim(tmp_path, active_tool="codex", claim_path=tmp_path / "claim.json")
        assert ok is False
        assert any("tool mismatch" in warning for warning in warnings)

    def test_build_blockers_for_same_branch_mutating_other_tool(self, tmp_path: Path) -> None:
        with (
            patch.object(session_preflight, "build_system_context"),
            patch.object(
                session_preflight,
                "evaluate_system_policy",
                return_value=_decision(
                    blockers=[
                        PolicyIssue(
                            level="blocker",
                            code="parallel_mutating_claim",
                            message="Concurrent mutating session blocked: another tool already holds a fresh mutating claim on this branch.",
                        )
                    ]
                ),
            ),
        ):
            blockers = session_preflight.build_blockers(
                tmp_path,
                active_tool="codex",
                active_mode="mutating",
            )
        assert any("Concurrent mutating session blocked" in blocker for blocker in blockers)


class TestPrintReport:
    def test_returns_zero_when_clean(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "- **Tool:** Codex\n- **Date:** 2026-03-17\n- **Summary:** Clean\n")
        _mkfile(tmp_path / ".venv" / "Scripts" / "python.exe", "")
        _mkfile(tmp_path / ".venv-wsl" / "bin" / "python", "")

        with (
            patch.object(session_preflight, "recent_commits", return_value=["abc123 test"]),
            patch.object(session_preflight, "branch_name", return_value="main"),
            patch.object(session_preflight, "head_sha", return_value="abc123"),
            patch.object(session_preflight, "build_blockers", return_value=[]),
            patch.object(session_preflight, "build_warnings", return_value=[]),
        ):
            exit_code = session_preflight.print_report(tmp_path, context="generic")

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "SESSION PREFLIGHT" in out
        assert "clean" in out

    def test_quiet_mode_blocks_mutating_same_branch_conflict(self, tmp_path: Path) -> None:
        with (
            patch.object(
                session_preflight,
                "_evaluate_preflight_policy",
                return_value=(
                    [
                        "Concurrent mutating session blocked: another tool already holds a fresh mutating claim on this branch."
                    ],
                    [],
                ),
            ),
        ):
            exit_code = session_preflight.print_report(
                tmp_path,
                context="generic",
                claim_tool="codex",
                claim_mode="mutating",
                quiet=True,
            )
        assert exit_code == 2

    def test_quiet_mode_blocks_wrong_interpreter_for_mutating_session(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch.object(
                session_preflight,
                "_evaluate_preflight_policy",
                return_value=(
                    [
                        "Mutating session blocked: this context is using the wrong interpreter for the repo-managed environment."
                    ],
                    [],
                ),
            ),
        ):
            exit_code = session_preflight.print_report(
                tmp_path,
                context="codex-wsl",
                claim_tool="codex",
                claim_mode="mutating",
                quiet=True,
            )
        out = capsys.readouterr().out
        assert exit_code == 2
        assert "wrong interpreter" in out.lower()


class TestCliBootstrap:
    def test_script_help_runs_via_direct_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [sys.executable, "scripts/tools/session_preflight.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Session preflight for canompx3" in result.stdout
