from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from pipeline import system_context
from pipeline.system_context import (
    ActiveStage,
    AuthorityContext,
    PolicyIssue,
    SessionClaim,
    build_system_context,
    evaluate_system_policy,
    verify_claim,
    write_claim,
)


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _authority_stub() -> AuthorityContext:
    return AuthorityContext(
        authority_map_doc="docs/governance/system_authority_map.md",
        doctrine_docs=["CLAUDE.md", "TRADING_RULES.md"],
        backbone_modules=["pipeline/system_authority.py", "pipeline/system_context.py"],
        active_orb_instruments=["MGC", "MNQ"],
        active_profiles=["topstep_50k_mnq_auto"],
        published_relations={"active": "active_validated_setups", "deployable": "deployable_validated_setups"},
    )


class TestBuildSystemContext:
    def test_collects_stage_files_and_claims(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "- **Tool:** Codex\n- **Date:** 2026-04-12\n- **Summary:** Stage test\n")
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "test-stage.md",
            "\n".join(
                [
                    "---",
                    "task: Test stage",
                    "mode: IMPLEMENTATION",
                    "agent: codex",
                    "updated: 2026-04-12T00:00:00Z",
                    "scope_lock:",
                    "  - pipeline/system_context.py",
                    "---",
                ]
            ),
        )

        claim = SessionClaim(
            tool="codex",
            branch="main",
            head_sha="abc123",
            started_at="2026-04-12T00:00:00+00:00",
            pid=1,
            mode="mutating",
            root=str(tmp_path),
            fresh=True,
        )

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "list_claims", return_value=[claim]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic", active_tool="codex", active_mode="mutating")

        assert snapshot.handoff.exists is True
        assert snapshot.git.branch == "main"
        assert len(snapshot.active_stages) == 1
        assert snapshot.active_stages[0].scope_lock == ["pipeline/system_context.py"]
        assert len(snapshot.claims) == 1
        assert snapshot.authority.backbone_modules[-1] == "pipeline/system_context.py"

    def test_filters_claims_from_unrelated_repos(self, tmp_path: Path) -> None:
        current_root = tmp_path / "repo"
        unrelated_root = tmp_path / "other"
        current_root.mkdir(parents=True)
        unrelated_root.mkdir(parents=True)
        _mkfile(current_root / "HANDOFF.md", "# HANDOFF\n")

        relevant = SessionClaim(
            tool="codex",
            branch="main",
            head_sha="abc123",
            started_at="2026-04-12T00:00:00+00:00",
            pid=1,
            mode="mutating",
            root=str(current_root),
            fresh=True,
        )
        unrelated = SessionClaim(
            tool="codex",
            branch="main",
            head_sha="zzz999",
            started_at="2026-04-12T00:00:00+00:00",
            pid=2,
            mode="mutating",
            root=str(unrelated_root),
            fresh=True,
        )

        def _canonical_side_effect(path: Path) -> tuple[Path, Path | None]:
            resolved = path.resolve()
            if resolved == current_root.resolve():
                return current_root, current_root / ".git"
            if resolved == unrelated_root.resolve():
                return unrelated_root, unrelated_root / ".git"
            raise AssertionError(f"unexpected path: {resolved}")

        with (
            patch.object(system_context, "_canonical_repo_root", side_effect=_canonical_side_effect),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "list_claims", return_value=[relevant, unrelated]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(current_root, context_name="generic", active_tool="codex", active_mode="mutating")

        assert [claim.root for claim in snapshot.claims] == [str(current_root)]


class TestInferContextName:
    def test_detects_codex_wsl_from_prefix_even_when_interpreter_is_symlinked(self, tmp_path: Path) -> None:
        current_python = Path("/usr/bin/python3")
        # Use a resolved path so the .resolve() inside infer_context_name is a no-op,
        # avoiding Windows drive-letter mismatch with the unresolved mock value.
        current_prefix = (tmp_path / ".venv-wsl").resolve()

        with patch.object(system_context, "_expected_prefix", side_effect=[current_prefix, None]):
            context_name = system_context.infer_context_name(
                tmp_path,
                current_python=current_python,
                current_prefix=current_prefix,
            )

        assert context_name == "codex-wsl"


class TestEvaluateSystemPolicy:
    def test_mutating_session_blocks_wrong_interpreter(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")
        _mkfile(tmp_path / ".venv-wsl" / "bin" / "python", "")

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "list_claims", return_value=[]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
            patch.object(system_context.sys, "executable", "/usr/bin/python3"),
            patch.object(system_context.sys, "prefix", "/usr"),
        ):
            snapshot = build_system_context(tmp_path, context_name="codex-wsl", active_tool="codex", active_mode="mutating")
            decision = evaluate_system_policy(snapshot, "session_start_mutating")

        assert decision.allowed is False
        assert any(issue.code == "wrong_interpreter" for issue in decision.blockers)

    def test_orientation_warns_when_active_stage_files_exist(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "scope.md",
            "---\ntask: Scope\nmode: IMPLEMENTATION\nscope_lock:\n  - scripts/tools/session_preflight.py\n---\n",
        )

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "list_claims", return_value=[]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic")
            decision = evaluate_system_policy(snapshot, "orientation")

        assert decision.allowed is True
        assert any(issue.code == "active_stage_files" for issue in decision.warnings)

    def test_orientation_distinguishes_status_unavailable_from_dirty(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(
                system_context,
                "git_status_details",
                return_value=([], False, "git status unavailable or timed out"),
            ),
            patch.object(system_context, "list_claims", return_value=[]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic")
            decision = evaluate_system_policy(snapshot, "orientation")

        assert snapshot.git.dirty is False
        assert any(issue.code == "git_status_unavailable" for issue in decision.warnings)


class TestVerifyClaim:
    def test_verify_claim_detects_head_mismatch(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        write_claim(claim_path, tool="codex", branch="main", head="abc123", mode="mutating", root=str(tmp_path))

        with (
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="def456"),
        ):
            ok, warnings = verify_claim(tmp_path, active_tool="codex", claim_path=claim_path)

        assert ok is False
        assert any("HEAD mismatch" in warning for warning in warnings)


class TestCliBootstrap:
    def test_system_context_script_help_runs_via_direct_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [sys.executable, "scripts/tools/system_context.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0
        assert "Show canonical project system context" in result.stdout
