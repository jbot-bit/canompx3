from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline import system_context
from pipeline.system_context import (
    AuthorityContext,
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
            snapshot = build_system_context(
                tmp_path, context_name="generic", active_tool="codex", active_mode="mutating"
            )

        assert snapshot.handoff.exists is True
        assert snapshot.git.branch == "main"
        assert len(snapshot.active_stages) == 1
        assert snapshot.active_stages[0].scope_lock == ["pipeline/system_context.py"]
        assert len(snapshot.claims) == 1
        assert snapshot.authority.backbone_modules[-1] == "pipeline/system_context.py"

    def test_closed_stage_files_do_not_count_as_active(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "- **Tool:** Codex\n- **Date:** 2026-04-12\n- **Summary:** Stage test\n")
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "executed-stage.md",
            "\n".join(
                [
                    "---",
                    "task: Executed stage",
                    "mode: RESEARCH",
                    "---",
                    "# Stage",
                    "",
                    "## Execution Outcome",
                    "",
                    "- Executed and closed.",
                ]
            ),
        )
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "closed-stage.md",
            "\n".join(
                [
                    "---",
                    "task: Closed stage",
                    "mode: IMPLEMENTATION",
                    "status: closed -> KILL",
                    "---",
                    "# Stage",
                ]
            ),
        )
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "mode-closed-stage.md",
            "\n".join(
                [
                    "---",
                    "task: Mode closed stage",
                    "mode: CLOSED",
                    "---",
                    "# Stage",
                ]
            ),
        )
        _mkfile(
            tmp_path / "docs" / "runtime" / "stages" / "active-stage.md",
            "\n".join(
                [
                    "---",
                    "task: Active stage",
                    "mode: IMPLEMENTATION",
                    "scope_lock:",
                    "  - pipeline/system_context.py",
                    "---",
                    "# Stage",
                ]
            ),
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

        assert len(snapshot.active_stages) == 1
        assert Path(snapshot.active_stages[0].path).name == "active-stage.md"

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
            snapshot = build_system_context(
                current_root, context_name="generic", active_tool="codex", active_mode="mutating"
            )

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
            snapshot = build_system_context(
                tmp_path, context_name="codex-wsl", active_tool="codex", active_mode="mutating"
            )
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

    def test_mutating_session_ignores_same_checkout_claim_when_mount_path_casing_differs(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")
        claim = SessionClaim(
            tool="codex",
            branch="main",
            head_sha="abc123",
            started_at="2026-04-12T00:00:00+00:00",
            pid=1,
            mode="mutating",
            root="/mnt/c/Users/joshd/canompx3",
            fresh=True,
        )

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "git_hooks_path", return_value=(".githooks", True)),
            patch.object(system_context, "list_claims", return_value=[claim]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
            patch.object(
                system_context,
                "_paths_same_location",
                side_effect=lambda left, right: Path(str(left)).name == "canompx3" and Path(str(right)) == tmp_path,
            ),
            patch.object(system_context.sys, "executable", str((tmp_path / ".venv-wsl" / "bin" / "python").resolve())),
            patch.object(system_context.sys, "prefix", str((tmp_path / ".venv-wsl").resolve())),
        ):
            _mkfile(tmp_path / ".venv-wsl" / "bin" / "python", "")
            snapshot = build_system_context(
                tmp_path, context_name="codex-wsl", active_tool="codex", active_mode="mutating"
            )
            decision = evaluate_system_policy(snapshot, "session_start_mutating")

        assert decision.allowed is True
        assert not any(issue.code == "parallel_mutating_claim" for issue in decision.blockers)

    def test_mutating_session_blocks_when_precommit_hook_inactive(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "git_hooks_path", return_value=(".git/hooks", False)),
            patch.object(system_context, "list_claims", return_value=[]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic")
            decision = evaluate_system_policy(snapshot, "session_start_mutating")

        assert decision.allowed is False
        assert any(issue.code == "precommit_hook_inactive" for issue in decision.blockers)

    def test_orientation_warns_when_precommit_hook_inactive(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")

        with (
            patch.object(system_context, "_canonical_repo_root", return_value=(tmp_path, tmp_path / ".git")),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(system_context, "git_status_details", return_value=([], True, None)),
            patch.object(system_context, "git_hooks_path", return_value=(None, False)),
            patch.object(system_context, "list_claims", return_value=[]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic")
            decision = evaluate_system_policy(snapshot, "orientation")

        assert decision.allowed is True
        assert any(issue.code == "precommit_hook_inactive" for issue in decision.warnings)

    def test_orientation_warns_when_handoff_drifted_from_queue(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# stale\n")
        _mkfile(
            tmp_path / "docs" / "runtime" / "action-queue.yaml",
            "\n".join(
                [
                    "schema_version: 1",
                    "updated_at: 2026-04-24T00:00:00+00:00",
                    "items:",
                    "  - id: first",
                    "    title: First thing",
                    "    class: research",
                    "    status: ready",
                    "    priority: P1",
                    "    close_before_new_work: true",
                    "    owner_hint: codex",
                    "    last_verified_at: 2026-04-24",
                    "    freshness_sla_days: 2",
                    "    next_action: Do first",
                    "    exit_criteria: Finish first",
                    "    blocked_by: []",
                    "    decision_refs: []",
                    "    evidence_refs: []",
                    "    notes_ref: docs/runtime/stages/first.md",
                    "    override_note:",
                ]
            ),
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

        assert any(issue.code == "handoff_queue_mismatch" for issue in decision.warnings)
        assert any(issue.code == "close_first_carryover" for issue in decision.warnings)

        # 2026-05-17 footgun guard: the handoff_queue_mismatch warning must NOT
        # recommend `--write` — that flag is destructive (overwrites HANDOFF.md
        # and deletes session prose). The pulse hint should point operators at
        # the safe stdout-render-then-hand-patch path instead.
        mismatch_warning = next(issue for issue in decision.warnings if issue.code == "handoff_queue_mismatch")
        assert mismatch_warning.detail is not None, "handoff_queue_mismatch must carry a detail hint"
        assert "--write" not in mismatch_warning.detail, (
            "handoff_queue_mismatch detail must not recommend `render-handoff --write` "
            "— that command silently destroys HANDOFF.md session prose"
        )

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

    def test_read_only_session_warns_but_not_blocks_on_mutating_peer(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "HANDOFF.md", "# HANDOFF\n")
        peer = SessionClaim(
            tool="other-tool",
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
            patch.object(system_context, "list_claims", return_value=[peer]),
            patch.object(system_context, "_build_authority_context", return_value=_authority_stub()),
        ):
            snapshot = build_system_context(tmp_path, context_name="generic", active_tool="codex")
            decision = evaluate_system_policy(snapshot, "session_start_read_only")

        assert decision.allowed is True
        assert not decision.blockers
        assert any(issue.code == "parallel_session_present" for issue in decision.warnings)


class TestVerifyClaim:
    def test_write_claim_uses_session_owner_pid_when_present(self, tmp_path: Path, monkeypatch) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("CANOMPX3_SESSION_OWNER", "pid:4242")

        claim = write_claim(claim_path, tool="codex", branch="main", head="abc123", mode="mutating", root=str(tmp_path))

        assert claim.pid == 4242

    def test_read_claim_marks_dead_same_runtime_owner_stale(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(system_context, "_current_runtime_tag", return_value="wsl"):
            write_claim(claim_path, tool="codex", branch="main", head="abc123", mode="mutating", root=str(tmp_path))

        with (
            patch.object(system_context, "_current_runtime_tag", return_value="wsl"),
            patch.object(system_context, "_pid_is_live", return_value=False),
        ):
            claim = system_context.read_claim(claim_path)

        assert claim is not None
        assert claim.runtime == "wsl"
        assert claim.fresh is False

    def test_read_claim_marks_legacy_dead_wsl_owner_stale(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text(
            json.dumps(
                {
                    "tool": "codex",
                    "branch": "main",
                    "head_sha": "abc123",
                    "started_at": "2099-04-12T00:00:00+00:00",
                    "pid": 815,
                    "mode": "mutating",
                    "root": "/mnt/c/Users/joshd/canompx3",
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(system_context, "_current_runtime_tag", return_value="wsl"),
            patch.object(system_context, "_pid_is_live", return_value=False),
        ):
            claim = system_context.read_claim(claim_path)

        assert claim is not None
        assert claim.runtime == ""
        assert claim.fresh is False

    def test_read_claim_keeps_cross_runtime_owner_fresh(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text(
            json.dumps(
                {
                    "tool": "claude",
                    "branch": "main",
                    "head_sha": "abc123",
                    "started_at": "2099-04-12T00:00:00+00:00",
                    "pid": 77,
                    "mode": "mutating",
                    "root": r"C:\repo",
                    "runtime": "windows",
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(system_context, "_current_runtime_tag", return_value="wsl"),
            patch.object(system_context, "_pid_is_live", return_value=False),
        ):
            claim = system_context.read_claim(claim_path)

        assert claim is not None
        assert claim.fresh is True

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

    def test_verify_claim_fails_when_owner_process_is_dead(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(system_context, "_current_runtime_tag", return_value="wsl"):
            write_claim(claim_path, tool="codex", branch="main", head="abc123", mode="mutating", root=str(tmp_path))

        with (
            patch.object(system_context, "_current_runtime_tag", return_value="wsl"),
            patch.object(system_context, "_pid_is_live", return_value=False),
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
        ):
            ok, warnings = verify_claim(tmp_path, active_tool="codex", claim_path=claim_path)

        assert ok is False
        assert any("owner process is no longer running" in warning for warning in warnings)

    def test_verify_claim_allows_same_checkout_when_mount_path_casing_differs(self, tmp_path: Path) -> None:
        claim_path = tmp_path / ".git" / "claim.json"
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        write_claim(
            claim_path,
            tool="codex",
            branch="main",
            head="abc123",
            mode="mutating",
            root="/mnt/c/Users/joshd/canompx3",
        )

        with (
            patch.object(system_context, "branch_name", return_value="main"),
            patch.object(system_context, "head_sha", return_value="abc123"),
            patch.object(
                system_context,
                "_paths_same_location",
                side_effect=lambda left, right: (
                    Path(str(left)).name == "canompx3" and Path(str(right)).name == "canompx3"
                ),
            ),
        ):
            ok, warnings = verify_claim(Path("/mnt/c/users/joshd/canompx3"), active_tool="codex", claim_path=claim_path)

        assert ok is True
        assert not any("Root mismatch" in warning for warning in warnings)


class TestReadClaimExceptionHandling:
    def test_read_claim_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        claim_path = tmp_path / "corrupt.json"
        claim_path.write_text("{invalid json", encoding="utf-8")
        result = system_context.read_claim(claim_path)
        assert result is None

    def test_read_claim_returns_none_on_unexpected_exception_and_logs_warning(self, tmp_path: Path, caplog) -> None:
        import json as _json

        claim_path = tmp_path / "claim.json"
        claim_path.write_text(
            _json.dumps(
                {
                    "tool": "codex",
                    "branch": "main",
                    "head_sha": "abc123",
                    "started_at": "2099-04-12T00:00:00+00:00",
                    "pid": 1,
                    "mode": "mutating",
                    "root": str(tmp_path),
                    "runtime": "windows",
                }
            ),
            encoding="utf-8",
        )
        import logging

        with (
            patch.object(system_context, "_claim_is_fresh", side_effect=RuntimeError("unexpected")),
            caplog.at_level(logging.WARNING, logger="pipeline.system_context"),
        ):
            result = system_context.read_claim(claim_path)

        assert result is None
        assert any("unexpected error" in record.message for record in caplog.records)


class TestCliBootstrap:
    # Full investigation history + four falsified hypotheses live at
    # docs/runtime/stages/fix-system-context-bootstrap-help-fork.md
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="CI hang on Windows runner; see fix-system-context-bootstrap-help-fork.md",
    )
    def test_system_context_script_help_runs_via_direct_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]

        result = subprocess.run(
            [sys.executable, "scripts/tools/system_context.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        assert result.returncode == 0
        assert "Show canonical project system context" in result.stdout
