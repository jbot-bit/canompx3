from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts.infra import codex_publish


def _state(
    *,
    branch: str = "codex/topic",
    upstream: str | None = "origin/codex/topic",
    staged: list[str] | None = None,
    unstaged: list[str] | None = None,
    status: list[str] | None = None,
) -> codex_publish.GitState:
    return codex_publish.GitState(
        root=r"C:\repo",
        branch=branch,
        head="abc123",
        upstream=upstream,
        staged=staged or [],
        unstaged=unstaged or [],
        status=status or [],
    )


def test_detached_head_blocks_publish_state() -> None:
    blockers, _warnings = codex_publish.classify_state(_state(branch="HEAD"))

    assert any("Detached HEAD" in blocker for blocker in blockers)


def test_handoff_unstaged_churn_warns_but_does_not_block() -> None:
    blockers, warnings = codex_publish.classify_state(_state(status=[" M HANDOFF.md"]))

    assert blockers == []
    assert any("HANDOFF.md has local unstaged churn" in warning for warning in warnings)


def test_staged_handoff_warns_for_deliberate_confirmation() -> None:
    blockers, warnings = codex_publish.classify_state(_state(staged=["HANDOFF.md"]))

    assert blockers == []
    assert any("HANDOFF.md is staged" in warning for warning in warnings)


def test_mixed_staged_unstaged_codex_python_blocks() -> None:
    blockers, _warnings = codex_publish.classify_state(
        _state(
            staged=["scripts/infra/codex_publish.py"],
            unstaged=["scripts/infra/codex_publish.py"],
        )
    )

    assert any("unstaged changes" in blocker for blocker in blockers)


def test_session_policy_delegates_to_session_preflight(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish.session_preflight, "build_blockers", return_value=["parallel block"]) as blockers,
        patch.object(codex_publish.session_preflight, "build_warnings", return_value=["parallel warn"]) as warnings,
    ):
        policy = codex_publish.check_session_policy(tmp_path)

    assert policy.blockers == ["parallel block"]
    assert policy.warnings == ["parallel warn"]
    assert blockers.call_args.kwargs["active_mode"] == "mutating"
    assert blockers.call_args.kwargs["active_tool"] == "codex"
    assert warnings.call_args.kwargs["active_mode"] == "mutating"
    assert warnings.call_args.kwargs["active_tool"] == "codex"


def test_parallel_mutating_claim_blocks_push_plan(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish, "collect_git_state", return_value=_state()),
        patch.object(codex_publish, "check_session_policy", return_value=codex_publish.SessionPolicy(["parallel"], [])),
    ):
        plan = codex_publish.build_plan(tmp_path, "push")

    assert not plan.ok
    assert "parallel" in plan.blockers


def test_parallel_session_warning_surfaces_without_blocking_preflight(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish, "collect_git_state", return_value=_state()),
        patch.object(
            codex_publish,
            "check_session_policy",
            return_value=codex_publish.SessionPolicy([], ["Parallel session present on this branch."]),
        ),
    ):
        plan = codex_publish.build_plan(tmp_path, "preflight")

    assert plan.ok
    assert any("Parallel session present" in warning for warning in plan.warnings)
    assert ["git", "diff", "--check"] in plan.commands


def test_preflight_plans_scoped_codex_python_checks() -> None:
    state = _state(staged=["scripts/infra/codex_publish.py", "trading_app/live_config.py"])

    commands = codex_publish.preflight_commands(state)

    assert ["git", "diff", "--check"] in commands
    assert ["ruff", "check", "scripts/infra/codex_publish.py"] in commands
    assert not any("trading_app/live_config.py" in command for command in commands)


def test_select_labels_uses_only_available_labels(tmp_path: Path) -> None:
    with patch.object(codex_publish, "available_labels", return_value={"codex"}):
        present, missing = codex_publish.select_labels(tmp_path)

    assert present == ["codex"]
    assert missing == ["codex-automation"]


def test_pr_plan_updates_existing_pr_with_available_labels(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish, "collect_git_state", return_value=_state()),
        patch.object(codex_publish, "check_session_policy", return_value=codex_publish.SessionPolicy([], [])),
        patch.object(codex_publish, "select_labels", return_value=(["codex"], ["codex-automation"])),
        patch.object(codex_publish, "existing_pr_number", return_value="369"),
    ):
        plan = codex_publish.build_plan(tmp_path, "pr")

    assert plan.commands == [["gh", "pr", "edit", "369", "--add-label", "codex"]]
    assert plan.missing_labels == ["codex-automation"]


def test_pr_plan_creates_when_no_existing_pr(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish, "collect_git_state", return_value=_state(branch="codex/new")),
        patch.object(codex_publish, "check_session_policy", return_value=codex_publish.SessionPolicy([], [])),
        patch.object(codex_publish, "select_labels", return_value=(["codex"], [])),
        patch.object(codex_publish, "existing_pr_number", return_value=None),
    ):
        plan = codex_publish.build_plan(tmp_path, "pr")

    assert plan.commands == [
        ["gh", "pr", "create", "--base", "main", "--head", "codex/new", "--fill", "--label", "codex"]
    ]


def test_pr_plan_does_not_query_github_when_blocked(tmp_path: Path) -> None:
    with (
        patch.object(codex_publish, "collect_git_state", return_value=_state(branch="HEAD")),
        patch.object(codex_publish, "check_session_policy", return_value=codex_publish.SessionPolicy([], [])),
        patch.object(codex_publish, "select_labels") as labels,
        patch.object(codex_publish, "existing_pr_number") as existing,
    ):
        plan = codex_publish.build_plan(tmp_path, "pr")

    assert not plan.ok
    assert plan.commands == []
    labels.assert_not_called()
    existing.assert_not_called()


def test_evidence_path_lives_under_git_dir(tmp_path: Path) -> None:
    with patch.object(codex_publish, "_git_output", return_value=".git"):
        target = codex_publish.evidence_dir(tmp_path)

    assert target == tmp_path / ".git" / "canompx3" / "codex_publish"
