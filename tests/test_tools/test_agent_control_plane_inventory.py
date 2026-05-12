from __future__ import annotations

from types import SimpleNamespace

from scripts.tools.agent_control_plane_inventory import (
    ControlPlaneWorkstream,
    _recommended_action,
    render_markdown,
)


def _item(*, dirty: bool = False, state: str | None = "active", branch: str = "wt-codex-example") -> SimpleNamespace:
    return SimpleNamespace(dirty=dirty, state=state, branch=branch)


def test_recommended_action_prioritizes_dirty_worktree() -> None:
    assert _recommended_action(_item(dirty=True, state="handoff")) == "inspect_dirty_worktree_before_assignment"


def test_recommended_action_handles_handoff() -> None:
    assert _recommended_action(_item(state="handoff", branch="main")) == "claim_or_close_handoff"


def test_recommended_action_handles_task_branch() -> None:
    assert _recommended_action(_item()) == "review_for_pr_or_close"


def test_render_markdown_includes_workstream_policy_fields() -> None:
    payload = {
        "repo": "/repo",
        "main_branch": "main",
        "origin_main_head": "abc123",
        "workstreams": [
            ControlPlaneWorkstream(
                tool="codex",
                name="paperclip-control-plane-eval",
                path="/repo/.worktrees/tasks/codex/paperclip-control-plane-eval",
                branch="wt-codex-paperclip-control-plane-eval",
                head="abc123",
                state="active",
                purpose="Evaluate control plane",
                dirty=False,
                last_opened_at=None,
                handoff_note=None,
                recommended_action="review_for_pr_or_close",
                write_policy="isolated_worktree_only",
                truth_policy="repo_canonical_layers_only",
            ).__dict__
        ],
    }

    rendered = render_markdown(payload)

    assert "paperclip-control-plane-eval" in rendered
    assert "review_for_pr_or_close" in rendered
