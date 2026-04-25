from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.tools import session_router


def test_returns_requested_root_when_no_conflict(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session_router, "branch_name", lambda root: "main")
    monkeypatch.setattr(session_router, "in_linked_worktree", lambda root: False)
    monkeypatch.setattr(session_router, "conflicting_mutating_claims", lambda root, branch: [])

    decision = session_router.route_session(tmp_path, tool="codex", mode="mutating", task="Close queue")

    assert decision.routed is False
    assert decision.resolved_root == str(tmp_path.resolve())
    assert decision.reason == "no_conflict"


def test_auto_routes_main_checkout_when_conflict_exists(monkeypatch, tmp_path: Path) -> None:
    routed = tmp_path / ".worktrees" / "tasks" / "codex" / "close-queue"
    monkeypatch.setattr(session_router, "branch_name", lambda root: "main")
    monkeypatch.setattr(session_router, "in_linked_worktree", lambda root: False)
    monkeypatch.setattr(
        session_router,
        "conflicting_mutating_claims",
        lambda root, branch: [SimpleNamespace(tool="codex", branch="main", mode="mutating")],
    )
    monkeypatch.setattr(
        session_router.worktree_manager,
        "create_worktree",
        lambda tool, name, purpose=None: routed,
    )

    decision = session_router.route_session(tmp_path, tool="codex", mode="mutating", task="Close queue")

    assert decision.routed is True
    assert decision.resolved_root == str(routed.resolve())
    assert decision.workstream_name == "close-queue"
    assert decision.reason == "parallel_mutating_claim_detected"


def test_does_not_nest_when_already_in_linked_worktree(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(session_router, "branch_name", lambda root: "wt-codex-task")
    monkeypatch.setattr(session_router, "in_linked_worktree", lambda root: True)

    decision = session_router.route_session(tmp_path, tool="codex", mode="mutating", task="Something")

    assert decision.routed is False
    assert decision.reason == "already_isolated_in_worktree"


def test_conflicting_mutating_claims_filters_to_same_branch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        session_router,
        "_same_repo_claims",
        lambda root: [
            SimpleNamespace(tool="codex", branch="main", mode="mutating", root=str(tmp_path)),
            SimpleNamespace(tool="claude", branch="main", mode="read-only", root=str(tmp_path)),
            SimpleNamespace(tool="codex", branch="wt-codex-other", mode="mutating", root=str(tmp_path)),
        ],
    )

    claims = session_router.conflicting_mutating_claims(tmp_path, "main")

    assert len(claims) == 1
    assert claims[0].tool == "codex"


def test_derive_workstream_name_falls_back_to_timestamp(monkeypatch) -> None:
    class FakeNow:
        def strftime(self, fmt: str) -> str:
            return "20260423-120000"

    class FakeDateTime:
        @staticmethod
        def now(tz):
            return FakeNow()

    monkeypatch.setattr(session_router, "datetime", FakeDateTime)
    assert session_router.derive_workstream_name(None) == "parallel-20260423-120000"
