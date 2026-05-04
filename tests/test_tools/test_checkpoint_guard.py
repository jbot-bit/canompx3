from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.tools import checkpoint_guard


def test_classify_closeout_state_blocks_result_without_durable_surface() -> None:
    result_files, durable_files, blockers = checkpoint_guard.classify_closeout_state(
        [
            "docs/audit/results/2026-04-23-something.md",
            "research/some_runner.py",
        ]
    )
    assert result_files == ["docs/audit/results/2026-04-23-something.md"]
    assert durable_files == []
    assert blockers


def test_classify_closeout_state_allows_result_with_handoff() -> None:
    result_files, durable_files, blockers = checkpoint_guard.classify_closeout_state(
        [
            "docs/audit/results/2026-04-23-something.md",
            "HANDOFF.md",
        ]
    )
    assert result_files == ["docs/audit/results/2026-04-23-something.md"]
    assert durable_files == ["HANDOFF.md"]
    assert blockers == []


def test_classify_closeout_state_allows_result_with_action_queue() -> None:
    result_files, durable_files, blockers = checkpoint_guard.classify_closeout_state(
        [
            "docs/audit/results/2026-04-23-something.md",
            "docs/runtime/action-queue.yaml",
        ]
    )
    assert result_files == ["docs/audit/results/2026-04-23-something.md"]
    assert durable_files == ["docs/runtime/action-queue.yaml"]
    assert blockers == []


def test_mutating_claim_conflicts_blocks_multiple_same_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        checkpoint_guard,
        "_same_repo_claims",
        lambda: [
            SimpleNamespace(tool="claude", mode="mutating", branch="main"),
            SimpleNamespace(tool="codex", mode="mutating", branch="main"),
            SimpleNamespace(tool="codex-search", mode="read-only", branch="main"),
        ],
    )
    details, blockers = checkpoint_guard.mutating_claim_conflicts("main")
    assert details == ["claude@main", "codex@main"]
    assert blockers


def test_build_report_combines_closeout_and_claim_checks(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_guard, "_run_git", lambda *args: SimpleNamespace(returncode=0, stdout="main\n"))
    monkeypatch.setattr(
        checkpoint_guard,
        "mutating_claim_conflicts",
        lambda branch: (["claude@main", "codex@main"], ["parallel mutating conflict"]),
    )
    report = checkpoint_guard.build_report(["docs/audit/results/x.md"])
    assert report.branch == "main"
    assert "parallel mutating conflict" in report.blockers
    assert any("durable closeout surface" in item for item in report.blockers)


def test_same_repo_claims_filters_to_current_repo_anchor(monkeypatch, tmp_path: Path) -> None:
    other_root = tmp_path / "other"
    monkeypatch.setattr(checkpoint_guard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        checkpoint_guard,
        "list_claims",
        lambda fresh_only=True: [
            SimpleNamespace(root=str(tmp_path), tool="codex", branch="main", mode="mutating"),
            SimpleNamespace(root=str(other_root), tool="claude", branch="main", mode="mutating"),
        ],
    )
    monkeypatch.setattr(
        checkpoint_guard,
        "_repo_anchor",
        lambda root: Path("/repo-a") if Path(root) == tmp_path else Path("/repo-b"),
    )
    claims = checkpoint_guard._same_repo_claims()
    assert len(claims) == 1
    assert claims[0].tool == "codex"
