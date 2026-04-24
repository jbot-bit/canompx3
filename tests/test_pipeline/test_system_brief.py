from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from context.registry import TaskCandidate
from pipeline.system_brief import build_system_brief
from pipeline.system_context import PolicyDecision


def _mkfile(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _decision() -> PolicyDecision:
    return PolicyDecision(decision_id="1", action="orientation", allowed=True)


def test_build_system_brief_returns_orientation_surface(tmp_path: Path) -> None:
    for rel in (
        "AGENTS.md",
        "HANDOFF.md",
        "CLAUDE.md",
        "CODEX.md",
        "docs/governance/document_authority.md",
        "docs/governance/system_authority_map.md",
        "pipeline/system_authority.py",
        "pipeline/system_context.py",
        "pipeline/work_capsule.py",
        "pipeline/system_brief.py",
        "context/registry.py",
        "context/institutional.py",
    ):
        _mkfile(tmp_path / rel)
    _mkfile(tmp_path / "docs/runtime/decision-ledger.md", "- `decision-a` — keep startup bounded\n")
    _mkfile(tmp_path / "docs/runtime/debt-ledger.md", "- `debt-a` — still open\n")

    with (
        patch("pipeline.system_brief.build_system_context") as mock_snapshot,
        patch("pipeline.system_brief.evaluate_system_policy", return_value=_decision()),
        patch("pipeline.system_brief._load_capsule_summary", return_value=(None, [])),
    ):
        mock_snapshot.return_value.generated_at = "2026-04-13T00:00:00+00:00"
        mock_snapshot.return_value.work_queue.exists = True
        mock_snapshot.return_value.work_queue.open_count = 2
        mock_snapshot.return_value.work_queue.close_first_open_count = 1
        mock_snapshot.return_value.work_queue.stale_count = 0
        mock_snapshot.return_value.work_queue.top_items = []
        mock_snapshot.return_value.work_queue.handoff_matches_rendered = True
        mock_snapshot.return_value.authority.active_work_truth = "docs/runtime/action-queue.yaml"
        mock_snapshot.return_value.authority.local_ownership_truth = ".session/work_queue_leases.json"
        mock_snapshot.return_value.authority.baton_surface = "HANDOFF.md"
        payload = build_system_brief(tmp_path)

    assert payload["task_id"] == "system_orientation"
    assert payload["route_id"] == "system_orientation"
    assert payload["verification_profile"] == "orientation"
    assert payload["decision_refs"]
    assert payload["debt_refs"]
    assert payload["blocking_issues"] == []
    assert payload["work_queue"]["active_work_truth"] == "docs/runtime/action-queue.yaml"
    assert payload["work_queue"]["baton_context"] == "HANDOFF.md"


def test_build_system_brief_fails_closed_on_ambiguous_route(tmp_path: Path) -> None:
    with (
        patch("pipeline.system_brief.build_system_context") as mock_snapshot,
        patch("pipeline.system_brief.evaluate_system_policy", return_value=_decision()),
        patch("pipeline.system_brief._load_capsule_summary", return_value=(None, [])),
        patch(
            "pipeline.system_brief.resolve_from_text",
            return_value=(
                None,
                (
                    TaskCandidate(task_id="alpha", score=10, matched_terms=("same phrase",)),
                    TaskCandidate(task_id="beta", score=10, matched_terms=("same phrase",)),
                ),
            ),
        ),
    ):
        mock_snapshot.return_value.generated_at = "2026-04-13T00:00:00+00:00"
        mock_snapshot.return_value.work_queue.exists = False
        mock_snapshot.return_value.work_queue.open_count = 0
        mock_snapshot.return_value.work_queue.close_first_open_count = 0
        mock_snapshot.return_value.work_queue.stale_count = 0
        mock_snapshot.return_value.work_queue.top_items = []
        mock_snapshot.return_value.work_queue.handoff_matches_rendered = None
        mock_snapshot.return_value.authority.active_work_truth = "docs/runtime/action-queue.yaml"
        mock_snapshot.return_value.authority.local_ownership_truth = ".session/work_queue_leases.json"
        mock_snapshot.return_value.authority.baton_surface = "HANDOFF.md"
        payload = build_system_brief(tmp_path, task_text="same phrase")

    assert payload["task_id"] == "system_orientation"
    assert any(issue["code"] in {"ambiguous_route", "missing_route"} for issue in payload["blocking_issues"])
