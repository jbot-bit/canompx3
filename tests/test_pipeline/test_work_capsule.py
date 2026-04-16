from __future__ import annotations

import json
from pathlib import Path

from pipeline import work_capsule


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_ensure_work_capsule_scaffold_creates_capsule_and_stage(tmp_path: Path) -> None:
    capsule_path, stage_path = work_capsule.ensure_work_capsule_scaffold(
        tmp_path,
        tool="codex",
        name="system-brain",
        branch="wt-codex-system-brain",
        purpose="Build the startup read-model.",
    )

    assert capsule_path.exists()
    assert stage_path.exists()

    capsule = work_capsule.read_work_capsule(capsule_path)
    assert capsule.task_id == "system_orientation"
    assert capsule.route_id == "system_orientation"
    assert capsule.briefing_level == "mutating"


def test_evaluate_current_capsule_reads_managed_worktree_metadata(tmp_path: Path) -> None:
    meta = {"name": "system-brain", "capsule_path": "docs/runtime/capsules/system-brain.md"}
    _mkfile(tmp_path / work_capsule.WORKTREE_META, json.dumps(meta))
    capsule_path, stage_path = work_capsule.ensure_work_capsule_scaffold(
        tmp_path,
        tool="codex",
        name="system-brain",
        branch="wt-codex-system-brain",
        purpose="Build the startup read-model.",
    )
    stage_path.write_text(
        work_capsule.render_stage_markdown(
            name="system-brain",
            tool="codex",
            capsule_path=capsule_path.relative_to(tmp_path),
        ),
        encoding="utf-8",
    )

    summary, issues = work_capsule.evaluate_current_capsule(tmp_path)

    assert summary is not None
    assert summary["path"] == "docs/runtime/capsules/system-brain.md"
    assert any(issue.code == "capsule_missing_scope" for issue in issues)
