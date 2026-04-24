from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import work_queue


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_queue(root: Path) -> None:
    _mkfile(
        root / "docs" / "runtime" / "action-queue.yaml",
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
                "  - id: second",
                "    title: Second thing",
                "    class: runtime",
                "    status: ready",
                "    priority: P2",
                "    close_before_new_work: false",
                "    owner_hint: codex",
                "    last_verified_at: 2026-04-10",
                "    freshness_sla_days: 1",
                "    next_action: Do second",
                "    exit_criteria: Finish second",
                "    blocked_by: []",
                "    decision_refs: []",
                "    evidence_refs: []",
                "    notes_ref: docs/runtime/stages/second.md",
                "    override_note:",
            ]
        ),
    )


class TestRenderHandoff:
    def test_render_uses_queue_top_items(self, tmp_path: Path) -> None:
        _seed_queue(tmp_path)
        rendered = work_queue.render_handoff_text(tmp_path, tool="Codex", date="2026-04-24", summary="Queue test")

        assert "## Next Steps — Active" in rendered
        assert "1. First thing — Do first" in rendered
        assert "2. Second thing — Do second" in rendered
        assert "docs/runtime/action-queue.yaml" in rendered

    def test_snapshot_detects_handoff_mismatch(self, tmp_path: Path) -> None:
        _seed_queue(tmp_path)
        _mkfile(tmp_path / "HANDOFF.md", "# stale\n")

        snapshot = work_queue.queue_snapshot(tmp_path)

        assert snapshot.exists is True
        assert snapshot.handoff_matches_rendered is False
        assert snapshot.stale_count == 1


class TestClaiming:
    def test_claim_requires_override_when_close_first_remains(self, tmp_path: Path) -> None:
        _seed_queue(tmp_path)

        with pytest.raises(ValueError, match="override note"):
            work_queue.claim_item(
                tmp_path,
                item_id="second",
                session_id="codex:main:abc",
                tool="codex",
                branch="main",
                worktree=str(tmp_path),
            )

    def test_claim_records_override_and_lease(self, tmp_path: Path) -> None:
        _seed_queue(tmp_path)

        lease = work_queue.claim_item(
            tmp_path,
            item_id="second",
            session_id="codex:main:abc",
            tool="codex",
            branch="main",
            worktree=str(tmp_path),
            override_note="urgent runtime fix",
        )

        assert lease.claimed_item_ids == ["second"]
        queue = work_queue.load_queue(tmp_path)
        second = next(item for item in queue.items if item.id == "second")
        assert second.override_note == "urgent runtime fix"
