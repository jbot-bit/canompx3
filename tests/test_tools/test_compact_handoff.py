from __future__ import annotations

from pathlib import Path

from scripts.tools import compact_handoff


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_compact_handoff_archives_existing_and_renders_compact(tmp_path: Path) -> None:
    handoff = tmp_path / "HANDOFF.md"
    _mkfile(
        handoff,
        "\n".join(
            [
                "# HANDOFF.md — Cross-Tool Session Baton",
                "",
                "## Update (2026-04-22 autonomous discovery reround)",
                "",
                "Route stack refreshed and stale exact bridge parked.",
                "",
                "### Next move",
                "",
                "- Build a bounded non-geometry shortlist",
                "- Keep verification fail-closed",
            ]
        ),
    )

    archive_path, compact = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=tmp_path / "docs" / "handoffs" / "archived",
        tool="Codex",
        date="2026-04-22",
        summary="Slimmed HANDOFF into a compact baton and archived the full history snapshot.",
        next_steps=[
            "Resume MNQ autonomous discovery from a bounded non-geometry shortlist refresh.",
            "Keep route-level verification fail-closed when repo blockers are red.",
        ],
        blockers=["Criterion 11 control report is still missing for topstep_50k_mnq_auto."],
        references=["docs/plans/2026-04-22-handoff-baton-compaction.md"],
    )

    assert archive_path.exists()
    assert "autonomous discovery reround" in archive_path.read_text(encoding="utf-8")
    rendered = handoff.read_text(encoding="utf-8")
    assert rendered == compact
    assert "## Last Session" in rendered
    assert "- **Tool:** Codex" in rendered
    assert "## Next Steps — Active" in rendered
    assert "1. Resume MNQ autonomous discovery from a bounded non-geometry shortlist refresh." in rendered
    assert "## Blockers / Warnings" in rendered
    assert "docs/plans/2026-04-22-handoff-baton-compaction.md" in rendered
    assert "docs/handoffs/archived/2026-04-22-root-handoff-archive.md" in rendered


def test_compact_handoff_falls_back_to_existing_metadata(tmp_path: Path) -> None:
    handoff = tmp_path / "HANDOFF.md"
    _mkfile(
        handoff,
        "\n".join(
            [
                "## Last Session",
                "- **Tool:** Claude",
                "- **Date:** 2026-03-17",
                "- **Summary:** Built pulse",
                "",
                "## Next Steps — Active",
                "1. Phase 1: do thing",
                "2. Phase 2: do other thing",
                "",
                "## Blockers / Warnings",
                "- Pre-existing test failure: broken thing",
            ]
        ),
    )

    _, rendered = compact_handoff.compact_handoff(
        handoff_path=handoff,
        archive_dir=tmp_path / "docs" / "handoffs" / "archived",
        tool=None,
        date=None,
        summary=None,
        next_steps=[],
        blockers=[],
        references=[],
    )

    assert "- **Tool:** Claude" in rendered
    assert "- **Date:** 2026-03-17" in rendered
    assert "- **Summary:** Built pulse" in rendered
    assert "1. Phase 1: do thing" in rendered
    assert "2. Phase 2: do other thing" in rendered
    assert "- Pre-existing test failure: broken thing" in rendered
