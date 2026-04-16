"""Tests for scripts.tools.claude_superpower_brief."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.tools.claude_superpower_brief import build_brief
from scripts.tools.project_pulse import PulseItem, PulseReport


def _mkfile(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_report() -> PulseReport:
    return PulseReport(
        generated_at="2026-04-03T12:00:00Z",
        cache_hit=True,
        git_head="abc123",
        git_branch="main",
        items=[
            PulseItem(category="broken", severity="high", source="tests", summary="1 failing test"),
            PulseItem(category="decaying", severity="medium", source="staleness", summary="daily features stale"),
            PulseItem(category="paused", severity="low", source="git", summary="3 uncommitted files"),
        ],
        handoff_tool="Codex",
        handoff_date="2026-04-03",
        handoff_summary="Dashboard fix in progress",
        handoff_next_steps=["Verify the Claude hook integration"],
        upcoming_sessions=[
            {"label": "LONDON_METALS", "brisbane_time": "17:00", "hours_away": 5.7},
            {"label": "CME_REOPEN", "brisbane_time": "08:00", "hours_away": 20.0},
        ],
        recommendation="Finish the current integration before taking new work.",
    )


class TestClaudeSuperpowerBrief:
    def test_build_brief_renders_high_signal_sections(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "docs" / "runtime" / "STAGE_STATE.md",
            "mode: execute\ntask: superpower claude\n",
        )
        _mkfile(
            tmp_path / "MEMORY.md",
            "# Memory\n\n## Trading\nnotes\n\n## Tooling\nnotes\n",
        )
        today = date.today()
        yesterday = today - timedelta(days=1)
        _mkfile(tmp_path / "memory" / f"{today.isoformat()}.md", "today")
        _mkfile(tmp_path / "memory" / f"{yesterday.isoformat()}.md", "yesterday")

        with patch("scripts.tools.claude_superpower_brief.build_pulse", return_value=_sample_report()):
            brief = build_brief(root=tmp_path, mode="session-start")

        assert "SUPERPOWER BRIEF:" in brief
        assert "Stage [legacy]: superpower claude — execute" in brief
        assert "Last: Codex (2026-04-03) — Dashboard fix in progress" in brief
        assert "Next: Finish the current integration before taking new work." in brief
        assert "Active step: Verify the Claude hook integration" in brief
        assert "Broken: 1 failing test" in brief
        assert "Decaying: daily features stale" in brief
        assert "Paused: 3 uncommitted files" in brief
        assert "Upcoming: LONDON_METALS 17:00 (+5.7h) | CME_REOPEN 08:00 (+20.0h)" in brief
        assert "Memory topics: Trading | Tooling" in brief
        assert f"Recent notes: {today.isoformat()}.md | {yesterday.isoformat()}.md" in brief

    def test_post_compact_mode_adds_compact_rule(self, tmp_path: Path) -> None:
        with patch("scripts.tools.claude_superpower_brief.build_pulse", return_value=_sample_report()):
            brief = build_brief(root=tmp_path, mode="post-compact")

        assert "Compact rule: re-check live files before trusting prior context." in brief

    def test_stale_notes_are_marked_stale_not_recent(self, tmp_path: Path) -> None:
        _mkfile(tmp_path / "memory" / "2026-01-01.md", "old")

        with patch("scripts.tools.claude_superpower_brief.build_pulse", return_value=_sample_report()):
            brief = build_brief(root=tmp_path, mode="interactive")

        assert "Notes stale: latest 2026-01-01.md" in brief
        assert "Recent notes:" not in brief

    def test_build_brief_falls_back_when_pulse_raises(self, tmp_path: Path) -> None:
        _mkfile(
            tmp_path / "docs" / "runtime" / "STAGE_STATE.md",
            "mode: implementation\ntask: harden hooks\n",
        )
        _mkfile(
            tmp_path / "HANDOFF.md",
            "\n".join(
                [
                    "## Last Session",
                    "- **Tool:** Codex",
                    "- **Date:** 2026-04-03",
                    "- **Summary:** Added superpower brief",
                ]
            ),
        )
        _mkfile(tmp_path / "MEMORY.md", "## Tooling\n")

        with patch("scripts.tools.claude_superpower_brief.build_pulse", side_effect=RuntimeError("boom")):
            brief = build_brief(root=tmp_path, mode="post-compact")

        assert "Brief degraded: RuntimeError" in brief
        assert "Stage [legacy]: harden hooks — implementation" in brief
        assert "Last: Codex (2026-04-03) — Added superpower brief" in brief
        assert "Memory topics: Tooling" in brief
        assert "Compact rule: re-check live files before trusting prior context." in brief
