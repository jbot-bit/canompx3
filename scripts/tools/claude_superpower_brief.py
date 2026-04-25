#!/usr/bin/env python3
"""High-signal workspace brief for Claude hooks and on-demand commands.

Clean-room implementation inspired by publicly discussed ideas around:
- concise "brief" output modes
- stronger context persistence across compaction
- memory pointers instead of dumping raw history

This script intentionally uses only repo-local truth sources and existing
project helpers. It does not depend on or reproduce any leaked code.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.project_pulse import PulseReport, build_pulse  # noqa: E402


def _extract_memory_topics(root: Path, limit: int = 4) -> list[str]:
    memory_md = root / "MEMORY.md"
    if not memory_md.exists():
        return []

    topics: list[str] = []
    for line in memory_md.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            topic = stripped.removeprefix("## ").strip()
            if topic:
                topics.append(topic)
        if len(topics) >= limit:
            break
    return topics


def _recent_memory_notes(root: Path, limit: int = 2) -> list[str]:
    memory_dir = root / "memory"
    if not memory_dir.exists():
        return []
    notes = sorted(memory_dir.glob("????-??-??.md"), reverse=True)
    return [p.name for p in notes[:limit]]


def _memory_note_line(root: Path, limit: int = 2, stale_after_days: int = 7) -> str | None:
    notes = _recent_memory_notes(root, limit=limit)
    if not notes:
        return None

    latest = notes[0]
    try:
        latest_day = date.fromisoformat(latest.removesuffix(".md"))
        age_days = (date.today() - latest_day).days
    except ValueError:
        return f"  Recent notes: {' | '.join(notes)}"

    if age_days <= stale_after_days:
        return f"  Recent notes: {' | '.join(notes)}"
    return f"  Notes stale: latest {latest} ({age_days}d old)"


def _extract_handoff_context(root: Path) -> tuple[str | None, str | None, str | None]:
    handoff = root / "HANDOFF.md"
    if not handoff.exists():
        return None, None, None

    tool: str | None = None
    when: str | None = None
    summary: str | None = None
    in_last_session = False

    for raw_line in handoff.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("## Last Session"):
            in_last_session = True
            continue
        if in_last_session and line.startswith("## "):
            break
        if not in_last_session:
            continue
        if line.startswith("- **Tool:** "):
            tool = line.removeprefix("- **Tool:** ").strip()
        elif line.startswith("- **Date:** "):
            when = line.removeprefix("- **Date:** ").strip()
        elif line.startswith("- **Summary:** "):
            summary = line.removeprefix("- **Summary:** ").strip()
    return tool, when, summary


def _fallback_lines(root: Path, *, mode: str, error: Exception) -> list[str]:
    lines = ["SUPERPOWER BRIEF:", f"  Brief degraded: {error.__class__.__name__}"]

    # Read all stage files (stages/*.md + legacy STAGE_STATE.md)
    stages_dir = root / "docs" / "runtime" / "stages"
    if stages_dir.is_dir():
        for sf in sorted(stages_dir.glob("*.md")):
            if sf.name == ".gitkeep":
                continue
            try:
                sc = sf.read_text(encoding="utf-8", errors="replace")
                sm = re.search(r"^mode:\s*(.+)$", sc, flags=re.MULTILINE)
                st = re.search(r"^task:\s*(.+)$", sc, flags=re.MULTILINE)
                if st or sm:
                    lines.append(f"  Stage [{sf.stem}]: {st.group(1) if st else '?'} — {sm.group(1) if sm else '?'}")
            except OSError:
                pass
    legacy_file = root / "docs" / "runtime" / "STAGE_STATE.md"
    if legacy_file.exists():
        content = legacy_file.read_text(encoding="utf-8", errors="replace")
        mode_match = re.search(r"^mode:\s*(.+)$", content, flags=re.MULTILINE)
        task_match = re.search(r"^task:\s*(.+)$", content, flags=re.MULTILINE)
        if task_match or mode_match:
            lines.append(
                f"  Stage [legacy]: {task_match.group(1) if task_match else '?'} — {mode_match.group(1) if mode_match else '?'}"
            )

    tool, when, summary = _extract_handoff_context(root)
    if summary:
        lines.append(f"  Last: {tool or '?'} ({when or '?'}) — {summary}")

    topics = _extract_memory_topics(root)
    if topics:
        lines.append(f"  Memory topics: {' | '.join(topics)}")

    note_line = _memory_note_line(root)
    if note_line:
        lines.append(note_line)

    if mode == "post-compact":
        lines.append("  Compact rule: re-check live files before trusting prior context.")

    return lines


def _top_items(report: PulseReport, category: str, limit: int = 2) -> list[str]:
    return [item.summary for item in report.items if item.category == category][:limit]


def _render_lines(report: PulseReport, *, mode: str, root: Path) -> list[str]:
    lines: list[str] = ["SUPERPOWER BRIEF:"]

    # Read all stage files (stages/*.md + legacy STAGE_STATE.md)
    stages_dir = root / "docs" / "runtime" / "stages"
    if stages_dir.is_dir():
        for sf in sorted(stages_dir.glob("*.md")):
            if sf.name == ".gitkeep":
                continue
            try:
                sc = sf.read_text(encoding="utf-8", errors="replace")
                sm = re.search(r"^mode:\s*(.+)$", sc, flags=re.MULTILINE)
                st = re.search(r"^task:\s*(.+)$", sc, flags=re.MULTILINE)
                if st or sm:
                    lines.append(f"  Stage [{sf.stem}]: {st.group(1) if st else '?'} — {sm.group(1) if sm else '?'}")
            except OSError:
                pass
    legacy_file = root / "docs" / "runtime" / "STAGE_STATE.md"
    if legacy_file.exists():
        content = legacy_file.read_text(encoding="utf-8", errors="replace")
        mode_match = re.search(r"^mode:\s*(.+)$", content, flags=re.MULTILINE)
        task_match = re.search(r"^task:\s*(.+)$", content, flags=re.MULTILINE)
        if task_match or mode_match:
            lines.append(
                f"  Stage [legacy]: {task_match.group(1) if task_match else '?'} — {mode_match.group(1) if mode_match else '?'}"
            )

    if report.handoff_summary:
        tool = report.handoff_tool or "?"
        when = report.handoff_date or "?"
        lines.append(f"  Last: {tool} ({when}) — {report.handoff_summary}")

    if report.recommendation:
        lines.append(f"  Next: {report.recommendation}")

    if report.handoff_next_steps:
        lines.append(f"  Active step: {report.handoff_next_steps[0]}")

    work_queue = report.system_identity.get("work_queue", {}) if isinstance(report.system_identity, dict) else {}
    if isinstance(work_queue, dict) and work_queue.get("open_count") is not None:
        queue_bits = [f"open={work_queue.get('open_count')}"]
        if work_queue.get("close_first_open_count"):
            queue_bits.append(f"close-first={work_queue.get('close_first_open_count')}")
        if work_queue.get("stale_count"):
            queue_bits.append(f"stale={work_queue.get('stale_count')}")
        top_ids = work_queue.get("top_items") or []
        if top_ids and isinstance(top_ids, list) and isinstance(top_ids[0], dict):
            first_id = top_ids[0].get("id")
            if first_id:
                queue_bits.append(f"top={first_id}")
        if queue_bits:
            lines.append(f"  Queue: {' | '.join(queue_bits)}")

    broken = _top_items(report, "broken")
    if broken:
        lines.append(f"  Broken: {' | '.join(broken)}")

    decaying = _top_items(report, "decaying")
    if decaying:
        lines.append(f"  Decaying: {' | '.join(decaying)}")

    paused = _top_items(report, "paused")
    if paused:
        lines.append(f"  Paused: {' | '.join(paused)}")

    if report.upcoming_sessions:
        top_sessions = []
        for session in report.upcoming_sessions[:2]:
            top_sessions.append(f"{session['label']} {session['brisbane_time']} (+{session['hours_away']}h)")
        lines.append(f"  Upcoming: {' | '.join(top_sessions)}")

    topics = _extract_memory_topics(root)
    if topics:
        lines.append(f"  Memory topics: {' | '.join(topics)}")

    note_line = _memory_note_line(root)
    if note_line:
        lines.append(note_line)

    if mode == "post-compact":
        lines.append("  Compact rule: re-check live files before trusting prior context.")

    return lines


def build_brief(*, root: Path, mode: str = "session-start") -> str:
    try:
        report = build_pulse(
            root=root,
            fast=True,
            skip_drift=True,
            skip_tests=True,
        )
    except Exception as exc:
        return "\n".join(_fallback_lines(root, mode=mode, error=exc))
    return "\n".join(_render_lines(report, mode=mode, root=root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a concise Claude workspace brief")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Project root")
    parser.add_argument(
        "--mode",
        choices=("session-start", "post-compact", "interactive"),
        default="interactive",
        help="Rendering mode",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    # Windows terminals default to cp1252 which can't handle → and other Unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(build_brief(root=root, mode=args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
