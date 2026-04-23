#!/usr/bin/env python3
"""Archive the current root HANDOFF and rewrite it as a compact baton."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEADER = """# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.
"""


@dataclass
class HandoffContext:
    tool: str | None = None
    date: str | None = None
    summary: str | None = None
    next_steps: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def _strip_markup(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def parse_existing_handoff(text: str) -> HandoffContext:
    lines = text.splitlines()

    legacy = _parse_legacy_handoff(lines)
    if legacy.tool or legacy.date or legacy.summary:
        return legacy

    rolling = _parse_rolling_handoff(lines)
    if rolling.tool or rolling.date or rolling.summary:
        return rolling

    return HandoffContext()


def _parse_legacy_handoff(lines: list[str]) -> HandoffContext:
    context = HandoffContext()
    section: str | None = None

    for line in lines:
        if line.startswith("## Last Session"):
            section = "metadata"
            continue
        if re.match(r"^## Next Steps", line):
            section = "next_steps"
            continue
        if line.startswith("## Blockers") or line.startswith("## Blockers / Warnings"):
            section = "blockers"
            continue
        if line.startswith("## "):
            section = None
            continue

        stripped = line.strip()
        if section == "metadata":
            if line.startswith("- **Tool:** "):
                context.tool = line.removeprefix("- **Tool:** ").strip()
            elif line.startswith("- **Date:** "):
                context.date = line.removeprefix("- **Date:** ").strip()
            elif line.startswith("- **Summary:** "):
                context.summary = line.removeprefix("- **Summary:** ").strip()
        elif section == "next_steps" and stripped and not stripped.startswith("Phases "):
            context.next_steps.append(_strip_markup(re.sub(r"^(?:[-*]|\d+\.)\s+", "", stripped)))
        elif section == "blockers" and stripped.startswith("- "):
            context.blockers.append(_strip_markup(stripped[2:]))

    return context


def _parse_rolling_handoff(lines: list[str]) -> HandoffContext:
    context = HandoffContext(tool="Update log")
    date_match = None
    headline: str | None = None
    title: str | None = None
    next_steps: list[str] = []
    pending_step_heading: str | None = None

    for idx, line in enumerate(lines):
        if line.startswith("## Update (") and date_match is None:
            date_match = re.search(r"\((\d{4}-\d{2}-\d{2})", line)
            title = _strip_markup(line[3:])
            for follow in lines[idx + 1 :]:
                stripped = follow.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                headline = _strip_markup(stripped)
                break
            continue

        if re.match(r"^###\s+Next", line):
            pending_step_heading = None
            continue

        if line.startswith("### ") and "next" in line.lower():
            pending_step_heading = _strip_markup(line.removeprefix("### "))
            continue

        stripped = line.strip()
        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
        if bullet:
            entry = _strip_markup(bullet.group(1))
            if pending_step_heading:
                next_steps.append(f"{pending_step_heading} — {entry}")
                pending_step_heading = None
            else:
                next_steps.append(entry)
            continue

        if pending_step_heading and stripped and not stripped.startswith("#"):
            next_steps.append(f"{pending_step_heading} — {_strip_markup(stripped)}")
            pending_step_heading = None

    if date_match:
        context.date = date_match.group(1)
    if headline:
        context.summary = headline
    elif title:
        context.summary = title
    context.next_steps = next_steps

    return context


def render_compact_handoff(
    *,
    tool: str,
    date: str,
    summary: str,
    next_steps: list[str],
    blockers: list[str],
    references: list[str],
    archive_relpath: str,
) -> str:
    lines = [
        HEADER.rstrip(),
        "",
        "## Last Session",
        f"- **Tool:** {tool}",
        f"- **Date:** {date}",
        f"- **Summary:** {summary}",
    ]

    if next_steps:
        lines.extend(["", "## Next Steps — Active"])
        for idx, step in enumerate(next_steps, start=1):
            lines.append(f"{idx}. {step}")

    if blockers:
        lines.extend(["", "## Blockers / Warnings"])
        for blocker in blockers:
            lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Durable References",
            "- `docs/runtime/decision-ledger.md`",
            "- `docs/runtime/debt-ledger.md`",
        ]
    )
    for ref in references:
        lines.append(f"- `{ref}`")
    lines.append(f"- `{archive_relpath}`")

    lines.append("")
    return "\n".join(lines)


def unique_archive_path(archive_dir: Path, stem: str) -> Path:
    candidate = archive_dir / f"{stem}.md"
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = archive_dir / f"{stem}-{suffix}.md"
        if not candidate.exists():
            return candidate
        suffix += 1


def compact_handoff(
    *,
    handoff_path: Path,
    archive_dir: Path,
    tool: str | None,
    date: str | None,
    summary: str | None,
    next_steps: list[str],
    blockers: list[str],
    references: list[str],
) -> tuple[Path, str]:
    original = handoff_path.read_text(encoding="utf-8", errors="replace") if handoff_path.exists() else ""
    existing = parse_existing_handoff(original)

    resolved_date = date or existing.date or datetime.now().strftime("%Y-%m-%d")
    resolved_tool = tool or existing.tool or "Codex"
    resolved_summary = summary or existing.summary or "No session summary recorded."
    resolved_steps = next_steps or existing.next_steps[:5]
    resolved_blockers = blockers or existing.blockers[:5]

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = unique_archive_path(archive_dir, f"{resolved_date}-root-handoff-archive")
    archive_path.write_text(original, encoding="utf-8")

    archive_relpath = archive_path.relative_to(handoff_path.parent).as_posix()
    compact = render_compact_handoff(
        tool=resolved_tool,
        date=resolved_date,
        summary=resolved_summary,
        next_steps=resolved_steps,
        blockers=resolved_blockers,
        references=references,
        archive_relpath=archive_relpath,
    )
    handoff_path.write_text(compact, encoding="utf-8")
    return archive_path, compact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repo root containing HANDOFF.md")
    parser.add_argument("--tool", default=None, help="Tool label for the compact baton")
    parser.add_argument("--date", default=None, help="ISO date for the compact baton")
    parser.add_argument("--summary", default=None, help="One-line last-session summary")
    parser.add_argument("--next-step", action="append", default=[], help="Active next step. Repeat for multiple items.")
    parser.add_argument(
        "--warning", action="append", default=[], help="Active blocker/warning. Repeat for multiple items."
    )
    parser.add_argument(
        "--reference", action="append", default=[], help="Durable reference path. Repeat for multiple items."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    handoff_path = root / "HANDOFF.md"
    archive_dir = root / "docs" / "handoffs" / "archived"
    archive_path, _ = compact_handoff(
        handoff_path=handoff_path,
        archive_dir=archive_dir,
        tool=args.tool,
        date=args.date,
        summary=args.summary,
        next_steps=args.next_step,
        blockers=args.warning,
        references=args.reference,
    )
    print(f"Archived previous HANDOFF to {archive_path.relative_to(root).as_posix()}")
    print(f"Rewrote compact baton at {handoff_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
