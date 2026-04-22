#!/usr/bin/env python3
"""Render a thin hiROI discovery capsule from the MNQ frontier and board outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.build_mnq_discovery_frontier import build_frontier


def _format_signed(value: float) -> str:
    if value == float("-inf"):
        return "n/a"
    return f"{value:+.3f}"


def _format_candidate(row: dict) -> str:
    return (
        f"{row['lane']} | {row['role']} {row['thesis']} | "
        f"evidence {row['evidence_score']:.1f} | "
        f"IS Δ {_format_signed(float(row['delta_is']))} | "
        f"OOS Δ {_format_signed(float(row['delta_oos']))} | "
        f"N_on_OOS {int(row['n_on_oos'])} | "
        f"status {row['status']}"
    )


def _render_section(title: str, items: list[str]) -> list[str]:
    lines = [f"## {title}"]
    if not items:
        lines.append("- none")
        return lines
    lines.extend(f"- {item}" for item in items)
    return lines


def build_capsule(root: Path, state_path: Path, top_n: int = 5) -> str:
    frontier = build_frontier(
        root=root,
        ledger_path=root / ".session" / "mnq_discovery_frontier_ledger.json",
    )
    candidates = frontier["candidates"]
    review_batch = frontier.get("review_batch", [])
    family_top = [row for row in candidates if row["candidate_kind"] == "family"][:top_n]
    unsolved_transfers = [row for row in candidates if row["candidate_kind"] == "transfer"][:top_n]
    layered_top = [row for row in candidates if row["candidate_kind"] == "cell"][:top_n]

    state_note = "No prior loop state recorded."
    if state_path.exists():
        try:
            history = json.loads(state_path.read_text(encoding="utf-8")).get("history", [])
        except json.JSONDecodeError:
            history = []
        candidate_events = [row for row in history if row.get("candidate_id")]
        if candidate_events:
            last = candidate_events[-1]
        elif history:
            last = history[-1]
        else:
            last = None
        if last is not None:
            state_note = (
                f"Last loop status: {last.get('status', 'unknown')} | "
                f"summary: {last.get('summary', '').strip()} | "
                f"next focus: {last.get('next_focus', '').strip()}"
            )

    lines = [
        "# MNQ hiROI Discovery Capsule",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Queue Rule",
        "- Treat this capsule and `.session/mnq_discovery_frontier.json` as the primary hiROI queue surface.",
        "- `HANDOFF.md` is a cross-tool baton, not the discovery frontier. Only update it after a real queue decision from this iteration.",
        "- Already-promoted and already-closed candidates should be filtered out by durable repo truth before ranking the live queue.",
        "- Prefer fresh bounded non-geometry shortlist work when the frontier's family lane stays alive.",
        "",
        "## Last Loop State",
        f"- {state_note}",
        "",
        "## Frontier Summary",
        f"- candidates: {frontier['candidate_count']}",
        f"- queued: {frontier['queued_count']}",
        f"- families: {frontier['kind_counts']['family']}",
        f"- transfers: {frontier['kind_counts']['transfer']}",
        f"- cells: {frontier['kind_counts']['cell']}",
        "",
    ]
    lines.extend(_render_section("Review Batch", [_format_candidate(row) for row in review_batch]))
    lines.append("")
    lines.extend(_render_section("Top Family Candidates", [_format_candidate(row) for row in family_top]))
    lines.append("")
    lines.extend(_render_section("Top Transfer Candidates", [_format_candidate(row) for row in unsolved_transfers]))
    lines.append("")
    lines.extend(_render_section("Top Cell Candidates", [_format_candidate(row) for row in layered_top]))
    lines.extend(
        [
            "",
            "## Honest Next-Move Guidance",
            "- Start with the diversified review batch, not the raw global rank, so one candidate class does not monopolize the queue.",
            "- Default to the highest-ranked honest item inside the review batch unless the evidence gives a clear reason to skip it.",
            "- If the best live move is another exact transfer, run the cheap gate first and keep it to one bridge.",
            "- Do not spend an iteration on baton-only prose cleanup when the hiROI frontier can be advanced or explicitly parked from frontier evidence.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Worktree root")
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Optional state JSON path; defaults to .session/mnq_autonomous_discovery_state.json under root",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Rows per section")
    args = parser.parse_args()

    root = args.root.resolve()
    state_path = (
        args.state.resolve()
        if args.state is not None
        else root / ".session" / "mnq_autonomous_discovery_state.json"
    )
    print(build_capsule(root=root, state_path=state_path, top_n=args.top_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
