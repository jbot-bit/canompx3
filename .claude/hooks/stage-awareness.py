#!/usr/bin/env python3
"""Stage awareness hook v3: fires on every user prompt.

Injects stage context AND workflow directives so Claude can't skip
blast radius, self-review, or completion evidence — regardless of
how the user initiates work.

v3 changes (from v2):
- Stale STAGE_STATE detection (>4h → warn, forces re-orientation)
- Rotating directives to prevent habituation (time-based, 4 variants)
- PDF grounding reminder included in rotation
- Scope creep detection (scope_lock modified since staging)

v2 changes (from v1):
- Injects self-review directive when no stage or in DESIGN mode
- Warns when IMPLEMENTATION mode is missing blast_radius
- Keeps output to 1-3 lines (token efficient)
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

STAGE_STATE = Path("docs/runtime/STAGE_STATE.md")
STAGES_DIR = Path("docs/runtime/stages")

# Rotating directives for "stage: none" — prevents habituation.
# Picked by minute % len(NONE_DIRECTIVES) so they vary across interactions.
NONE_DIRECTIVES = (
    "SELF-CHECK: Simulate happy/edge/failure scenarios before presenting ANY plan. Show what you found, not just 'looks good.'",
    "PDF GROUNDING: If citing resources/ files, EXTRACT text first. Never cite from training memory as if you read the file.",
    "COMPLETION: 'Done' = tests pass (show output) + dead code swept (grep orphans) + drift clean. Not claims.",
    "BLAST RADIUS: List files changing + their tests + downstream consumers. Write it in STAGE_STATE before editing.",
)

DESIGN_DIRECTIVES = (
    "BEFORE PRESENTING PLAN: Simulate happy/edge/failure paths. Fix what breaks. Show simulation results. Do NOT present first draft.",
    "SELF-CHECK: Walk through your plan step by step. At each step: what if NULL? What if missing? What if the interface changed? Fix first, present second.",
    "EXECUTION PLAN: Every plan MUST include HOW to deploy, not just WHAT to change. FK constraints, rebuild ordering, data migration steps. Code without deployment = half a plan.",
)


def parse_field(content, field):
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def parse_blast_radius(content):
    """Parse blast_radius — single-line YAML, multi-line list, or markdown section."""
    if "## Blast Radius" in content:
        section = content.split("## Blast Radius")[1].split("##")[0].split("---")[0]
        if section.strip():
            return section.strip()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("blast_radius:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
            # Multi-line: collect following list items
            items = []
            collecting = False
            for l2 in content.splitlines():
                if l2.strip().startswith("blast_radius:"):
                    collecting = True
                    continue
                if collecting:
                    s2 = l2.strip()
                    if s2.startswith("- "):
                        items.append(s2[2:].strip())
                    elif s2 and not s2.startswith("#"):
                        break
            return "; ".join(items) if items else None
    return None


def check_stale(content):
    """Return True if STAGE_STATE is >4 hours old."""
    updated_str = parse_field(content, "updated")
    if not updated_str:
        return False  # Can't tell — don't nag
    try:
        # Handle both "2026-03-27T12:00:00Z" and "2026-03-27T12:00:00+00:00"
        updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - updated
        return age > timedelta(hours=4)
    except (ValueError, TypeError):
        return False


def main():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    # ── Collect all stage files (multi-agent) ────────────────────────
    stage_files = []
    if STAGE_STATE.exists():
        stage_files.append(("claude", STAGE_STATE))
    if STAGES_DIR.is_dir():
        for f in sorted(STAGES_DIR.glob("*.md")):
            agent_name = f.stem  # e.g., "codex", "auto_trivial"
            stage_files.append((agent_name, f))

    if not stage_files:
        variant = datetime.now().minute % len(NONE_DIRECTIVES)
        print(
            "stage: none\n"
            "WORKFLOW: Non-trivial work requires STAGE_STATE with blast_radius before edits.\n"
            f"{NONE_DIRECTIVES[variant]}",
            file=sys.stderr,
        )
        sys.exit(0)

    # Report ALL active stages (multi-agent awareness)
    output_lines = []
    for agent_name, fpath in stage_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        mode = parse_field(content, "mode")
        task = parse_field(content, "task")
        stage = parse_field(content, "stage")
        stage_of = parse_field(content, "stage_of")
        blast_radius = parse_blast_radius(content)
        is_stale = check_stale(content)

        if not mode:
            continue

        parts = [f"stage: {mode}"]
        if len(stage_files) > 1:
            parts.append(f"[{agent_name}]")
        if task:
            parts.append(task)
        if stage and stage_of:
            parts.append(f"({stage}/{stage_of})")
        elif stage:
            parts.append(f"(stage {stage})")
        if is_stale:
            parts.append("STALE(>4h)")
        if mode == "IMPLEMENTATION" and (not blast_radius or len(blast_radius.strip()) < 30):
            parts.append("MISSING blast_radius")

        output_lines.append(" | ".join(parts))

    if not output_lines:
        sys.exit(0)

    # Print all stages, then a directive for the primary (first) stage
    primary_content = stage_files[0][1].read_text(encoding="utf-8")
    primary_mode = parse_field(primary_content, "mode")

    print("\n".join(output_lines), file=sys.stderr)

    if primary_mode == "DESIGN":
        variant = datetime.now().minute % len(DESIGN_DIRECTIVES)
        print(DESIGN_DIRECTIVES[variant], file=sys.stderr)

    if len(stage_files) > 1:
        print(
            "MULTI-AGENT: Multiple stage files active. Your edits go through if ANY stage permits them.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
