#!/usr/bin/env python3
"""Stage awareness hook: fires on every user prompt.

Low-token nudge — reads STAGE_STATE.md and outputs a one-line context
so Claude always knows the current stage without being asked.
"""

import json
import sys
from pathlib import Path

STAGE_STATE = Path("docs/runtime/STAGE_STATE.md")


def parse_field(content, field):
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def main():
    # Read user's message from stdin (hook receives it)
    try:
        json.load(sys.stdin)  # consume stdin (hook protocol)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    if not STAGE_STATE.exists():
        # No active stage — brief note
        print("stage: none", file=sys.stderr)
        sys.exit(0)

    content = STAGE_STATE.read_text(encoding="utf-8")
    mode = parse_field(content, "mode")
    task = parse_field(content, "task")
    stage = parse_field(content, "stage")
    stage_of = parse_field(content, "stage_of")

    if not mode:
        sys.exit(0)

    # One-line context — minimal tokens
    parts = [f"stage: {mode}"]
    if task:
        parts.append(task)
    if stage and stage_of:
        parts.append(f"({stage}/{stage_of})")
    elif stage:
        parts.append(f"(stage {stage})")

    print(" | ".join(parts), file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
