#!/usr/bin/env python3
"""Session start hook: injects orientation context on fresh sessions.

Fires on: SessionStart (matcher: startup, resume, compact, clear).
Output to stderr is shown to Claude as context.
"""

import json
import subprocess
import sys
from pathlib import Path


def main():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    session_type = event.get("session_type", "startup")
    lines = []

    if session_type == "startup":
        lines.append("NEW SESSION — Auto-orientation:")

        # Check for active stage
        stage_file = Path("docs/runtime/STAGE_STATE.md")
        if stage_file.exists():
            content = stage_file.read_text(encoding="utf-8")
            for field in ("mode", "task"):
                for line in content.splitlines():
                    if line.strip().startswith(f"{field}:"):
                        lines.append(f"  Active stage: {line.strip()}")
                        break
        else:
            lines.append("  No active stage.")

        # Check for uncommitted changes
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                files = result.stdout.strip().split("\n")
                lines.append(f"  Uncommitted: {len(files)} files")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Last 3 commits for context
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                lines.append("  Recent commits:")
                for commit in result.stdout.strip().split("\n"):
                    lines.append(f"    {commit}")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    elif session_type == "resume":
        lines.append("RESUMED SESSION — Check HANDOFF.md for last known state.")

    elif session_type == "compact":
        # PostCompact hook handles this separately
        pass

    elif session_type == "clear":
        lines.append("CONTEXT CLEARED — Re-read STAGE_STATE.md if active work exists.")

    if lines:
        print("\n".join(lines), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
