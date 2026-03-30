#!/usr/bin/env python3
"""Post-compaction context re-injection.

After context compaction, critical project state may be lost.
This hook re-injects the most important context that Claude needs
to continue working correctly.

Fires on: PostCompact (informational, no decision control).
Output to stdout is added to Claude's context.
"""

import json
import sys
from pathlib import Path


def main():
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    lines = ["POST-COMPACTION CONTEXT RE-INJECTION:"]

    # 1. Active stage state
    stage_file = Path("docs/runtime/STAGE_STATE.md")
    if stage_file.exists():
        content = stage_file.read_text(encoding="utf-8")
        # Extract key fields
        for field in ("mode", "task", "stage", "stage_of"):
            for line in content.splitlines():
                if line.strip().startswith(f"{field}:"):
                    lines.append(f"  STAGE {field}: {line.strip()}")
                    break

    # 2. Uncommitted changes
    import subprocess
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            files = result.stdout.strip().split("\n")
            lines.append(f"  UNCOMMITTED: {len(files)} files — {', '.join(files[:5])}")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # 3. Critical reminders (most common post-compaction mistakes)
    lines.extend([
        "  REMINDERS:",
        "  - Canonical sources: asset_configs, dst, config, cost_model, paths — NEVER hardcode",
        "  - daily_features JOIN must include AND o.orb_minutes = d.orb_minutes",
        "  - One-way dep: pipeline/ → trading_app/ (never reversed)",
        "  - Data first: query before reading code for data questions",
        "  - 2026 holdout is SACRED — no discovery on 2026 data",
    ])

    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
