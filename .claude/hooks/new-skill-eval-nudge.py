#!/usr/bin/env python3
"""PostToolUse hook: when a new SKILL.md is written, nudge to create eval + run skill-improve.

Catches the failure mode from 2026-05-16: assistant wrote /recall SKILL.md without
testing it. /skill-improve exists and is the right loop, but only fires when invoked.
This hook reminds at the moment a new SKILL.md is born.

Triggers ONLY when:
- Tool was Write (not Edit — edits to existing SKILL.md don't count)
- Path matches .claude/skills/<name>/SKILL.md
- That skill does NOT already have eval/eval.json

Fail-open: any error → exit 0 silently.
"""

import json
import sys
from pathlib import Path


def _read_input() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def main() -> int:
    payload = _read_input()
    tool = payload.get("tool_name", "")
    if tool != "Write":
        return 0

    tool_input = payload.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return 0

    # Normalize and check pattern
    p = Path(file_path)
    parts = p.as_posix().split("/")
    if "skills" not in parts:
        return 0
    if p.name != "SKILL.md":
        return 0

    try:
        skills_idx = parts.index("skills")
        skill_name = parts[skills_idx + 1]
    except (ValueError, IndexError):
        return 0

    skill_dir = p.parent
    eval_path = skill_dir / "eval" / "eval.json"
    if eval_path.exists():
        return 0  # already has eval — skill-improve can already run

    msg = (
        f"[new-skill] {skill_name}/SKILL.md written without eval/eval.json. "
        f"Next steps: (1) write {eval_path.as_posix()} with ≥3 assertion-based tests, "
        f"(2) run `/skill-improve {skill_name}` to verify the skill actually triggers + passes assertions. "
        "Untested skills are the 'phantom-stage' class — they look done but aren't."
    )
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
