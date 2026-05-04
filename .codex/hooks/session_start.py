#!/usr/bin/env python3
"""Small session-start hints for repo-local Codex sessions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_payload() -> dict[str, object]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_context(payload: dict[str, object]) -> list[str]:
    hints: list[str] = []
    cwd = str(payload.get("cwd") or os.getcwd())

    task_route = PROJECT_ROOT / ".session" / "task-route.md"
    if task_route.exists():
        hints.append("Read `.session/task-route.md` before loading broader repo docs.")

    if cwd.startswith("/mnt/"):
        hints.append("This is a fallback `/mnt/...` checkout. Prefer a WSL-home clone for mutating Codex work.")

    if os.name != "nt" and not (PROJECT_ROOT / ".venv-wsl" / "bin" / "python").exists():
        hints.append(
            "Repo WSL env is missing. Run `python3 scripts/infra/codex_local_env.py setup --platform wsl` before mutating."
        )

    return hints


def main() -> int:
    payload = _load_payload()
    hints = _build_context(payload)
    if not hints:
        return 0

    print(
        json.dumps(
            {
                "continue": True,
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n".join(f"- {hint}" for hint in hints),
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
