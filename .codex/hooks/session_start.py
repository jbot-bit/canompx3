#!/usr/bin/env python3
"""Small session-start hints for repo-local Codex sessions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_payload() -> dict[str, object]:
    raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
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

    try:
        from scripts.infra import codex_parity

        report = codex_parity.check_parity(PROJECT_ROOT)
    except Exception as exc:  # pragma: no cover - hook must fail open.
        hints.append(f"Codex parity check could not run: {exc}")
    else:
        if not report["ok"]:
            missing_refs = report.get("missing_refs", {})
            missing_files = report.get("missing_files", [])
            ref_count = sum(len(paths) for paths in missing_refs.values()) if isinstance(missing_refs, dict) else 0
            file_count = len(missing_files) if isinstance(missing_files, list) else 0
            hints.append(
                "Codex parity drift detected "
                f"({file_count} missing file(s), {ref_count} missing Claude reference(s)); "
                "run `python3 scripts/infra/codex_parity.py` before relying on Claude-equivalent routing."
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
