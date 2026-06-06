#!/usr/bin/env python3
"""PreToolUse guard for the headless autopilot runner.

Blocks Tier-B Edit/Write/Bash actions when an autopilot run is active, and
records a `BLOCKED_TIER_B` line to the run journal so the operator can read,
on return, exactly what the unattended run refused to do.

ACTIVATION: only when env `AUTOPILOT_RUN=1` is set. In every normal interactive
session this hook is a no-op (exit 0 immediately). The runner sets AUTOPILOT_RUN
plus AUTOPILOT_RUN_ID before invoking `claude -p`.

Classification is delegated to `scripts/autopilot/tier_guard.classify_action`
(the single source of truth, fail-CLOSED). This hook is fail-OPEN on its own
errors: any failure to read/parse input -> exit 0, never blocks a session it
cannot understand. The fail-closed safety lives inside tier_guard.

Block channel: PreToolUse supports `permissionDecision: "deny"` via
`hookSpecificOutput`. We emit that AND exit 2 with a stderr reason (defense in
depth across CLI versions).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = PROJECT_ROOT / "docs" / "runtime" / "autopilot"

# Make tier_guard importable regardless of cwd.
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))


def _journal_path(run_id: str) -> Path:
    safe = "".join(c for c in run_id if c.isalnum() or c in ("-", "_")) or "unknown"
    return AUTOPILOT_DIR / f"{safe}.jsonl"


def _append_journal(run_id: str, entry: dict) -> None:
    try:
        path = _journal_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # journalling must never break the block decision


def main() -> None:
    # Inert outside an autopilot run.
    if os.environ.get("AUTOPILOT_RUN") != "1":
        sys.exit(0)

    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)  # fail-open on unreadable input

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input", {}) or {}

    try:
        from tier_guard import classify_action  # type: ignore
    except ImportError:
        # If the classifier is unavailable we cannot make a safe decision.
        # Fail-open here (the runner's --allowedTools + branch guard still apply).
        sys.exit(0)

    tier, reason = classify_action(tool, tool_input)
    if tier != "B":
        sys.exit(0)

    run_id = os.environ.get("AUTOPILOT_RUN_ID", "unknown")
    target = tool_input.get("file_path") or tool_input.get("command") or ""
    _append_journal(
        run_id,
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "BLOCKED_TIER_B",
            "tool": tool,
            "target": target,
            "reason": reason,
        },
    )

    deny_reason = (
        f"BLOCKED_TIER_B: {tool} on '{target}' is a Tier-B action ({reason}). "
        "Autopilot does not perform capital/schema/live actions unattended. "
        "Logged to the run journal; continue with Tier-A work and report this "
        "blocker in the final report."
    )
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": deny_reason,
        }
    }
    print(json.dumps(payload))
    print(deny_reason, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # main()'s deliberate sys.exit(0/2) — propagate the intended code.
        # (SystemExit is a BaseException; a blanket catch below would rewrite
        # an intentional block-exit-2 to 0, killing the exit-code defense path.)
        raise
    except BaseException:  # pragma: no cover - fail-open on unexpected errors
        sys.exit(0)
