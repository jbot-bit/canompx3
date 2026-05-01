#!/usr/bin/env python3
"""Read-Budget Guard — Tier 3 of discovery-loop hardening.

Tracks a session-scoped read budget. When the agent reads many files
without producing an edit to production code (pipeline/ or trading_app/),
inject a UserPromptSubmit warning encouraging a checkpoint.

Modes (selected by argv[1]):

  increment      — PostToolUse(Read): reads += 1
  reset-on-edit  — PostToolUse(Edit|Write): if file is in pipeline/ or
                   trading_app/, reset reads to 0
  inject         — UserPromptSubmit: emit soft warning at >=16 reads,
                   hard warning at >=26 reads
  reset-session  — SessionStart / PostCompact: clear state

State file: .claude/hooks/state/read-budget.json
  {"reads": int, "edits_to_prod": int, "last_warned_at": ISO|null}

Fail-open on every error (read state, write state, parse stdin).

Refs: docs/plans/discovery-loop-hardening.md § Tier 3.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / "state" / "read-budget.json"

SOFT_CAP = 16
HARD_CAP = 26
SOFT_COOLDOWN_MINUTES = 5

GUARDED_PATH_PREFIXES = (
    "pipeline/",
    "trading_app/",
)


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _is_prod_path(file_path: str) -> bool:
    norm = _norm(file_path)
    if not norm:
        return False
    return any(norm.startswith(p) or f"/{p}" in f"/{norm}" for p in GUARDED_PATH_PREFIXES)


def _load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}
    state.setdefault("reads", 0)
    state.setdefault("edits_to_prod", 0)
    state.setdefault("last_warned_at", None)
    return state


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass  # fail-open


def _read_event() -> dict:
    try:
        return json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        return {}


def _cmd_increment() -> None:
    state = _load_state()
    state["reads"] = int(state.get("reads", 0)) + 1
    _save_state(state)


def _cmd_reset_on_edit() -> None:
    event = _read_event()
    file_path = event.get("tool_input", {}).get("file_path", "")
    if not _is_prod_path(file_path):
        return  # docs/tests don't reset the budget
    state = _load_state()
    state["reads"] = 0
    state["edits_to_prod"] = int(state.get("edits_to_prod", 0)) + 1
    state["last_warned_at"] = None
    _save_state(state)


def _cmd_reset_session() -> None:
    _save_state(
        {"reads": 0, "edits_to_prod": 0, "last_warned_at": None}
    )


SOFT_MSG_TEMPLATE = (
    "READ-BUDGET GUARD (soft): {reads} reads with no edit to pipeline/ or "
    "trading_app/. Discovery may be drifting — is the read set converging "
    "on a falsifiable change? If yes, write the diff. If no, narrow the "
    "scope with `python scripts/tools/context_resolver.py --task \"<x>\" "
    "--format markdown`."
)

HARD_MSG_TEMPLATE = (
    "READ-BUDGET GUARD (hard): {reads} reads with no edit to production "
    "code. Stop and checkpoint. Choose ONE:\n"
    "  (a) Commit a TRIVIAL: declaration with the diff plan,\n"
    "  (b) Run `python scripts/tools/context_resolver.py --task \"<x>\" "
    "--format markdown` and narrow scope,\n"
    "  (c) Declare exploration complete and abort the current path."
)


def _cmd_inject() -> None:
    state = _load_state()
    reads = int(state.get("reads", 0))
    if reads < SOFT_CAP:
        return

    if reads >= HARD_CAP:
        msg = HARD_MSG_TEMPLATE.format(reads=reads)
        _emit(msg)
        state["last_warned_at"] = datetime.now(UTC).isoformat()
        _save_state(state)
        return

    # Soft cap with cooldown.
    last = state.get("last_warned_at")
    if last:
        try:
            age_min = (datetime.now(UTC) - datetime.fromisoformat(last)).total_seconds() / 60
            if age_min < SOFT_COOLDOWN_MINUTES:
                return
        except (TypeError, ValueError):
            pass

    msg = SOFT_MSG_TEMPLATE.format(reads=reads)
    _emit(msg)
    state["last_warned_at"] = datetime.now(UTC).isoformat()
    _save_state(state)


def _emit(message: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": message,
        }
    }
    print(json.dumps(payload))


COMMANDS = {
    "increment": _cmd_increment,
    "reset-on-edit": _cmd_reset_on_edit,
    "reset-session": _cmd_reset_session,
    "inject": _cmd_inject,
}


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    handler = COMMANDS.get(cmd)
    if handler is None:
        sys.exit(0)
    try:
        handler()
    except Exception:
        pass  # fail-open
    sys.exit(0)


if __name__ == "__main__":
    main()
