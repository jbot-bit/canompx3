#!/usr/bin/env python3
"""Inject a compact anti-bias / grounding reminder on relevant prompts.

The guard is intentionally short and selective:
- Fires only for research, review, deploy, audit, result-interpretation prompts
- Emits at most 1 short line
- Uses cooldowns to avoid repeating on every prompt

Goal: counter author-bias and unsupported claims without adding a heavy hook.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".bias-grounding-state.json"
DIRECTIVE_COOLDOWN_MINUTES = 20

TARGET_PROMPTS = re.compile(
    r"\b("
    r"research|review|audit|verify|validation|validate|bias|ground|"
    r"source|sources|literature|evidence|proof|prove|claim|claims|"
    r"result|results|deploy|promotion|promote|ready|readiness|"
    r"oos|holdout|backtest|significance|p.?value|fdr|dsr|sharpe"
    r")\b",
    re.IGNORECASE,
)

COMPACT_DIRECTIVE = (
    "RESEARCH MODE: Canon only, disconfirm first, tag MEASURED/INFERRED/UNSUPPORTED, then state edge, issue, next step."
)


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("last_key", None)
    state.setdefault("last_at", None)
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_emit(state: dict, key: str) -> bool:
    if key != state.get("last_key"):
        return True
    last_at = state.get("last_at")
    if not last_at:
        return True
    try:
        age_min = (datetime.now(UTC) - datetime.fromisoformat(last_at)).total_seconds() / 60
    except (TypeError, ValueError):
        return True
    return age_min >= DIRECTIVE_COOLDOWN_MINUTES


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    prompt = event.get("prompt", "")
    if not TARGET_PROMPTS.search(prompt):
        sys.exit(0)

    directives = [COMPACT_DIRECTIVE]
    key = " || ".join(directives)
    state = load_state()
    if not should_emit(state, key):
        sys.exit(0)

    print(COMPACT_DIRECTIVE, file=sys.stderr)

    state["last_key"] = key
    state["last_at"] = datetime.now(UTC).isoformat()
    save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    main()
