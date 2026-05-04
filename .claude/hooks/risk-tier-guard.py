#!/usr/bin/env python3
"""Inject a compact risk-tier hint for expensive reasoning only when justified.

Goals:
- Keep the default session cheap for routine work.
- Escalate reasoning/review expectations on genuinely risky prompts.
- Emit concise context only when the tier changes or cooldown expires.

The output uses JSON `additionalContext`, so Claude receives the hint directly.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / ".claude" / "hooks" / ".risk-tier-state.json"
COOLDOWN_MINUTES = 20

CRITICAL_RE = re.compile(
    r"\b("
    r"real capital|live trading|deploy|production|promotion|promote|"
    r"broker|order routing|position sizing|risk limit|kill switch|"
    r"account routing|account safety|capital review|deploy readiness|"
    r"readiness|live path|runtime control|threat model|security review"
    r")\b",
    re.IGNORECASE,
)

HIGH_RE = re.compile(
    r"\b("
    r"pipeline|check_drift|schema|migration|duckdb|database|timezone|dst|"
    r"session boundary|session time|orb window|concurrency|worktree|hook|"
    r"mutex|review|audit|verify|validation|backtest|research|hypothesis|"
    r"holdout|oos|p.?value|fdr|slippage|cost model|execution engine|"
    r"trading_app/live|refresh_data|outcome_builder"
    r")\b",
    re.IGNORECASE,
)


def load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("last_tier", None)
    state.setdefault("last_at", None)
    return state


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def classify(prompt: str) -> tuple[str, str] | None:
    if CRITICAL_RE.search(prompt):
        return (
            "critical",
            "RISK TIER: critical. Keep exploration lean; reserve high-reasoning or an independent review for final decisions. Require execution evidence before done.",
        )
    if HIGH_RE.search(prompt):
        return (
            "high",
            "RISK TIER: high. Default to normal reasoning for exploration, then escalate only for review/decision points. Require targeted tests, drift, and explicit review.",
        )
    return None


def should_emit(state: dict, tier: str) -> bool:
    if state.get("last_tier") != tier:
        return True
    last_at = state.get("last_at")
    if not last_at:
        return True
    try:
        age_min = (datetime.now(UTC) - datetime.fromisoformat(last_at)).total_seconds() / 60
    except (TypeError, ValueError):
        return True
    return age_min >= COOLDOWN_MINUTES


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    prompt = str(event.get("prompt", ""))
    result = classify(prompt)
    if result is None:
        sys.exit(0)

    tier, context = result
    state = load_state()
    if not should_emit(state, tier):
        sys.exit(0)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    print(json.dumps(payload))

    state["last_tier"] = tier
    state["last_at"] = datetime.now(UTC).isoformat()
    save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    main()
