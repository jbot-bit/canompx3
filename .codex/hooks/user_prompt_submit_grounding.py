#!/usr/bin/env python3
"""Inject compact grounding context only for prompts that need it."""

from __future__ import annotations

import json
import re
import sys

PROMPT_PATTERNS = {
    "claim_scrutiny": re.compile(
        r"\b(review|audit|deploy|deployment|promot|readiness|production|capital|research|result|results|"
        r"backtest|strategy|validation|validate|oos|live|safety)\b"
    ),
    "codex_layer": re.compile(r"\b(codex|mcp|plugin|plugins|skill|skills|hook|hooks|agent|agents)\b"),
}


def _load_payload() -> dict[str, object]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_context(prompt: str) -> list[str]:
    text = prompt.lower()
    hints: list[str] = []

    if PROMPT_PATTERNS["claim_scrutiny"].search(text):
        hints.extend(
            [
                "Treat prior summaries as claims, not evidence.",
                "Lead with disconfirming checks and label conclusions `MEASURED`, `INFERRED`, or `UNSUPPORTED`.",
                "Ground thresholds and methodology in repo-local literature or primary sources.",
            ]
        )

    if PROMPT_PATTERNS["codex_layer"].search(text):
        hints.append(
            "Prefer Codex-owned surfaces (`.codex/`, `.agents/skills`, `scripts/infra/codex_*`) and do not mutate Claude-owned surfaces unless explicitly requested."
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        deduped.append(hint)
    return deduped


def main() -> int:
    payload = _load_payload()
    prompt = str(payload.get("prompt") or "")
    hints = _build_context(prompt)
    if not hints:
        return 0

    print(
        json.dumps(
            {
                "continue": True,
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n".join(f"- {hint}" for hint in hints),
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
