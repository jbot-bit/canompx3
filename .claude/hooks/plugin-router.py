#!/usr/bin/env python3
"""Plugin/data router for Claude UserPromptSubmit.

Injects one compact reminder when the prompt mentions external plugin domains.
This keeps Claude aligned with `.codex/PLUGIN_ROUTING.md` without making any
plugin a source of project truth.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_FILE = PROJECT_ROOT / ".claude" / "rules" / "plugin-routing.md"

ROUTES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(datadog|logs?|metrics?|traces?|apm|latency|p99|monitor|monitors|incident|observability|alerting)\b",
            re.IGNORECASE,
        ),
        "PLUGIN ROUTE: Datadog only for instrumented runtime telemetry; otherwise use repo runtime artifacts and propose instrumentation.",
    ),
    (
        re.compile(
            r"\b(marcopolo|external data|warehouse|lakehouse|s3|crm|jira|api export|broker export|data source)\b",
            re.IGNORECASE,
        ),
        "PLUGIN ROUTE: MarcoPolo requires a named external source/scope and must not replace gold-db/repo MCP truth.",
    ),
    (
        re.compile(r"\b(supabase|postgres|postgresql|rls|auth|storage|edge function|pgvector)\b", re.IGNORECASE),
        "PLUGIN ROUTE: Supabase is for real Postgres/Auth/RLS app work; canompx3 market data stays DuckDB/gold-db.",
    ),
    (
        re.compile(r"\b(spreadsheet|xlsx|workbook|slides?|deck|presentation|docx|word document)\b", re.IGNORECASE),
        "PLUGIN ROUTE: Artifact plugins are explicit-output only; durable canompx3 truth stays Markdown/YAML/code/DB.",
    ),
    (
        re.compile(r"\b(gmail|email|calendar|meeting|circleback|what did we discuss)\b", re.IGNORECASE),
        "PLUGIN ROUTE: Gmail/Calendar/Circleback are personal/admin context only, never repo or trading authority.",
    ),
    (
        re.compile(r"\b(circleci|circle ci)\b", re.IGNORECASE),
        "PLUGIN ROUTE: CircleCI is explicit-only; default canompx3 CI is GitHub Actions.",
    ),
    (
        re.compile(r"\b(posthog|hugging face|huggingface|agent sdk|plugin-dev|skill-creator)\b", re.IGNORECASE),
        "PLUGIN ROUTE: This plugin is explicit-only unless the task is analytics, model/dataset work, agent apps, or durable plugin/skill editing.",
    ),
]


def _load_event() -> dict[str, object]:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
        if not raw:
            return {}
        event = json.loads(raw)
        return event if isinstance(event, dict) else {}
    except Exception:
        return {}


def _classify(prompt: str) -> str | None:
    for pattern, message in ROUTES:
        if pattern.search(prompt):
            return message
    return None


def main() -> int:
    if not RULES_FILE.exists():
        return 0
    event = _load_event()
    prompt = str(event.get("prompt") or "")
    if len(prompt.split()) < 2:
        return 0
    message = _classify(prompt)
    if not message:
        return 0
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": message,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

