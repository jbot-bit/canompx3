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
    "live_risk": re.compile(r"\b(live|broker|webhook|kill|flatten|account|profile|prop|order|execution|risk)\b"),
    "research_method": re.compile(
        r"\b(research|backtest|strategy|validation|validate|oos|holdout|fdr|dsr|minbtl|edge|significant)\b"
    ),
    "test_gap": re.compile(r"\b(test gap|coverage|pytest|missing tests|stale test|verify|verification)\b"),
    "implementation": re.compile(r"\b(implement|fix|build|change|edit|patch|do it|go)\b"),
    "debugging": re.compile(r"\b(debug|bug|error|failure|failing|broken|regression|flake|flaky)\b"),
    "parallel_agents": re.compile(r"\b(parallel|subagent|subagents|swarm|agent team|agents|independent tasks)\b"),
    "completion": re.compile(r"\b(done|complete|finished|ship|commit|push|pr|pull request|merge)\b"),
    "superpowers": re.compile(r"\b(superpowers|plugin|plugins|skill|skills)\b"),
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
                "Ground thresholds and methodology in `docs/institutional/literature/`; use `resources/` raw PDFs only when an extract is missing.",
            ]
        )

    if PROMPT_PATTERNS["research_method"].search(text):
        hints.extend(
            [
                "For research or strategy-validation claims, route through `research-methodologist` / `canompx3_reviewer` and require local-literature grounding.",
                "Do not call a result edge/significant/validated/deployable without RESEARCH_RULES.md evidence plus `docs/institutional/literature/` support.",
            ]
        )

    if PROMPT_PATTERNS["live_risk"].search(text):
        hints.append(
            "For live, broker, webhook, account-routing, kill/flatten, or prop-profile work, route through `live-risk-auditor` / `canompx3_reviewer` before trusting readiness."
        )

    if PROMPT_PATTERNS["test_gap"].search(text):
        hints.append(
            "For coverage or verification uncertainty, route through `test-coverage-scout` / `canompx3_reviewer` and return exact pytest targets."
        )

    if PROMPT_PATTERNS["implementation"].search(text):
        hints.append(
            "For non-trivial implementation, use a scoped writer (`executor` or `canompx3_worker`) only after blast-radius/planning; no parallel same-file edits."
        )

    if PROMPT_PATTERNS["debugging"].search(text):
        hints.append(
            "For bugs, failures, flakes, or regressions, use `superpowers:systematic-debugging` under canompx3 rules; identify evidence before changing code."
        )

    if PROMPT_PATTERNS["parallel_agents"].search(text):
        hints.append(
            "For parallel work, use `superpowers:dispatching-parallel-agents` only for independent domains; prefer read-only scouts and never allow same-file or parallel DB writers."
        )

    if PROMPT_PATTERNS["completion"].search(text):
        hints.append(
            "Before claiming completion, use `superpowers:verification-before-completion` plus canompx3 verification gates; report actual command evidence and residual gaps."
        )

    if PROMPT_PATTERNS["superpowers"].search(text):
        hints.append(
            "Superpowers is useful for process discipline, but canompx3 authority wins: no unmanaged `docs/superpowers/` truth, no unmanaged worktrees, and no generic TDD override of research/live gates."
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
