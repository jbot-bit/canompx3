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
    "quality_grounding": re.compile(
        r"\b(check|improve|implement|fix|build|change|edit|patch|review|audit|verify|plan|design|"
        r"hardening|cleanup|refactor|debug|investigate|diagnose|make (it|this|that) better|do it|go|"
        r"2p|second[- ]pass|two[- ]pass|double[- ]check|fresh eyes|another pass|sanity check|"
        r"red[- ]team|critique|stress[- ]test|take a look|look over|is this good|"
        r"what am i missing|poke holes|blind spots?|hold up|sense check|smell test|"
        r"thoughts on|qa|will this work|does this work|spot flaws?|find flaws?|gotchas?|risks?)\b"
    ),
    "parallel_agents": re.compile(r"\b(parallel|subagent|subagents|swarm|agent team|agents|independent tasks)\b"),
    "completion": re.compile(r"\b(done|complete|finished|ship|commit|push|pr|pull request|merge)\b"),
    "superpowers": re.compile(r"\b(superpowers|plugin|plugins|skill|skills)\b"),
    "observability": re.compile(
        r"\b(datadog|logs?|metrics?|traces?|apm|latency|p99|monitor|monitors|incident|observability|alerting)\b"
    ),
    "external_data": re.compile(
        r"\b(marcopolo|external data|warehouse|lakehouse|s3|crm|jira|api export|broker export|data source)\b"
    ),
    "supabase": re.compile(r"\b(supabase|postgres|postgresql|rls|auth|storage|edge function|pgvector)\b"),
    "artifact_plugins": re.compile(
        r"\b(spreadsheet|xlsx|workbook|slides?|deck|presentation|docx|word document)\b"
    ),
    "personal_context": re.compile(r"\b(gmail|email|calendar|meeting|circleback|what did we discuss)\b"),
    "circleci": re.compile(r"\b(circleci|circle ci)\b"),
    "resource_grounding": re.compile(
        r"(^|\s)/(resource|resources|lit|literature)\b|\b(resource grounding|local literature|local corpus|"
        r"trading bible|trading bibles|canonical truth|grounding truth)\b"
    ),
    "research_fetch_sources": re.compile(
        r"\b(research|fetch|look up|lookup|search|find sources?|source this|documentation|docs|"
        r"official sources?|primary sources?|changelog|release notes?|upgrade|upgrades|fixes|"
        r"user comments?|forum|forums|reddit|github issues?|stackoverflow)\b"
    ),
}


def _load_payload() -> dict[str, object]:
    raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace").strip()
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

    if PROMPT_PATTERNS["quality_grounding"].search(text):
        hints.append(
            "Run a compact truth check before acting, then second-pass the plan/work for gaps, silences, bias, errors, and simpler improvements. Keep it targeted; widen only if risk demands it."
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

    if PROMPT_PATTERNS["observability"].search(text):
        hints.append(
            "For observability prompts, use Datadog only for instrumented runtime telemetry; otherwise ground in repo runtime artifacts first and propose instrumentation instead of inventing Datadog truth."
        )

    if PROMPT_PATTERNS["external_data"].search(text):
        hints.append(
            "For external-source analysis, MarcoPolo is allowed only with a named source/scope; it must not replace `gold-db`, `repo-state`, `research-catalog`, or `strategy-lab` as canompx3 truth."
        )

    if PROMPT_PATTERNS["supabase"].search(text):
        hints.append(
            "Use Supabase only for real Supabase/Postgres/Auth/RLS/Storage work; canompx3 market data stays DuckDB/`gold-db` unless a deliberate migration is requested."
        )

    if PROMPT_PATTERNS["artifact_plugins"].search(text):
        hints.append(
            "Use Spreadsheets/Presentations/Documents only for explicit artifact deliverables; default canompx3 durable truth remains Markdown, YAML, code, DB, and generated audit/runtime files."
        )

    if PROMPT_PATTERNS["personal_context"].search(text):
        hints.append(
            "Use Gmail/Google Calendar/Circleback only for personal/admin context the user asked for; never treat them as repo or trading-system authority."
        )

    if PROMPT_PATTERNS["circleci"].search(text):
        hints.append(
            "CircleCI is explicit-only in canompx3; default CI investigation should use GitHub Actions unless the user specifically names CircleCI."
        )

    if PROMPT_PATTERNS["resource_grounding"].search(text):
        hints.append(
            "RESOURCE ROUTE: local-PC corpus only. Run PDF/tooling + coverage checks; use extracts when covered, raw resources only if present locally; no skim/guess."
        )

    if PROMPT_PATTERNS["research_fetch_sources"].search(text):
        hints.append(
            "SOURCE ROUTE: separate official/primary sources from unofficial user reports. Use official docs/changelogs/releases/code first; treat forums/issues/comments as cautionary signals unless corroborated."
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
