# Subagent Architecture Design

**Date:** 2026-03-07
**Status:** Approved
**Goal:** Maximize code quality, speed, and context efficiency through autonomous subagents

## Problem Statement

Three recurring failure modes in the current workflow:
1. **Incomplete understanding before editing** — code changes made without fully mapping callers, importers, tests, and downstream effects
2. **Missing verification after editing** — changes claimed "done" without running drift checks, tests, or behavioral audits
3. **Expensive routine queries** — simple data lookups (trade book, fitness, strategy counts) burn Opus tokens and take 30+ seconds

## Solution: 3 Project Subagents

### Agent 1: `blast-radius` (Pre-Edit Scout)
- **Purpose:** Read-only impact analysis BEFORE any production code change
- **Model:** `sonnet` (fast, cheap, good at code reading)
- **Tools:** Read, Grep, Glob, Bash (no Edit, no Write)
- **Memory:** `project` (learns codebase patterns over time)
- **Trigger:** Auto-dispatched before modifying production code in pipeline/ or trading_app/

### Agent 2: `verify-complete` (Post-Edit Auditor)
- **Purpose:** Verify completeness and correctness AFTER code changes
- **Model:** `sonnet` (fast execution)
- **Tools:** Read, Grep, Glob, Bash, Edit (with strict guardian rules)
- **Memory:** `project` (remembers recurring failures)
- **Background:** `true` (non-blocking)
- **Edit constraints:** Minimal fixes only (lint, imports, test assertions). Cannot refactor or restructure.

### Agent 3: `db-analyst` (Fast Data Lookups)
- **Purpose:** All gold.db queries — strategies, fitness, trade book, performance
- **Model:** `haiku` (fast, cheap, perfect for SQL + formatting)
- **Tools:** Bash, Read (no Edit, no Write)
- **MCP:** `gold-db`
- **Trigger:** Any question about strategies, performance, fitness, or trading data

## What's NOT Redundant

| Existing Skill | New Agent | Relationship |
|---|---|---|
| `/bloomey-review` | `blast-radius` | Different trigger: skill is post-hoc manual review, agent is pre-edit automatic scout |
| `/health-check` | `verify-complete` | Different trigger: skill is manual, agent auto-runs after edits |
| `/trade-book`, `/regime-check` | `db-analyst` | Same domain, different model: Haiku vs Opus = 10x cheaper/faster |
| `/quant-debug` | (none) | No overlap — remains user-triggered for real debugging |

## Guardian Rules (for verify-complete edits)

1. Never edit what you haven't read
2. Never claim fixed without showing test output
3. Fail-closed: if verification fails after edit, revert and report
4. Minimal diff only — no surrounding cleanup
5. One-way dependency: never import trading_app/ from pipeline/
6. Canonical sources only — never hardcode lists/numbers
7. Evidence before assertion — command output required
