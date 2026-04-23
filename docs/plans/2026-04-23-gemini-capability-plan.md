# Gemini Capability and Orchestration Plan

**Date:** 2026-04-23
**Status:** ACTIVE
**Authority:** Defer to `CLAUDE.md`. Complements `CODEX.md`.

## Objective
Elevate Gemini CLI to be an advanced, capable peer to Claude and Codex in the Canompx3 repository, utilizing Gemini-native tools while strictly deferring to Claude's canonical authority.

## Background & Motivation
The Canompx3 repository is a high-rigor, institutional-grade quantitative trading environment governed by strict rules (`RESEARCH_RULES.md`, `TRADING_RULES.md`, `HANDOFF.md`). Claude is established as the "top dog and canonical authority" (`CLAUDE.md`). Codex serves as the secondary implementation and audit layer (`CODEX.md`). 

To match and exceed this capability, Gemini must adopt the same rigorous institutional standards (Bailey-López de Prado discipline, Fail-closed logic, 2-Pass Implementation) but execute them using its unique native architecture: sub-agent delegation, parallel tool calling, and persistent project memory.

## Scope & Impact
- **Authority:** Gemini will operate as a high-speed, parallel-executing orchestrator. It will explicitly defer to `CLAUDE.md` and the `.claude/` directory for architectural truth and conflict resolution.
- **Execution:** Gemini will fully adopt the "2-Pass Implementation Method" and "Design Proposal Gate" mandated by the project.
- **Data Truth:** Gemini will respect `gold.db` as the sole data truth and use `run_shell_command` to interface with existing project tools (e.g., `context_resolver.py`) when native investigation requires it.

## Proposed Solution: Gemini-Native Orchestration

### 1. Deference to Claude (The "Boss")
Gemini will internalize the hierarchy defined in `CODEX.md`:
- Claude (`CLAUDE.md`, `.claude/`) dictates the architecture and rules.
- Gemini operates as a strategic orchestrator for implementation, review, verification, and audit.
- Gemini will never mutate `CLAUDE.md` or `.claude/` configurations unless explicitly requested.

### 2. Gemini-Native Workflow Mapping
Gemini will replace sequential script execution with parallel, native tools where advantageous, but fall back to canonical scripts when project truth is required.

| Project Mandate | Claude/Codex Tooling | Gemini Native Execution Strategy |
| :--- | :--- | :--- |
| **Discovery / Context** | `context_resolver.py` / MCP `get_canonical_context` | Invoke **`codebase_investigator`** subagent for deep architectural mapping; parallel `grep_search` and `read_file` for instant context gathering. Use `context_resolver.py` via shell when querying canonical doctrine. |
| **Database Queries** | MCP `query_trading_db` | Use `run_shell_command` with DuckDB CLI or Python scripts to query `gold.db` directly and safely, ensuring read-only constraints are met. |
| **2-Pass Implementation** | Sequential reading, writing, and testing | Parallel file reads for context -> Plan Mode (Design Proposal Gate) -> Implement -> Parallel execution of `pytest` and `check_drift.py` via shell. |
| **Long-term Grounding** | Session context / `.claude/memory/` | Continuous use of **`save_memory`** (project scope) to persist critical institutional truths, current strategy branches (e.g., Phase D Gate 0), and "KILL/ALIVE" verdicts across sessions. |
| **Batch/Repetitive Tasks** | Sequential scripting | Invoke **`generalist`** subagent to handle massive refactors, complex log parsing, or multi-file fixes efficiently without clogging the main session context. |

### 3. Operational Directives
When executing any task in this repo, Gemini will automatically:
1.  **Check Context:** Read `HANDOFF.md` and recent `docs/plans/` using `read_file` or `glob`.
2.  **Gate Verification:** Before proposing code, simulate the happy path, edge cases, and failure modes.
3.  **Audit First:** Run `python pipeline/check_drift.py` and `pytest` after any implementation before claiming a task is done.
4.  **Update Handoff:** Ensure `HANDOFF.md` is updated if durable decisions or unfinished work remains.

## Alternatives Considered
- **Strict Emulation:** Acting exactly like Codex by exclusively relying on `run_shell_command` to drive all interactions through existing Python scripts. *Rejected* because it ignores Gemini's powerful parallel processing and subagent delegation capabilities, resulting in slower execution and higher token usage.

## Verification
- Gemini will use `save_memory` to permanently imprint this operational model for the Canompx3 project.
- The first task executed under this model will require zero human prompting to run `check_drift.py` or verify against `TRADING_RULES.md`.
