# MCP Config Patch — `sequential-thinking` (UNCOMMITTED)

**Date:** 2026-05-01
**Status:** **UNCOMMITTED** — DO NOT merge into `.mcp.json` until eval completes and `discipline-checklist.md` reports PASS.
**Scope:** propose-only.

> **Note on `memory/<file>.md` references:** throughout this doc, `memory/<file>.md` paths resolve to the auto-memory directory at `C:/Users/joshd/.claude/projects/C--Users-joshd-canompx3/memory/` (per the `claudeMd` "auto memory" section), not the project repo.

---

## Proposed `.mcp.json` block

Append the following entry to the `mcpServers` object in `.mcp.json`:

```json
"sequential-thinking": {
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-sequential-thinking"
  ]
}
```

This matches the upstream README **verbatim** (verified 2026-05-01 via WebFetch against `https://raw.githubusercontent.com/modelcontextprotocol/servers/main/src/sequentialthinking/README.md`):

> **NPX Invocation:** `npx -y @modelcontextprotocol/server-sequential-thinking`
>
> **Claude Desktop Config Block:** `{ "mcpServers": { "sequential-thinking": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"] } } }`

## Tool exposed

A single tool: `sequential_thinking`.

Input schema (verbatim from upstream README):

- `thought` (string): The current thinking step
- `nextThoughtNeeded` (boolean): Whether another thought step is needed
- `thoughtNumber` (integer): Current thought number
- `totalThoughts` (integer): Estimated total thoughts needed
- `isRevision` (boolean, optional): Whether this revises previous thinking
- `revisesThought` (integer, optional): Which thought is being reconsidered
- `branchFromThought` (integer, optional): Branching point thought number
- `branchId` (string, optional): Branch identifier
- `needsMoreThoughts` (boolean, optional): If more thoughts are needed

## Constraint check — `cryptography<47` pin

Per `memory/feedback_mcp_venv_drift_cryptography47.md` (referenced from `MEMORY.md` index): "`cryptography==47` removed `hazmat.backends`; `authlib==1.7` still imports it; FastMCP servers crash. Quick: `pip install 'cryptography<47'`. Real: `uv sync --frozen`."

**Applicability to sequential-thinking:** the upstream package `@modelcontextprotocol/server-sequential-thinking` is **Node-based** (npm, invoked via `npx`). It has **no Python dependency chain**, so the `cryptography<47` pin does **not** apply to this server. The pin remains binding for FastMCP / Authlib-based Python sidecars (CRG, gold-db, etc.) and is unaffected by this addition. Documenting explicitly so a future audit doesn't have to re-derive.

## Constraint check — local-scope shadowing

Per `memory/feedback_mcp_local_scope_shadows_project_scope.md`: "MCP local-scope shadows project-scope `.mcp.json` SILENTLY — same server name in both = local wins, `.mcp.json` env block IGNORED."

**Mitigation when this patch lands:** before merging into `.mcp.json`, run `claude mcp get sequential-thinking` to confirm there is no pre-existing local-scope entry shadowing it. If one exists, `claude mcp remove sequential-thinking -s local`, then restart.

## Constraint check — env-block hot-reload

Per `memory/feedback_mcp_env_requires_restart.md`: stdio MCP keeps spawn-time env, so any future env edits require Claude Code restart, not just config save. This server has no env block in the proposed patch, so the rule is moot for the initial add — but binding for any future modification.

## Why UNCOMMITTED

Per task spec hard rule: "Do NOT touch `.mcp.json`." This file is the **proposal**, not the patch. The patch lands only after:

1. Eval session runs the three rubric incidents with seq-thinking enabled.
2. `discipline-checklist.md` aggregates PASS across all four gates.
3. User approves the merge.

## Rollback plan

If post-merge eval shows regressions (e.g. seq-thinking output cited as "evidence" for the four `done` artefacts in institutional-rigor rule 8, in violation of the doctrine-no-regression gate), remove the entry from `.mcp.json` and document the regression in `memory/feedback_seq_thinking_*.md`.
