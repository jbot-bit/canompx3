---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Claude Code Standards Upgrade — Design Doc

**Date:** 2026-03-15
**Status:** Draft — pending user approval
**Motivation:** Token efficiency + Focus — reduce startup context, sharpen Claude behavior on core rules, align with official Anthropic best practices from code.claude.com.

---

## Background

### Official Standards (from code.claude.com)

| Principle | Source |
|-----------|--------|
| CLAUDE.md: target <200 lines, prune ruthlessly | [memory docs](https://code.claude.com/docs/en/memory) |
| Rules: use `paths:` frontmatter to scope, save context | [memory docs](https://code.claude.com/docs/en/memory#path-specific-rules) |
| Skills: on-demand knowledge, `.claude/skills/<name>/SKILL.md` | [skills docs](https://code.claude.com/docs/en/skills) |
| Skills: `disable-model-invocation: true` for side-effects | [skills docs](https://code.claude.com/docs/en/skills#control-who-invokes-a-skill) |
| Skills: `allowed-tools` for read-only skills | [skills docs](https://code.claude.com/docs/en/skills#restrict-tool-access) |
| Skills: `context: fork` for isolated execution | [skills docs](https://code.claude.com/docs/en/skills#run-skills-in-a-subagent) |
| Plugins: each adds skill descriptions + MCP schemas to context | [features overview](https://code.claude.com/docs/en/features-overview) |
| Hooks: zero context cost, deterministic | [hooks docs](https://code.claude.com/docs/en/hooks) |
| Subagents: isolated context, model/tool restrictions | [subagents docs](https://code.claude.com/docs/en/sub-agents) |

### Current State Audit

| Area | Finding |
|------|---------|
| Always-loaded context | ~50KB (CLAUDE.md + ARCHITECTURE.md + 8 unscoped rules + MEMORY.md) |
| Plugins | 15 enabled — 4 are dead, duplicate, or irrelevant |
| Commands | 21 in legacy `.claude/commands/` — should be `.claude/skills/` with frontmatter |
| Dead systems | NotebookLM MCP configured but retired; claude-mem hooks firing on every tool call but no data recorded since Feb 28 |
| Memory | 3 systems: auto-memory (ACTIVE, 41 files, 158KB), Pinecone (ACTIVE, 51 files), claude-mem (DEAD, 251MB DB, last record Feb 28) |
| Hooks | `pre-edit-guard.py` exists but NOT wired — gold.db/`.env` protection is unenforced |
| Permissions | 153 entries including 13 retired notebooklm and 20+ stale one-off Bash commands |

### Memory System Comparison (Why claude-mem is being removed)

| | Auto-memory (KEEPING) | Pinecone (KEEPING) | claude-mem (REMOVING) |
|---|---|---|---|
| Quality | 41 curated topic files, each a focused reference | 51 structured files across 5 tiers | 5,083 raw observations, unstructured session dumps |
| Routing | Recall: "what do we know about X" | Deep knowledge: "what did we find about Y" | Single flat search (redundant) |
| Active | Updated today (Mar 15) | Last sync active | Dead since Feb 28 |
| Size | 158KB plain markdown | Cloud-hosted | 251MB SQLite |
| Overhead | Zero hooks — MEMORY.md loads at startup | MCP tool schemas only | Hooks on EVERY tool call, even while broken |

---

## Phase 1: Free Wins (Zero Risk)

All items independently verified via blast radius analysis. Zero production code affected.

### 1A. Disable dead/duplicate plugins

| Plugin | Action | Reason | Blast Radius |
|--------|--------|--------|-------------|
| `claude-mem@claude-plugins-official` | **DISABLE** | Dead since Feb 28. Hooks fire every tool call for nothing. 4 permission entries become orphaned (harmless). No .py or command file references it. | **ZERO** |
| `superpowers@claude-plugins-official` | **DISABLE** | Duplicate of superpowers-extended-cc. All skills exist in both; extended-cc adds CC-native TaskCreate. The one live reference (workflow_method.md) calls extended-cc, not vanilla. No permission entries exist for vanilla. | **ZERO** |
| `coderabbit@claude-plugins-official` | **DISABLE** | Duplicate code review — keeping greptile. No file in the project references coderabbit. No permission entries. | **ZERO** |
| `frontend-design@claude-plugins-official` | **DISABLE** | Not relevant — we build Streamlit dashboards, not React/Vue. No file references it. No permission entries. | **ZERO** |

**Superpowers skill coverage after disabling vanilla:**
All skills used (brainstorming, requesting-code-review, writing-plans, executing-plans, verification-before-completion, test-driven-development, systematic-debugging, dispatching-parallel-agents, finishing-a-development-branch, receiving-code-review, using-git-worktrees, writing-skills) exist identically in superpowers-extended-cc with additional CC-native task management integration. Nothing is lost.

**Files changed:** `.claude/settings.json` (`enabledPlugins` — set 4 entries to `false`)
**Verification:** Start new Claude Code session, run `/trade-book`, confirm all skills listed via "what skills are available?"

### 1B. Remove retired NotebookLM MCP

- Delete `notebooklm` entry from `.mcp.json` (lines 11-22)
- Remove 13 `mcp__notebooklm__*` entries from `settings.local.json`
- NotebookLM server was already NOT in `enabledMcpjsonServers` — doubly dead
- `.claude/rules/notebooklm.md` is NOT deleted — it provides valid PDF routing and already says "NotebookLM MCP has been retired"

**Blast radius:** ZERO. Server was not functional. Rule file stays. Codex docs (.codex/INTEGRATIONS.md line 23-25) reference it as "retired" — minor staleness, informational only.
**Files changed:** `.mcp.json`, `.claude/settings.local.json`
**Verification:** Start new session, run `/mcp` to confirm only gold-db remains

### 1C. Wire `pre-edit-guard.py` as PreToolUse hook

The script exists at `.claude/hooks/pre-edit-guard.py`, is production-ready, blocks writes to `gold.db`, `gold.db.wal`, and `.env` via exit code 2. Currently unwired — no enforcement.

Add to `.claude/settings.json`:
```json
"PreToolUse": [
  {
    "matcher": "Edit|Write",
    "hooks": [
      {
        "type": "command",
        "command": "python .claude/hooks/pre-edit-guard.py",
        "timeout": 5
      }
    ]
  }
]
```

**Blast radius:** LOW. No existing workflow legitimately needs to Edit/Write gold.db directly. The hook does NOT block Bash commands that write to gold.db via DuckDB (only intercepts Edit/Write tool calls). Independent of existing PostToolUse hooks.
**Gap:** No test file exists for `pre-edit-guard.py`. Consistent with existing hook pattern (post-edit hooks also untested). Not blocking.
**Files changed:** `.claude/settings.json`
**Verification:** Attempt `Write` to a file named "test-gold.db" — should be blocked

### 1D. Clean permissions allow-list

| Category | Count | Action |
|----------|-------|--------|
| Retired `mcp__notebooklm__*` | 13 | Remove |
| Stale Bash one-offs (full path commands from old sessions) | ~20 | Remove |
| Active WebFetch domains | ~90 | Keep |
| Active MCP tools | ~22 | Keep |
| Active Skills | 6 | Keep |

**Blast radius:** ZERO. Removed permissions reference tools/commands that no longer exist. Removing them has no effect — they were granting permission to ghosts.
**Files changed:** `.claude/settings.local.json`
**Verification:** Session starts normally, no unexpected permission prompts for routine operations

---

## Phase 2: Context Diet (Low Risk)

### 2A. Scope specialist rules

| Rule File | Lines | Change | Blast Radius | Risk Notes |
|-----------|-------|--------|-------------|------------|
| `notebooklm.md` (42 lines) | Always → `paths: ["resources/**"]` | **LOW** | Only loaded when accessing resources/ PDFs. Risk: "where is the BH FDR paper?" without resources/ open won't have routing table. Mitigated: pinecone-assistant.md cross-references it. |
| `pinecone-assistant.md` (82 lines) | Always → **Convert to skill** | **LOW** | ~~Path-scoping to `scripts/tools/sync_pinecone*` is TOO NARROW~~ — this rule is needed for any "what did we find about X" query, not just syncs. Converting to a model-invocable skill with a rich description triggers on knowledge-routing queries semantically. |
| `audit-prompts.md` (53 lines) | Always → **Convert to skill** | **LOW** | Content is reactive guidance for audits. Model-invocable skill triggers on "audit", "health check", "guardian prompt". 2 Codex files need updating: `.codex/RULES.md`, `.codex/skills/canompx3-research/SKILL.md`. |
| `quant-agent-identity.md` (36 lines) | Always → `paths: ["pipeline/**", "trading_app/**", "research/**", "scripts/**"]` | **LOW** | Seven Sins awareness loads only when working with code. Not needed for data queries, git ops, or doc editing. |

**Context savings:** ~213 lines / ~12.5KB removed from sessions that don't trigger these paths/skills.
**Verification:** After scoping, test each scenario: ask about BH FDR, ask "what did we find about compressed spring", ask for audit — confirm routing still works.

### 2B. Deduplicate CLAUDE.md / ARCHITECTURE.md / rules

| Duplication | Where | Action |
|-------------|-------|--------|
| Fail-closed / Idempotent / One-way dependency | CLAUDE.md L46-49 AND `pipeline-patterns.md` L7-9 | Remove from `pipeline-patterns.md` (CLAUDE.md is authority) |
| Strategy classification thresholds | ARCHITECTURE.md AND `validation-workflow.md` | Remove from `validation-workflow.md`, add "see ARCHITECTURE.md" reference |
| DST FULLY RESOLVED | CLAUDE.md L57-59 AND pipeline-patterns.md | Remove from `pipeline-patterns.md` |
| Document authority table (13 entries) | CLAUDE.md L18-32 | Trim to 5 core docs + "see REPO_MAP.md for full index" |

**Blast radius:** LOW. Removing duplicated text — the authoritative copy remains. Rules that lost text still have their unique, non-duplicated content.
**Estimated savings:** ~30-40 lines across CLAUDE.md + rules.
**Verification:** `python pipeline/check_drift.py` passes. Read modified rules files to confirm remaining content is complete.

---

## Phase 3: Skills Migration (Medium Risk — Incremental)

### 3A. Migrate `.claude/commands/` → `.claude/skills/`

Anthropic merged commands into skills. Our 21 commands work but miss: `disable-model-invocation`, `allowed-tools`, `context: fork`, `agent`, supporting files. Migration is **incremental** — per Anthropic docs, skills take precedence over commands with the same name, so create skill → test → delete command.

**Frontmatter per skill type:**

| Type | Skills | Key Frontmatter | Rationale |
|------|--------|-----------------|-----------|
| Side-effect workflows | `rebuild-outcomes`, `validate-instrument`, `post-rebuild` | `disable-model-invocation: true` | User-triggered only — prevents Claude from auto-running rebuilds |
| Read-only queries | `trade-book`, `regime-check`, `discover` | `allowed-tools: Read, Grep, Glob, Bash` | No writes needed — query + display only |
| Design/planning | `4t`, `4tp`, `bloomey-review` | Default (model-invocable) | Claude should auto-trigger on "plan", "design", "review" |
| Debugging | `quant-debug`, `quant-tdd` | Default | Claude should auto-trigger on bugs/TDD |
| Verification | `quant-verify`, `verify-complete`, `integrity-guardian`, `health-check` | Default | Claude should auto-trigger on verification needs |
| Audit | `audit`, `audit-quick`, `audit-phase`, `m25-audit` | `disable-model-invocation: true` | User-triggered only — audits are heavyweight |
| Analysis | `blast-radius` | `context: fork`, `agent: blast-radius` | Runs in isolated subagent context |
| Autonomous | `ralph` | `disable-model-invocation: true` | User-triggered only — runs autonomously |

**Directory structure per skill:**
```
.claude/skills/<name>/
├── SKILL.md        # Frontmatter + existing content from commands/<name>.md
└── (optional supporting files — templates, reference docs)
```

**Name conflicts (command + agent share same name):**
- `blast-radius`: command dispatches the agent — skill replaces command, agent stays. No conflict (agents ≠ skills).
- `verify-complete`: same pattern — skill replaces command, agent stays.

**Codex files that reference `.claude/commands/` (must update after migration):**
1. `.codex/COMMANDS.md` — 11 explicit command paths
2. `.codex/WORKFLOWS.md` line 37 — `4t.md` path
3. `.codex/skills/canompx3-audit/SKILL.md` — 4 command paths
4. `.codex/skills/canompx3-workspace/SKILL.md` — generic reference
5. `.codex/skills/canompx3-verify/SKILL.md` — 2 command paths
6. `CODEX.md` lines 50, 102 — 2 references

These are **Codex documentation updates only** — they do not affect Claude Code operation.

**Migration order (lowest risk first):**
1. Batch 1: Read-only skills (`trade-book`, `regime-check`, `discover`) — most-used, easiest to test
2. Batch 2: Design skills (`4t`, `4tp`, `bloomey-review`) — model-invocable, test auto-triggering
3. Batch 3: Debug/verify skills (`quant-debug`, `quant-tdd`, `quant-verify`, `verify-complete`, `integrity-guardian`, `health-check`)
4. Batch 4: Side-effect + audit skills (`rebuild-outcomes`, `validate-instrument`, `post-rebuild`, `audit`, `audit-quick`, `audit-phase`, `m25-audit`)
5. Batch 5: Special skills (`blast-radius` with `context: fork`, `ralph` with `disable-model-invocation`)

**Per-batch process:**
1. Create `.claude/skills/<name>/SKILL.md` with frontmatter + existing content
2. Test: invoke `/name` — confirm skill loads and executes correctly
3. Delete `.claude/commands/<name>.md`
4. Re-test: invoke `/name` — confirm skill still works without command fallback
5. Commit batch

**Blast radius per batch:** LOW if incremental. Skills take precedence during coexistence, so the transition is safe. The old command serves as automatic fallback until deleted.
**Verification:** After each batch, invoke each migrated skill and confirm behavior matches the old command.

---

## Phase 4: Standards Reference (Zero Risk)

### 4A. Create `.claude/skills/claude-code-standards/SKILL.md`

A compressed reference of Anthropic's official best practices, source-linked. Created as a **skill** (not always-loaded), invocable via `/claude-code-standards` or auto-loaded when Claude needs guidance on where to put new instructions.

```yaml
---
name: claude-code-standards
description: Official Anthropic Claude Code best practices. Reference for where instructions belong (CLAUDE.md vs rules vs skills), context budget, skill authoring, subagent design, hook patterns.
disable-model-invocation: true
---
```

**Contents (~100-150 lines):**
- Feature taxonomy: CLAUDE.md vs rules vs skills vs MCP vs hooks vs subagents
- Decision framework: "where does this instruction belong?"
- Context budget: 200-line CLAUDE.md, path-scoped rules, skill descriptions budget
- Skill authoring: frontmatter fields, when to use each option
- Anti-patterns: kitchen sink session, over-specified CLAUDE.md, trust-then-verify gap
- Links to official docs for each section

**Blast radius:** ZERO — new file, no existing file modified.

---

## Gaps & Risks Acknowledged

| Gap | Status | Mitigation |
|-----|--------|------------|
| No test for `pre-edit-guard.py` | Known, not blocking | Consistent with existing hook pattern; script is simple (34 lines). Manual verification after wiring. |
| claude-mem 251MB database not cleaned up | Deferred | Disabling the plugin stops hooks but leaves `~/.claude-mem/` on disk. User can delete manually if desired. No urgency — it's local storage. |
| `pinecone-assistant.md` scoping could miss queries | Mitigated | Converting to skill (not path-scoping) ensures semantic triggering on knowledge queries regardless of which files are open. |
| Codex file updates after skills migration | Required | 6 Codex files need path updates. These are documentation-only — no Claude Code production impact. Can be done in same commit as each batch. |
| Skills migration testing | Required | Per-batch incremental migration with test-before-delete ensures nothing breaks. Skills coexist with commands during transition. |
| MEMORY.md at 200-line cliff | Known, deferred | Content past line 200 silently drops. Current file is right at the limit. Phase 2B deduplication may help. Full MEMORY.md audit deferred to separate task. |

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Startup context (tokens, est.) | ~30,000-35,000 | ~22,000-25,000 |
| Always-loaded rules | 8 unscoped | 4 unscoped + 4 conditional/skill |
| Plugins | 15 | 11 |
| Dead systems in config | 3 (NotebookLM MCP, claude-mem hooks, vanilla superpowers overlap) | 0 |
| Skills with frontmatter | 0 of 21 | 21 of 21 |
| Unwired hook scripts | 1 | 0 |
| Stale permissions | ~33 | 0 |
| Dead hooks firing per tool call | ~4 (claude-mem) | 0 |

---

## Implementation Order

1. **Phase 1** (free wins) — 1A+1B+1C+1D in one commit. Verify session starts clean.
2. **Phase 2A** (scope/convert 4 rules) — one commit. Verify routing scenarios.
3. **Phase 2B** (deduplicate) — one commit. Verify drift checks pass.
4. **Phase 3 Batch 1-5** (skills migration) — one commit per batch. Test each skill.
5. **Phase 4** (standards doc) — one commit.

**Testing after each phase:**
- [ ] `claude` starts without errors
- [ ] `/trade-book` works (most-used skill)
- [ ] `/regime-check` works
- [ ] Hooks fire correctly (edit a pipeline file → drift check runs)
- [ ] Memory recall works ("what do we know about compressed spring")
- [ ] Knowledge routing works ("what did we find about break quality" → Pinecone)
- [ ] PDF routing works ("where is the BH FDR paper" → resources/)

---

## What We're NOT Changing

- Auto-memory system (41 curated topic files — working well)
- Pinecone knowledge base (51 files across 5 tiers — working well)
- gold-db MCP server (core data access — working well)
- Subagent definitions (8 agents, already well-structured with model/tool restrictions)
- superpowers-extended-cc plugin (CC-native fork — actively used for brainstorming, code review, planning)
- CLAUDE.md core content (architectural principles, guardrails, behavioral rules)
- Hook scripts (post-edit-pipeline.py, post-edit-schema.py — proven and working)
- greptile, serena, context7, firecrawl, commit-commands, code-simplifier, pr-review-toolkit, feature-dev, security-guidance, explanatory-output-style, ralph-loop plugins (all actively used)
