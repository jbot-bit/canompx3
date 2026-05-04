---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Plan: CLAUDE.md Optimization & Context Efficiency

**Goal:** Reduce always-loaded token cost, improve instruction adherence, use official best practices.
**Risk level:** HIGH — CLAUDE.md is the single most critical file for Claude's behavior. Bad change = degraded AI across all sessions.

## Current State

- CLAUDE.md: 227 lines (~1,800 words)
- .claude/rules/: 512 lines across 11 files (~3,900 words), ALL loaded every session
- MEMORY.md: 228 lines but truncated at 200 (28 lines silently lost)
- Total always-loaded: ~7,300 words (~10K tokens)

## Official Best Practices (from docs.anthropic.com, saved in docs/reference/)

1. **Target under 200 lines per CLAUDE.md** — we're at 227
2. **`@path/to/file` imports** — expands referenced files into context at launch
3. **Path-specific rules** — YAML frontmatter `paths:` scopes rules to only load when touching matching files
4. **"For each line, ask: Would removing this cause Claude to make mistakes? If not, cut it."**
5. **"Bloated CLAUDE.md files cause Claude to ignore your actual instructions"**
6. **Skills for on-demand knowledge** — "domain knowledge or workflows that are only relevant sometimes, use skills instead"

---

## Phase 1: Verify @import Works (5 min, MUST DO FIRST)

Before changing anything, verify `@path/to/file` actually loads on this setup.

**Test:**
1. Create a test file `docs/test-import.md` with a unique string like `IMPORT_CANARY_12345`
2. Add `@docs/test-import.md` to CLAUDE.md temporarily
3. Start a new Claude Code session
4. Ask Claude "what is the import canary value?" — if it answers correctly, imports work
5. Remove test file and reference

**If imports DON'T work:** Abort Phase 2. Keep CLAUDE.md as-is, focus on Phase 3 only.
**If imports DO work:** Proceed to Phase 2.

---

## Phase 2: CLAUDE.md Slim-Down (20 min)

**What to extract to `docs/ARCHITECTURE.md` (already created):**
- Data flow diagram (lines 58-75) — 17 lines
- Key Commands block (lines 109-146) — 37 lines
- .env Configuration (lines 188-193) — 5 lines
- Price Data Sources table (lines 12-20) — 8 lines

**What to ADD to CLAUDE.md:**
- `@docs/ARCHITECTURE.md` import reference
- Compaction instruction: "When compacting, preserve: modified files list, test commands, current task status, key decisions"
- Reference to workflow-preferences.md

**What MUST STAY in CLAUDE.md (causes mistakes if removed):**
- Project overview (1 line)
- Document authority table (routing — Claude needs this to know WHERE to look)
- Design principles (fail-closed, idempotent, one-way dep)
- Time & calendar model (prevents timezone bugs)
- Database rules (prevents concurrent writes)
- Volatile data rule (prevents stale citations)
- 2-Pass implementation method (prevents rushed changes)
- Strategy classification rules (prevents misdiagnosis of low trade counts)

**Expected result:** CLAUDE.md ~140 lines (from 227). Below 200-line target.

**Risk:** If @import breaks silently in future Claude Code updates, extracted content disappears. Mitigation: keep docs/ARCHITECTURE.md as a standalone readable doc regardless.

---

## Phase 3: Path-Specific Rules (15 min)

Add YAML frontmatter to rules that are SAFELY scoped. Key question for each: "If Claude doesn't see this rule while working on file X, will it make a mistake?"

**SAFE to scope (irrelevant to most sessions):**

| Rule File | Lines | Scope To | Reasoning |
|-----------|-------|----------|-----------|
| `m25-audit.md` | 81 | `paths: scripts/tools/m25*` | Only relevant when running M2.5 audits |
| `notebooklm.md` | 37 | `paths: research/**` | Only relevant for research knowledge queries |
| `pinecone-assistant.md` | 81 | `paths: scripts/tools/sync_pinecone*` | Only relevant when syncing or querying Pinecone |
| `daily-features-joins.md` | 21 | `paths: ["pipeline/**", "trading_app/**", "research/**"]` | Only relevant when writing SQL/data code |
| `validation-workflow.md` | 39 | `paths: trading_app/strategy_validator*` | Only relevant when running validation |

**Savings:** ~259 lines NOT loaded on non-matching sessions.

**MUST stay always-loaded (dangerous to scope):**

| Rule File | Lines | Why Always |
|-----------|-------|-----------|
| `integrity-guardian.md` | 50 | Fail-closed behavior applies to ALL code changes |
| `quant-agent-identity.md` | 36 | Seven sins awareness applies to ALL analysis |
| `audit-prompts.md` | 53 | "Check specs before building" applies to everything |
| `mcp-usage.md` | 45 | MCP routing applies whenever querying data |
| `pipeline-patterns.md` | 21 | Core patterns apply to all pipeline work |
| `workflow-preferences.md` | 48 | User preferences apply to ALL sessions |

**Frontmatter format:**
```yaml
---
paths:
  - "scripts/tools/m25*"
---
# M2.5 Second-Opinion Audit Protocol
...
```

---

## Phase 4: MEMORY.md Pruning (15 min)

**Problem:** MEMORY.md is 228 lines, truncated at 200. Lines 201-228 are silently lost every session.

**Action:**
1. Read lines 200-228 to see what's being lost
2. Move any important content to topic files (which aren't truncated)
3. Trim MEMORY.md to ~150 lines max — remove stale entries, condense tables
4. The "Current State" section says "always query, never cite from memory" — so detailed findings should NOT be in MEMORY.md at all

**Risk:** Low. Topic files are read on-demand and aren't truncated.

---

## Phase 5: Verification

After all changes:
1. Start a fresh Claude Code session
2. Ask "what is in your context?" or test specific rules
3. Verify @import content is present
4. Verify path-specific rules DON'T load on unrelated files
5. Verify path-specific rules DO load on matching files
6. Run `python pipeline/check_drift.py` — all checks pass
7. Run `python -m pytest tests/ -x -q` — all tests pass

---

## NOT Doing

- **Vectoring rules into Pinecone** — behavioral rules must be in context at session start, not retrieved via search
- **Moving MCP routing to a skill** — needed too often, always-loaded is correct
- **Aggressive CLAUDE.md gutting** — the 140-line target is conservative on purpose
