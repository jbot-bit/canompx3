# Claude Code Standards Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Align Claude Code config with official Anthropic best practices — remove dead weight, scope rules, migrate commands to skills with proper frontmatter.

**Architecture:** Config-only changes across `.claude/settings.json`, `.claude/settings.local.json`, `.mcp.json`, `.claude/rules/`, and `.claude/commands/` → `.claude/skills/`. No production Python code changes. Incremental migration with verification after each phase.

**Tech Stack:** Claude Code config (JSON, YAML frontmatter, Markdown)

**Design doc:** `docs/plans/2026-03-15-claude-code-standards-upgrade-design.md`

---

## Phase 1: Free Wins (Zero Risk)

### Task 0: Disable 4 dead/duplicate plugins

**Files:**
- Modify: `.claude/settings.json:48-64`

**Step 1: Edit settings.json enabledPlugins**

Set these 4 entries to `false`:

```json
"superpowers@claude-plugins-official": false,
"coderabbit@claude-plugins-official": false,
"claude-mem@claude-plugins-official": false,
"frontend-design@claude-plugins-official": false
```

Keep all other plugins as `true`.

**Step 2: Verify the edit**

Read `.claude/settings.json` and confirm exactly 11 plugins remain `true`:
context7, code-simplifier, commit-commands, security-guidance, pr-review-toolkit, serena, feature-dev, explanatory-output-style, greptile, firecrawl, ralph-loop.

---

### Task 1: Remove NotebookLM from .mcp.json

**Files:**
- Modify: `.mcp.json`

**Step 1: Edit .mcp.json**

Remove the entire `notebooklm` server entry. The file should contain only:

```json
{
  "mcpServers": {
    "gold-db": {
      "type": "stdio",
      "command": "python",
      "args": [
        "trading_app/mcp_server.py"
      ],
      "env": {}
    }
  }
}
```

**Step 2: Verify**

Read `.mcp.json` and confirm only `gold-db` remains.

---

### Task 2: Wire pre-edit-guard.py as PreToolUse hook

**Files:**
- Modify: `.claude/settings.json:24-46` (hooks section)

**Step 1: Add PreToolUse hook**

Add a `PreToolUse` key to the `hooks` object, BEFORE the existing `PostToolUse`:

```json
"hooks": {
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
    ],
    "PostToolUse": [
```

**Step 2: Verify the hook script exists**

Run: `cat .claude/hooks/pre-edit-guard.py | head -5`
Expected: `#!/usr/bin/env python3` and `"""Pre-edit guard: block direct edits to gold.db`

---

### Task 3: Clean permissions allow-list

**Files:**
- Modify: `.claude/settings.local.json:3-157` (permissions.allow array)

**Step 1: Remove dead entries**

Remove these entries from the `permissions.allow` array:

**13 notebooklm entries:**
- `"mcp__notebooklm__notebook_list"`
- `"mcp__notebooklm__notebook_get"`
- `"mcp__notebooklm__notebook_query"`
- `"mcp__notebooklm__notebook_add_url"`
- `"mcp__notebooklm__notebook_create"`
- `"mcp__notebooklm__notebook_add_text"`
- `"mcp__notebooklm__source_describe"`
- `"mcp__notebooklm__refresh_auth"`
- `"mcp__notebooklm__healthcheck"`
- `"mcp__notebooklm__chat_with_notebook"`
- `"mcp__notebooklm__navigate_to_notebook"`
- `"mcp__notebooklm__send_chat_message"`
- `"mcp__notebooklm__get_quick_response"`

**4 claude-mem entries:**
- `"mcp__plugin_claude-mem_mcp-search__search"`
- `"mcp__plugin_claude-mem_mcp-search__save_memory"`
- `"mcp__plugin_claude-mem_mcp-search__get_observations"`
- `"mcp__plugin_claude-mem_mcp-search____IMPORTANT"`

**claude-mem dashboard:**
- `"WebFetch(domain:127.0.0.1)"`

**20 stale Bash one-offs** (lines 120-139 — full-path commands from old sessions):
Remove every `"Bash(...)"` entry. These are one-off approvals that accumulated.

**1 stale Read entry:**
- `"Read(//home/joshd/.claude/projects/**)"`

**Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('.claude/settings.local.json')); print('Valid JSON')"`
Expected: `Valid JSON`

**Step 3: Count remaining entries**

Run: `python -c "import json; d=json.load(open('.claude/settings.local.json')); print(len(d['permissions']['allow']), 'entries remaining')"`
Expected: ~113 entries (down from 153)

---

### Task 4: Verify Phase 1

**Step 1: Verify all files are valid JSON**

Run: `python -c "import json; [json.load(open(f)) for f in ['.claude/settings.json', '.claude/settings.local.json', '.mcp.json']]; print('All valid')"`
Expected: `All valid`

**Step 2: Commit Phase 1**

```bash
git add .claude/settings.json .claude/settings.local.json .mcp.json
git commit -m "chore: remove dead plugins, wire pre-edit guard, clean permissions

- Disable 4 plugins: claude-mem (dead since Feb 28), superpowers (duplicate
  of extended-cc), coderabbit (duplicate of greptile), frontend-design
  (not relevant)
- Remove retired notebooklm MCP server from .mcp.json
- Wire pre-edit-guard.py as PreToolUse hook (was unwired)
- Remove 39 stale permission entries (notebooklm, claude-mem, Bash one-offs)"
```

**Step 3: Smoke test (requires new session)**

Note for executor: after committing, start a new Claude Code session and verify:
- Session starts without errors
- `/trade-book` works
- Hooks fire on pipeline file edits

---

## Phase 2: Context Diet (Low Risk)

### Task 5: Path-scope notebooklm.md and quant-agent-identity.md

**Files:**
- Modify: `.claude/rules/notebooklm.md:1`
- Modify: `.claude/rules/quant-agent-identity.md:1`

**Step 1: Add frontmatter to notebooklm.md**

Prepend to `.claude/rules/notebooklm.md`:

```yaml
---
paths:
  - "resources/**"
---
```

Before the existing `# Academic Methodology — Local Resources` heading.

**Step 2: Add frontmatter to quant-agent-identity.md**

Prepend to `.claude/rules/quant-agent-identity.md`:

```yaml
---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "research/**"
  - "scripts/**"
---
```

Before the existing `# Quant Agent Identity — Bias Defense` heading.

**Step 3: Verify frontmatter syntax**

Run: `head -6 .claude/rules/notebooklm.md .claude/rules/quant-agent-identity.md`
Expected: Both show `---` / `paths:` / path entries / `---` then content.

---

### Task 6: Convert pinecone-assistant.md to skill

**Files:**
- Create: `.claude/skills/pinecone-assistant/SKILL.md`
- Delete: `.claude/rules/pinecone-assistant.md`

**Step 1: Create skill directory**

Run: `mkdir -p .claude/skills/pinecone-assistant`

**Step 2: Create SKILL.md**

Write `.claude/skills/pinecone-assistant/SKILL.md` with this frontmatter prepended to the existing content of `.claude/rules/pinecone-assistant.md`:

```yaml
---
name: pinecone-assistant
description: >
  Pinecone knowledge base routing for project history and research findings.
  Use when asked: "what did we find about X", "why did we do Y", "research on",
  "history of", "remind me", "what's the story", "NO-GOs", design decisions.
  Routes between Pinecone (project knowledge) and gold-db MCP (live data).
---
```

Followed by the full content of the existing `.claude/rules/pinecone-assistant.md`.

**Step 3: Delete the old rule**

Run: `rm .claude/rules/pinecone-assistant.md`

**Step 4: Verify skill exists**

Run: `cat .claude/skills/pinecone-assistant/SKILL.md | head -10`
Expected: YAML frontmatter with name, description, then `# Pinecone Assistant (orb-research)`

---

### Task 7: Convert audit-prompts.md to skill

**Files:**
- Create: `.claude/skills/audit-prompts/SKILL.md`
- Delete: `.claude/rules/audit-prompts.md`

**Step 1: Create skill directory**

Run: `mkdir -p .claude/skills/audit-prompts`

**Step 2: Create SKILL.md**

Write `.claude/skills/audit-prompts/SKILL.md` with this frontmatter prepended:

```yaml
---
name: audit-prompts
description: >
  Spec compliance and guardian prompt routing. Use before significant changes
  to production logic, schema, entry models, or pipeline data flow. Use when
  asked about health checks, system audits, guardian prompts, or "is everything
  in sync". Also triggers on spec checking before building features.
---
```

Followed by the full content of the existing `.claude/rules/audit-prompts.md`.

**Step 3: Delete the old rule**

Run: `rm .claude/rules/audit-prompts.md`

**Step 4: Verify**

Run: `cat .claude/skills/audit-prompts/SKILL.md | head -10`

---

### Task 8: Deduplicate CLAUDE.md and rules

**Files:**
- Modify: `CLAUDE.md:18-39`
- Modify: `.claude/rules/pipeline-patterns.md`
- Modify: `.claude/rules/validation-workflow.md`

**Step 1: Trim CLAUDE.md Document Authority table**

Replace lines 16-39 (the Document Authority section) with:

```markdown
## Document Authority

| Document | Scope | Conflict Rule |
|----------|-------|---------------|
| `CLAUDE.md` | Code structure, commands, guardrails, AI behavior | Wins for code decisions |
| `TRADING_RULES.md` | Trading rules, sessions, filters, research findings, NO-GOs | Wins for trading logic |
| `RESEARCH_RULES.md` | Research methodology, statistical standards, trading lens | Wins for research/analysis decisions |
| `ROADMAP.md` | Planned features, phase status | Updated on phase completion |
| `docs/specs/*.md` | Feature specs pending implementation | **Check before building ANY feature** |

Full file inventory → `REPO_MAP.md` (auto-generated, never hand-edit).
Frozen specs (`CANONICAL_*.txt`) → read-only; live code is truth.

**Conflict resolution:** Code behavior → CLAUDE.md. Trading logic → TRADING_RULES.md. Research → RESEARCH_RULES.md.
```

**Step 2: Trim pipeline-patterns.md**

Remove the "Core Principles" section (lines 7-9) — duplicated from CLAUDE.md. Remove "DST — Fully Resolved" section — duplicated from CLAUDE.md. Keep only the unique content:
- Database Write Pattern
- Time & Calendar

**Step 3: Trim validation-workflow.md**

Remove the "Strategy Classification" table at the end — duplicated from ARCHITECTURE.md. Replace with: `Classification thresholds → see docs/ARCHITECTURE.md.`

**Step 4: Verify drift checks pass**

Run: `python pipeline/check_drift.py`
Expected: All checks pass (count self-reported at runtime)

---

### Task 9: Verify Phase 2

**Step 1: Verify all changes**

Run: `git diff --stat`
Expected: Changes to rules, skills created, CLAUDE.md trimmed.

**Step 2: Commit Phase 2**

```bash
git add .claude/rules/ .claude/skills/ CLAUDE.md
git commit -m "chore: scope rules, convert routing to skills, deduplicate docs

- Path-scope notebooklm.md (resources/**) and quant-agent-identity.md
  (pipeline/trading_app/research/scripts)
- Convert pinecone-assistant.md to skill (semantic trigger on knowledge queries)
- Convert audit-prompts.md to skill (semantic trigger on audit/guardian queries)
- Trim CLAUDE.md document authority table (13→5 entries)
- Remove duplicated content from pipeline-patterns.md and validation-workflow.md"
```

---

## Phase 3: Skills Migration (5 Batches)

### Migration Pattern (applies to all batches)

For each command being migrated:

1. `mkdir -p .claude/skills/<name>/`
2. Create `SKILL.md` = YAML frontmatter + existing `.claude/commands/<name>.md` content
3. Invoke `/<name>` with a test argument — confirm it loads
4. `rm .claude/commands/<name>.md`
5. Re-invoke `/<name>` — confirm skill still works without command fallback

**Frontmatter template:**
```yaml
---
name: <name>
description: <first line of existing command file, which is the trigger description>
<additional fields per batch>
---
```

The `description` field comes from the first line of each command file (e.g., "Show current trading book with full strategy details: $ARGUMENTS"). Remove the `$ARGUMENTS` placeholder from the description — it goes in the body via the standard `$ARGUMENTS` substitution.

---

### Task 10: Skills Batch 1 — Read-only queries (trade-book, regime-check, discover)

**Files:**
- Create: `.claude/skills/trade-book/SKILL.md`
- Create: `.claude/skills/regime-check/SKILL.md`
- Create: `.claude/skills/discover/SKILL.md`
- Delete: `.claude/commands/trade-book.md`
- Delete: `.claude/commands/regime-check.md`
- Delete: `.claude/commands/discover.md`

**Frontmatter for all three:**
```yaml
---
name: <name>
description: <first line from command file>
allowed-tools: Read, Grep, Glob, Bash
---
```

**Step 1: Create trade-book skill**

```bash
mkdir -p .claude/skills/trade-book
```

Write `.claude/skills/trade-book/SKILL.md`:
- Frontmatter with `name: trade-book`, `description: Show current trading book with full strategy details.`, `allowed-tools: Read, Grep, Glob, Bash`
- Full content from `.claude/commands/trade-book.md` (including the "Use when:" line and all steps)

**Step 2: Create regime-check skill**

Same pattern. `name: regime-check`, description from first line of command.

**Step 3: Create discover skill**

Same pattern. `name: discover`, description from first line of command.

**Step 4: Test all three**

Invoke `/trade-book`, `/regime-check`, `/discover` — each should load and execute.

**Step 5: Delete old commands**

```bash
rm .claude/commands/trade-book.md .claude/commands/regime-check.md .claude/commands/discover.md
```

**Step 6: Re-test and commit**

```bash
git add .claude/skills/ .claude/commands/
git commit -m "feat: migrate trade-book, regime-check, discover to skills with allowed-tools"
```

---

### Task 11: Skills Batch 2 — Design/planning (4t, 4tp, bloomey-review)

**Files:**
- Create: `.claude/skills/4t/SKILL.md`, `.claude/skills/4tp/SKILL.md`, `.claude/skills/bloomey-review/SKILL.md`
- Delete: `.claude/commands/4t.md`, `.claude/commands/4tp.md`, `.claude/commands/bloomey-review.md`

**Frontmatter:** Default (model-invocable, all tools):
```yaml
---
name: <name>
description: <first line + "Use when:" triggers>
---
```

Same create/test/delete/re-test/commit pattern.

```bash
git commit -m "feat: migrate 4t, 4tp, bloomey-review to skills"
```

---

### Task 12: Skills Batch 3 — Debug/verify (6 commands)

**Files:**
- Create skills for: `quant-debug`, `quant-tdd`, `quant-verify`, `verify-complete`, `integrity-guardian`, `health-check`
- Delete corresponding commands

**Frontmatter:** Default (model-invocable, all tools).

Same pattern. Commit:
```bash
git commit -m "feat: migrate debug/verify commands to skills"
```

---

### Task 13: Skills Batch 4 — Side-effect + audit (7 commands)

**Files:**
- Create skills for: `rebuild-outcomes`, `validate-instrument`, `post-rebuild`, `audit`, `audit-quick`, `audit-phase`, `m25-audit`
- Delete corresponding commands

**Frontmatter — all with invocation control:**
```yaml
---
name: <name>
description: <first line>
disable-model-invocation: true
---
```

Same pattern. Commit:
```bash
git commit -m "feat: migrate side-effect/audit commands to skills with disable-model-invocation"
```

---

### Task 14: Skills Batch 5 — Special skills (blast-radius, ralph)

**Files:**
- Create: `.claude/skills/blast-radius/SKILL.md`, `.claude/skills/ralph/SKILL.md`
- Delete: `.claude/commands/blast-radius.md`, `.claude/commands/ralph.md`

**blast-radius frontmatter:**
```yaml
---
name: blast-radius
description: >
  Pre-edit impact analysis. Map all callers, importers, downstream effects,
  companion tests, and canonical source dependencies before modifying production code.
context: fork
agent: blast-radius
---
```

**ralph frontmatter:**
```yaml
---
name: ralph
description: >
  Run one Ralph Loop audit iteration. Finds Seven Sins violations, canonical
  integrity issues, and silent failures. Fixes highest-priority finding.
disable-model-invocation: true
---
```

Same create/test/delete/re-test pattern. Commit:
```bash
git commit -m "feat: migrate blast-radius (forked context) and ralph (manual-only) to skills"
```

---

### Task 15: Verify Phase 3 + update Codex refs

**Step 1: Confirm commands dir is empty**

Run: `ls .claude/commands/`
Expected: No .md files remaining.

**Step 2: Remove empty commands directory**

Run: `rmdir .claude/commands/ 2>/dev/null || echo "Dir not empty — check remaining files"`

**Step 3: Update Codex references**

Update these files to replace `.claude/commands/` paths with `.claude/skills/<name>/SKILL.md`:

1. `.codex/COMMANDS.md` — update all 11 command path references
2. `.codex/WORKFLOWS.md` line 37 — `.claude/commands/4t.md` → `.claude/skills/4t/SKILL.md`
3. `.codex/skills/canompx3-audit/SKILL.md` — 4 path updates
4. `.codex/skills/canompx3-workspace/SKILL.md` — update generic reference
5. `.codex/skills/canompx3-verify/SKILL.md` — 2 path updates
6. `CODEX.md` lines 50, 102 — 2 reference updates

**Step 4: Smoke test key skills**

Invoke each of these and confirm they load:
- `/trade-book` (read-only)
- `/4t` (design)
- `/quant-debug` (debug)
- `/rebuild-outcomes` (side-effect — should only invoke manually)
- `/blast-radius` (forked context)

**Step 5: Commit**

```bash
git add .claude/ .codex/ CODEX.md
git commit -m "chore: remove legacy commands dir, update Codex references to skills"
```

---

## Phase 4: Standards Reference

### Task 16: Create standards reference skill

**Files:**
- Create: `.claude/skills/claude-code-standards/SKILL.md`

**Step 1: Create skill directory**

Run: `mkdir -p .claude/skills/claude-code-standards`

**Step 2: Write SKILL.md**

Write `.claude/skills/claude-code-standards/SKILL.md` with content covering:

```yaml
---
name: claude-code-standards
description: >
  Official Anthropic Claude Code best practices reference. Use when deciding
  where to put new instructions, creating skills, designing subagents, or
  auditing project config against official standards.
disable-model-invocation: true
---
```

Body content (~100-150 lines):
- Feature taxonomy: CLAUDE.md vs rules vs skills vs MCP vs hooks vs subagents
- Decision framework: "where does this instruction belong?"
- Context budget: target <200 lines per CLAUDE.md, use paths: frontmatter, skill descriptions budget is 2% of context
- Skill authoring: frontmatter fields reference (name, description, disable-model-invocation, allowed-tools, context, agent)
- Subagent patterns: model selection (haiku for fast, sonnet for analysis, opus for complex), tool restrictions, memory scopes
- Hook patterns: PreToolUse guards, PostToolUse validation, zero context cost
- What NOT to put in CLAUDE.md: things Claude can infer, detailed API docs, frequently changing info, file-by-file descriptions
- Anti-patterns: kitchen sink session, over-specified CLAUDE.md, trust-then-verify gap, infinite exploration
- Source links to code.claude.com for each section

**Step 3: Commit**

```bash
git add .claude/skills/claude-code-standards/
git commit -m "feat: add official Claude Code standards reference skill"
```

---

## Verification Checklist (Final)

After all phases complete, start a fresh session and verify:

- [ ] Session starts without errors
- [ ] `/trade-book` works (skill with allowed-tools)
- [ ] `/regime-check` works (skill with allowed-tools)
- [ ] `/4t test-feature` works (model-invocable design skill)
- [ ] `/blast-radius outcome_builder.py` works (forked context skill)
- [ ] `/rebuild-outcomes MGC` loads only when manually invoked (disable-model-invocation)
- [ ] Hooks fire on pipeline file edit (drift check + targeted tests)
- [ ] Pre-edit guard blocks Write to gold.db
- [ ] Memory recall works ("what do we know about compressed spring")
- [ ] Knowledge routing works ("what did we find about break quality" → Pinecone skill fires)
- [ ] `/memory` shows fewer always-loaded rules vs before
- [ ] `python pipeline/check_drift.py` passes
- [ ] No vanilla superpowers skill duplicates visible
