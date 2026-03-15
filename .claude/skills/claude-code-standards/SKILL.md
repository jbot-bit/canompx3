---
name: claude-code-standards
description: >
  Official Anthropic Claude Code best practices reference. Use when deciding
  where to put new instructions, creating skills, designing subagents, or
  auditing project config against official standards.
disable-model-invocation: true
---
# Claude Code Official Standards Reference

Distilled from [code.claude.com](https://code.claude.com/docs) (fetched 2026-03-15).

## Feature Taxonomy — Where Does This Belong?

| Feature | Loads | Context Cost | Best For |
|---------|-------|-------------|----------|
| **CLAUDE.md** | Every session, full content | Every request | "Always do X" rules, build commands, project conventions |
| **`.claude/rules/`** (unscoped) | Every session | Every request | Topic-specific always-on rules |
| **`.claude/rules/`** (with `paths:`) | When matching files opened | Conditional | Language/directory-specific guidelines |
| **Skills** (`.claude/skills/`) | Descriptions at start, full content on invoke | Low until used | Reference material, invocable workflows |
| **MCP** | Tool schemas at start | Every request | External service connections |
| **Subagents** (`.claude/agents/`) | When spawned | Isolated | Context isolation, parallel work, specialized workers |
| **Hooks** | On trigger | Zero | Deterministic automation (lint, test, guard) |

Source: [Extend Claude Code](https://code.claude.com/docs/en/features-overview)

## Decision Framework

**Put it in CLAUDE.md** if Claude should always know it and removing it would cause mistakes.

**Put it in a rule** if it's always-on but topic-specific. Add `paths:` if it only matters for certain files.

**Put it in a skill** if it's reference material needed sometimes, or a workflow invoked with `/<name>`.

**Put it in a hook** if it MUST happen every time with zero exceptions (deterministic, not advisory).

**Put it in MCP** if it connects to an external service or database.

**Put it in a subagent** if it needs isolated context, specific tools/model, or parallel execution.

Source: [CLAUDE.md vs Rules vs Skills](https://code.claude.com/docs/en/features-overview#compare-similar-features)

## Context Budget

- **CLAUDE.md**: Target <200 lines. For each line ask: "Would removing this cause mistakes?" If not, cut it. [Source](https://code.claude.com/docs/en/memory#write-effective-instructions)
- **Rules**: Use `paths:` frontmatter to make conditional. Saves context for sessions that don't touch those files. [Source](https://code.claude.com/docs/en/memory#path-specific-rules)
- **Skills**: Descriptions budget is ~2% of context window (16K chars fallback). Use `disable-model-invocation: true` for zero cost until manually invoked. [Source](https://code.claude.com/docs/en/skills#troubleshooting)
- **MEMORY.md**: First 200 lines loaded at startup. Content past line 200 is silently dropped. [Source](https://code.claude.com/docs/en/memory#how-it-works)
- **If Claude ignores a rule**: The file is probably too long and the rule is getting lost. [Source](https://code.claude.com/docs/en/best-practices#avoid-common-failure-patterns)

## Skill Authoring

**Frontmatter fields** (all optional except `description`):

| Field | Purpose |
|-------|---------|
| `name` | Display name, becomes `/<name>` command |
| `description` | When to use — Claude matches tasks against this |
| `disable-model-invocation` | `true` = user-only, zero context cost. Use for side-effects |
| `allowed-tools` | Restrict tools (e.g., `Read, Grep, Glob, Bash` for read-only) |
| `context` | `fork` = run in isolated subagent context |
| `agent` | Which subagent type when `context: fork` |
| `user-invocable` | `false` = hidden from `/` menu, Claude-only |
| `model` | Model override for this skill |

Source: [Skills frontmatter reference](https://code.claude.com/docs/en/skills#frontmatter-reference)

**Skill types:**
- **Reference** (API conventions, style guides) → default, model-invocable
- **Workflows** (deploy, rebuild, audit) → `disable-model-invocation: true`
- **Read-only queries** (trade-book, regime-check) → `allowed-tools: Read, Grep, Glob, Bash`
- **Isolated analysis** (blast-radius) → `context: fork`, `agent: <name>`

## Subagent Patterns

| Model | When to Use |
|-------|-------------|
| `haiku` | Fast lookups, data retrieval, simple checks |
| `sonnet` | Code analysis, reviews, moderate complexity |
| `opus` | Complex reasoning, architecture decisions |
| `inherit` | Same as parent conversation (default) |

**Key fields:** `tools` (allowlist), `disallowedTools` (denylist), `model`, `memory` (user/project/local), `maxTurns`, `skills` (preload), `mcpServers` (scoped connections).

Source: [Subagents docs](https://code.claude.com/docs/en/sub-agents)

## Hook Patterns

- **PreToolUse** → Guard (block dangerous edits, validate commands)
- **PostToolUse** → Validate (run drift checks, targeted tests after edits)
- **Stop** → Quality gate (verify all tasks complete before finishing)
- **SessionStart** → Context injection (load env vars, recent git context)

Exit code 2 = block action. Exit code 0 = allow. Zero context cost.

Source: [Hooks reference](https://code.claude.com/docs/en/hooks)

## What NOT to Put in CLAUDE.md

| Exclude | Why |
|---------|-----|
| Things Claude can figure out by reading code | Wastes context |
| Standard language conventions Claude already knows | Redundant |
| Detailed API documentation | Link to docs instead |
| Information that changes frequently | Goes stale, use skills or MCP |
| File-by-file descriptions of the codebase | Claude can explore |
| Long explanations or tutorials | Not actionable instructions |

Source: [Write effective instructions](https://code.claude.com/docs/en/memory#write-effective-instructions)

## Anti-Patterns

| Pattern | Fix |
|---------|-----|
| Kitchen sink session (unrelated tasks in one context) | `/clear` between tasks |
| Correcting over and over (context polluted with failures) | After 2 fails, `/clear` and rewrite prompt |
| Over-specified CLAUDE.md (rules get lost in noise) | Prune ruthlessly, move to skills |
| Trust-then-verify gap (no test, no proof) | Always provide verification |
| Infinite exploration (reads hundreds of files) | Scope narrowly or use subagents |

Source: [Best practices](https://code.claude.com/docs/en/best-practices#avoid-common-failure-patterns)
