---
name: canompx3-claude-parity
description: Use when Codex must cover a canompx3 workflow that exists in Claude commands, hooks, agents, rules, or skills but has no dedicated Codex-native wrapper.
---

# Canompx3 Claude Parity

Use this skill to fill Codex capability gaps by copying from Claude-owned
source material into Codex-owned execution, without mutating Claude files.

## Boundaries

- `CLAUDE.md` and `.claude/` are canonical source material.
- Codex-owned surfaces are `.codex/`, `.agents/skills/`, and `scripts/infra/codex_*`.
- Do not edit `CLAUDE.md`, `.claude/`, `claude.bat`, or Claude-owned settings/hooks unless the user explicitly asks.
- Prefer references and thin wrappers over duplicating full Claude docs. If a copied workflow becomes Codex-specific, record the adaptation under `.codex/`.

## Fast Route

1. Identify the missing workflow class:
   - command recipe -> `.claude/commands/`
   - specialist prompt -> `.claude/agents/`
   - project rule -> `.claude/rules/`
   - reusable workflow -> `.claude/skills/`
   - safety hook behavior -> `.claude/hooks/` plus tests
2. Read only the smallest relevant Claude file(s).
3. Execute through Codex tools and MCPs.
4. If the gap will recur, add or update a Codex-owned index, wrapper, hook, or launcher.
5. Verify the Codex-owned artifact and leave Claude files untouched.

## Current Parity Indexes

- Commands: `.codex/COMMANDS.md`
- Agents: `.codex/AGENTS.md`
- Rules: `.codex/RULES.md`
- Hooks: `.codex/HOOKS.md`
- Integrations and MCPs: `.codex/INTEGRATIONS.md`
- Workflows: `.codex/WORKFLOWS.md`
- Repo-local Codex skills: `.codex/skills/README.md`

## Claude Capability Sources

- Commands: `.claude/commands/audit-code.md`, `check.md`, `cherry-pick.md`,
  `crg-*.md`, `discover-edge.md`, `highrisk-review.md`, `nogo.md`,
  `promote-queue.md`, `verify-finding.md`
- Agents: `.claude/agents/blast-radius.md`, `db-analyst.md`,
  `evidence-auditor.md`, `executor.md`, `live-risk-auditor.md`,
  `preflight-auditor.md`, `ralph-loop.md`, `research-methodologist.md`,
  `verify-complete.md`
- Skills: `.claude/skills/audit/SKILL.md`,
  `.claude/skills/blast-radius/SKILL.md`,
  `.claude/skills/brain/SKILL.md`,
  `.claude/skills/capital-review/SKILL.md`,
  `.claude/skills/code-review/SKILL.md`,
  `.claude/skills/design/SKILL.md`,
  `.claude/skills/discover/SKILL.md`,
  `.claude/skills/next/SKILL.md`,
  `.claude/skills/open-pr/SKILL.md`,
  `.claude/skills/orient/SKILL.md`,
  `.claude/skills/pinecone-assistant/SKILL.md`,
  `.claude/skills/post-rebuild/SKILL.md`,
  `.claude/skills/propose-hypothesis/SKILL.md`,
  `.claude/skills/quant-debug/SKILL.md`,
  `.claude/skills/quant-tdd/SKILL.md`,
  `.claude/skills/ralph/SKILL.md`,
  `.claude/skills/rebuild-outcomes/SKILL.md`,
  `.claude/skills/recall/SKILL.md`,
  `.claude/skills/regime-check/SKILL.md`,
  `.claude/skills/research/SKILL.md`,
  `.claude/skills/resume-rebase/SKILL.md`,
  `.claude/skills/ship/SKILL.md`,
  `.claude/skills/skill-improve/SKILL.md`,
  `.claude/skills/stage-gate/SKILL.md`,
  `.claude/skills/task-splitter/SKILL.md`,
  `.claude/skills/trade-book/SKILL.md`,
  `.claude/skills/validate-instrument/SKILL.md`,
  `.claude/skills/verify/SKILL.md`
- Rules: use `.codex/RULES.md` for the full current rule index.
- Hooks: use `.codex/HOOKS.md` to find Claude hook behaviors, then read the
  matching `.claude/hooks/*.py` only to copy behavior into Codex-owned hooks,
  launchers, or doctors. Do not edit Claude hooks.

## MCP Smoke Expectations

Normal Codex project sessions should expose these repo-local read-only MCPs:

- `repo-state`
- `gold-db`
- `research-catalog`
- `strategy-lab`

The shared `.mcp.json` also declares `code-review-graph`; treat it as explicit
structural-review support, not routine project truth. User-level Codex MCPs such
as `openaiDeveloperDocs` are useful only for their own domains.
