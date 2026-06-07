# Codex Agent Routing

Codex uses Claude's agent prompts as canonical specialist recipes. This file is
the Codex-side index so sessions do not have to rediscover the mapping.

## Canonical Claude Agents

- `.claude/agents/blast-radius.md` - change impact and ownership boundaries
- `.claude/agents/db-analyst.md` - DB-backed investigation and schema/data checks
- `.claude/agents/evidence-auditor.md` - claim scrutiny and evidence grading
- `.claude/agents/executor.md` - scoped implementation after plan/design gates
- `.claude/agents/live-risk-auditor.md` - broker, runtime, account, webhook, kill/flatten, and order-routing risk
- `.claude/agents/preflight-auditor.md` - launch/preflight readiness checks
- `.claude/agents/ralph-loop.md` - Ralph autonomous audit loop context
- `.claude/agents/research-methodologist.md` - research, OOS, DSR, FDR, MinBTL, and validation discipline
- `.claude/agents/verify-complete.md` - done-definition and verification closure

## Codex Equivalents

- Use `mcp__repo_state__` for startup packets, pulse, routes, and strict-truth views.
- Use `mcp__gold_db__` for read-only canonical DB facts.
- Use `mcp__research_catalog__` for literature, prereg, and audit-result grounding.
- Use `mcp__strategy_lab__` for strategy readiness and lane-allocation facts.
- Use `.codex/agents/canompx3_explorer.toml` for read-only context scouting.
- Use `.codex/agents/canompx3_reviewer.toml` for second-pass review.
- Use `.codex/agents/canompx3_worker.toml` only after scope, ownership, and verification gates are clear.

## Routing Rule

When Claude has a specialist prompt and Codex has no native equivalent, read the
Claude agent prompt as source material and adapt the workflow into the current
Codex tool surface. Do not copy decisions into Codex-private memory as shared
truth; durable cross-tool decisions go in `HANDOFF.md` or `docs/plans/`.

