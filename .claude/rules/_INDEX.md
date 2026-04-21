---
paths:
  - ".claude/rules/**"
---
# Rules Index

Master index of `.claude/rules/`. Every rule here is auto-loaded by Claude Code
when a matching path in its frontmatter is touched. Two rules are unconditionally
global (`auto-skill-routing`, `workflow-preferences`).

Use this file to locate the rule you need without grepping 19 files.

---

## Integrity & Discipline (what-not-to-do)

- [`integrity-guardian.md`](integrity-guardian.md) — Seven non-negotiable rules (authority hierarchy, canonical sources, fail-closed, impact awareness, evidence-over-assertion, spec compliance, never-trust-metadata). Enforced by `scripts/tools/audit_behavioral.py`.
- [`institutional-rigor.md`](institutional-rigor.md) — Working-style hard rules: self-review, canonical delegation, no dead code, no silent failures, extract-before-dismiss, verify-before-claim. Supersedes "just ship it".
- [`quant-agent-identity.md`](quant-agent-identity.md) — Seven Sins of Quant Investing awareness table. Scan-time reminder during research/strategy code review.

## Research Methodology & Audits

- [`research-truth-protocol.md`](research-truth-protocol.md) — Canonical vs derived layers, validated-universe rule, Mode A holdout discipline, Phase 0 literature grounding, canonical filter delegation.
- [`backtesting-methodology.md`](backtesting-methodology.md) — 14 mandatory rules for any backtest / discovery scan. Load-scoped to research/, strategy_*, audit docs, institutional docs.
- [`backtesting-methodology-failure-log.md`](backtesting-methodology-failure-log.md) — Companion failure log. Append-only history of methodology violations. Self-scoped.
- [`quant-audit-protocol.md`](quant-audit-protocol.md) — Audit procedure for research claims. Used by `/code-review` and audit skills.
- [`quant-audit-failure-patterns.md`](quant-audit-failure-patterns.md) — Known failure patterns found in past audits. Append-only.

## Workflow & Stage Management

- [`auto-skill-routing.md`](auto-skill-routing.md) — **Global.** Intent → skill mapping (commit → just execute, "broken" → /quant-debug, etc.). Companion to `data-first-guard.py` hook.
- [`workflow-preferences.md`](workflow-preferences.md) — **Global.** User preferences: data-first, implementation gating, git ops, response style, no performative self-correction.
- [`stage-gate-protocol.md`](stage-gate-protocol.md) — Stage file structure (`docs/runtime/stages/<slug>.md`). Hook-enforced by `.claude/hooks/stage-gate-guard.py`.
- [`validation-workflow.md`](validation-workflow.md) — Validator CLI flags + full rebuild chain for outcome_builder changes.
- [`m25-audit.md`](m25-audit.md) — M25 pre-commit audit scan. Catches unverified suggestions in production code.

## Domain-Specific

- [`daily-features-joins.md`](daily-features-joins.md) — The triple-join trap (`trading_day` + `symbol` + `orb_minutes`). CTE guard. LAG() queries must filter `orb_minutes = 5`.
- [`pipeline-patterns.md`](pipeline-patterns.md) — DB write pattern (DELETE-then-INSERT), time/calendar model (Brisbane 9am boundary).
- [`strategy-awareness.md`](strategy-awareness.md) — Blueprint routing (§3 test sequence, §5 NO-GO, §10 assumptions). Variable coverage rule.
- [`mcp-usage.md`](mcp-usage.md) — gold-db MCP tool selection decision framework. Always prefer MCP over raw SQL.

## Infrastructure & Safety

- [`branch-discipline.md`](branch-discipline.md) — Never branch a PR from local `main` without verifying `origin/main`. Pre-PR diff-scope verification.
- [`large-file-reads.md`](large-file-reads.md) — Files >1800 lines: wc -l first, then offset+limit. Use Grep to find the section.

---

## Related-rules table (cross-references)

When you hit one of these, check the adjacent ones:

| When editing | Also consult |
|---|---|
| Any `research/**` script | `research-truth-protocol.md`, `backtesting-methodology.md`, `strategy-awareness.md` |
| `trading_app/strategy_*` | `validation-workflow.md`, `research-truth-protocol.md`, `strategy-awareness.md`, `integrity-guardian.md` |
| `trading_app/config.py` or filters | `integrity-guardian.md` (§canonical delegation), `daily-features-joins.md` |
| `pipeline/check_drift.py` | `integrity-guardian.md`, `large-file-reads.md` |
| `.claude/rules/*.md` | This file (`_INDEX.md`) — update when adding/removing rules |
| PR files / `HANDOFF.md` / `docs/plans/` | `branch-discipline.md` |
| `trading_app/mcp_server.py` or reports | `mcp-usage.md` |

---

## Load policy

Rules use YAML frontmatter `paths:` to declare when they auto-load. Example:

```yaml
---
paths:
  - "trading_app/strategy_*"
  - "research/**"
---
```

The above rule only loads into Claude's context when a matching file is edited.
The two exceptions without `paths:` (`auto-skill-routing.md`, `workflow-preferences.md`)
load on every CLAUDE.md injection — use sparingly; keep them short.

To convert a file to on-demand only: add `paths: - ".claude/rules/<filename>"` so
it only self-injects on its own edit (pattern used by `quant-audit-failure-patterns.md`).
