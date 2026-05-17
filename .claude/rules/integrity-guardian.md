---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "scripts/**"
  - "research/**"
---

# Integrity Guardian — Behavioral Rules

Eight non-negotiable rules for every change. Violations are caught by `scripts/tools/audit_behavioral.py`.

## 1. Authority Hierarchy
Defer to the governing document — never restate its content:
- Code structure/guardrails: `CLAUDE.md`
- Trading logic/filters/sessions: `TRADING_RULES.md`
- Research methodology/statistics: `RESEARCH_RULES.md`
- Feature specs: `docs/specs/*.md` (check BEFORE building)

## 2. Canonical Sources — Never Hardcode
→ Canonical-source authority table moved to `institutional-rigor.md` § 10 (single source of truth, 2026-05-17 dedup). Behavioral rule retained here: import from the canonical module, never inline lists / magic numbers / timing constants.

Specific authorities preserved here for grep-ability:
- **`orb_utc_window` authority** — single source for ORB window end UTC across `outcome_builder`, `execution_engine`, `build_daily_features`. Never re-implement; never fall back to `break_ts`; raise `ValueError` instead. See `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.
- **`holdout_policy` authority** — Mode A sacred-window constants; never inline `date(2026, 1, 1)` or `datetime(2026, 4, 8, ...)` anywhere downstream. Declaration-consistency drift check (`check_holdout_policy_declaration_consistency`) enforces docs ↔ code parity. See `docs/plans/archive/2026-04/2026-04-07-holdout-policy-decision.md`.

## 3. Fail-Closed Mindset
- Never report success after an exception or timeout
- Never hardcode check counts (`"all 17 checks"`) — compute dynamically
- Never catch `Exception` and return success in health/audit paths
- Check subprocess return codes — zero is the only success

## 4. Impact Awareness
For every production file change, ask:
- Test file updated? (check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py`)
- Doc references still accurate?
- Drift checks still pass? (`python pipeline/check_drift.py`)
- Spec in `docs/specs/` compliant?

## 5. Evidence Over Assertion — Generation Is Not Validation
Provide verifiable output, not claims. Show command output, row counts, test results.
- **Generation is not validation.** No LLM output is trusted until verified with execution evidence.
- For code review findings: trace the execution path (file:line → call → file:line) before claiming a bug exists. Confident wrong findings are worse than no findings.
- Use semi-formal reasoning: PREMISE → TRACE → EVIDENCE → CONCLUSION. Do not report findings where TRACE or EVIDENCE is empty.

## 6. Spec Compliance
Check `docs/specs/` before building ANY feature. If a spec exists, follow it exactly.

## 7. Never Trust Metadata — Always Verify
→ Full doctrine moved to `institutional-rigor.md` § 11 (single source of truth, 2026-05-17 dedup). Behavioral checklist retained:
- Bundle fields, check labels, docstrings, row-count memory — none are evidence.
- Verify drift checks via known-violation injection.
- **Reading code is not verifying code.** Verifying requires execution + output inspection.

## 8. Research Finding Staleness — Never Inline Stats
Never inline research stats (p-values, N counts) in code. Use `@research-source` + `@revalidated-for` annotations. Full rules → `.claude/rules/research-truth-protocol.md`.
