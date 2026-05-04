---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "scripts/**"
  - "research/**"
---

# Integrity Guardian — Behavioral Rules

Seven non-negotiable rules for every change. Violations are caught by `scripts/tools/audit_behavioral.py`.

## 1. Authority Hierarchy
Defer to the governing document — never restate its content:
- Code structure/guardrails: `CLAUDE.md`
- Trading logic/filters/sessions: `TRADING_RULES.md`
- Research methodology/statistics: `RESEARCH_RULES.md`
- Feature specs: `docs/specs/*.md` (check BEFORE building)

## 2. Canonical Sources — Never Hardcode
Import from the single source of truth. Never inline lists or magic numbers.
| Data | Source |
|------|--------|
| Active instruments | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| All instrument configs | `pipeline.asset_configs.ASSET_CONFIGS` |
| Session catalog | `pipeline.dst.SESSION_CATALOG` |
| ORB window timing (UTC) | `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` |
| Entry models / filters | `trading_app.config` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| DB path | `pipeline.paths.GOLD_DB_PATH` |
| Holdout policy (Mode A) | `trading_app.holdout_policy` — `HOLDOUT_SACRED_FROM`, `HOLDOUT_GRANDFATHER_CUTOFF`, `enforce_holdout_date()` |

**`orb_utc_window` authority:** single source of truth for "compute ORB window end UTC" shared by backtest (`outcome_builder`), live engine (`execution_engine`), and feature builder (`build_daily_features`). Any divergence between these paths is a look-ahead bias risk per Chan Ch 1 p4. Never re-implement in another file; never fall back to `break_ts` when canonical inputs are missing — raise `ValueError` instead. See `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.

**`holdout_policy` authority:** single source of truth for Mode A sacred-window constants. Consumed by `pipeline.check_drift.check_holdout_contamination()` (contamination detector), `pipeline.check_drift.check_holdout_policy_declaration_consistency()` (docs ↔ code drift detector), `trading_app.strategy_discovery.main()` (CLI enforcement via `enforce_holdout_date()`), and `trading_app.strategy_validator._check_mode_a_holdout_integrity()` (pre-promotion gate). Never inline `date(2026, 1, 1)` or `datetime(2026, 4, 8, ...)` in any downstream file. Changing the canonical values requires a new amendment to `docs/institutional/pre_registered_criteria.md` and a matching update to `RESEARCH_RULES.md`; the declaration-consistency drift check catches any such drift. See `docs/plans/archive/2026-04/2026-04-07-holdout-policy-decision.md` for the Mode B → Mode A correction audit trail.

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
Metadata, comments, docstrings, bundle fields, and config labels are NOT evidence.
- Never trust a model bundle's `rr_target_lock` field without querying what data it trained on
- Never trust a check's `PASSED` label without confirming the check actually tests what it claims
- Never trust a function docstring's description of behavior without reading the code
- Never trust row counts from memory — execute the query and read the output
- When inspecting ML models, trace FROM the database query THROUGH the code TO the model output
- When verifying drift checks, inject a known violation and confirm it's caught
- **Reading code is not verifying code. Verifying requires execution + output inspection.**

## 8. Research Finding Staleness — Never Inline Stats
Never inline research stats (p-values, N counts) in code. Use `@research-source` + `@revalidated-for` annotations. Full rules → `.claude/rules/research-truth-protocol.md`.
