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
| Entry models / filters | `trading_app.config` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| DB path | `pipeline.paths.GOLD_DB_PATH` |

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

## 5. Evidence Over Assertion
Provide verifiable output, not claims. Show command output, row counts, test results.

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
