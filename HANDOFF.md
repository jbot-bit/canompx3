# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex (WSL)
- **Date:** 2026-05-10
- **Commit:** c87c344f — fix(deployability): persist selected-profile readiness gate
- **Files changed:** 8 files
  - `pipeline/audit_log.py`
  - `scripts/tools/full_shelf_deployability_audit.py`
  - `scripts/tools/pipeline_status.py`
  - `tests/test_pipeline/test_pipeline_status.py`
  - `tests/test_trading_app/test_deployability.py`
  - `trading_app/db_manager.py`
  - `trading_app/deployability.py`
  - `trading_app/deployability_state.py`

## Session Note
- Selected profile gate `topstep_50k_mnq_auto` passed with 2 MNQ controlled-live-pilot candidates, 0 hard blockers, and 0 institutional-language-approved lanes; append-only readiness rows were written to `deployment_readiness_evaluations` with rebuild id `manual-deployability-20260510`.

## Next Steps — Active
1. Treat current MNQ book as controlled live pilot only: 1 contract, no institutional/full-production language, no scaling until SR watch recheck, chain warnings, and event-tail debt are explicitly reviewed.
2. If moving to broker execution, run the normal live preflight immediately before session start; do not rely on this handoff as live-market freshness.
3. Continue PR48 MES q45_exec bridge / MGC shadow-only closeout / Track D MNQ COMEX_SETTLE Gate 0 runner only after preserving the controlled-pilot deployability state above.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
