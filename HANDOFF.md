# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex (WSL)
- **Date:** 2026-05-10
- **Commit:** 7cfacd6b — fix(pipeline): harden rebuild resume chain
- **Files changed:** 7 files
  - `REPO_MAP.md`
  - `pipeline/build_daily_features.py`
  - `scripts/tools/pipeline_status.py`
  - `scripts/tools/run_rebuild_with_sync.sh`
  - `scripts/tools/stress_test_chain_integrity.py`
  - `tests/test_pipeline/test_build_daily_features.py`
  - `tests/test_pipeline/test_pipeline_status.py`

## Next Steps — Active
1. Treat current MNQ book as controlled live pilot only: 1 contract, no institutional/full-production language, no scaling until SR watch recheck, chain warnings, and event-tail debt are explicitly reviewed.
2. If moving to broker execution, run the normal live preflight immediately before session start; do not rely on this handoff as live-market freshness.
3. Continue PR48 MES q45_exec bridge / MGC shadow-only closeout / Track D MNQ COMEX_SETTLE Gate 0 runner only after preserving the controlled-pilot deployability state above.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
