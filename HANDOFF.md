# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-12
- **Commit:** c9ec08f3 — docs(institutional): wire new extracts into backtesting-methodology and mechanism_priors auto-load
- **Files changed:** 4 files
  - `.claude/rules/backtesting-methodology.md`
  - `HANDOFF.md`
  - `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`
  - `docs/institutional/mechanism_priors.md`

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
