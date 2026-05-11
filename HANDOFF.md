# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-11
- **Commit:** b2470202 — fix(prereg): declare scratch_policy on 2026-05-07 chordia unlock (Criterion 13)
- **Files changed:** 3 files
  - `HANDOFF.md`
  - `docs/audit/hypotheses/2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml`
  - `docs/runtime/stages/pr-257-audit-trail-corrections.md`

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
