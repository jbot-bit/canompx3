# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-12
- **Commit:** 7bf689a4 — docs(audit): post-self-review polish — scratch-policy doc + CLI invocations
- **Files changed:** 6 files
  - `HANDOFF.md`
  - `docs/audit/results/2026-05-12-sr-alarm-3lane-summary.md`
  - `docs/audit/results/2026-05-12-sr-alarm-comex-settle-rr1.5.md`
  - `docs/audit/results/2026-05-12-sr-alarm-nyse-open-rr1.md`
  - `docs/audit/results/2026-05-12-sr-alarm-us-data-1000-rr1.5.md`
  - `research/sr_alarm_steps_3_4_5_2026_05_12.py`

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
