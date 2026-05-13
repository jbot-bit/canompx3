# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-12
- **Commit:** 204dc964 — docs(institutional): mark Tolusic + Howard as INGESTED in PENDING_ACQUISITION
- **Files changed:** 2 files
  - `HANDOFF.md`
  - `docs/institutional/literature/PENDING_ACQUISITION_market_profile.md`

## Current Session Addendum
- **Tool:** Codex (WSL)
- **Date:** 2026-05-12
- **Summary:** Fixed `gold-db` MCP WAL replay failure by moving stale orphan `/home/joshd/canompx3/gold.db.wal` to `/tmp/canompx3-gold.db.wal.stale-2026-05-12-setup-debug`; `gold-db` MCP table counts and MGC fitness now work. Wrote and ran bounded prereg `2026-05-12-vwap-dead-confluence-reaudit-v1`; result `PASS_STANDS` confirms VWAP remains closed as a broad confluence family while exact VWAP lanes remain exact-lane facts only. CB Insights remains cached/enabled as a plugin descriptor but not callable until connector/app install completes.

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.
2. Deployment-coverage decision (2026-05-12) — 78 ROUTABLE_DORMANT strategies (sum annual_r=809.4R) blocked behind inactive profile whitelists. See `docs/audit/results/2026-05-12-deployment-coverage-orphans.md`. Audit is read-only; activation/whitelist edits are a separate decision per profile.
3. Strict 3.79 exposure closeout (2026-05-12) — `docs/audit/results/2026-05-12-deployed-lanes-chordia-strict-379-exposure-audit.md`: zero live-capital exposure to chordia-loader-has-theory-silent-downgrade debt. Debt entry remains open (loader behavior unchanged) but exposure verified clean.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
