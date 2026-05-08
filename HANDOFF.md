# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code (autonomous)
- **Date:** 2026-04-28
- **Summary:** Queue-backed baton refreshed from canonical active-work registry.

## Next Steps — Active
1. PR48 MES q45_exec bridge — Define the honest bridge from the alive MES q45_exec research branch into a bounded runtime surface.
2. PR48 MGC shadow-only observation closeout — Observe the MGC shadow-only context in dashboard and live logs and record whether the visibility path behaves as designed.
3. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Blockers / Warnings
- Stale queue items need re-verification: mes_q45_exec_bridge, pr48_mgc_shadow_observation, track_d_mnq_comex_settle_gate0_runner_design

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
