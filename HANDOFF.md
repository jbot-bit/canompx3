# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-05-10
- **Summary:** Deployability guardrail pass completed for `topstep_50k_mnq_auto`. Active runtime allocation is now 2 MNQ lanes; `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` is marked `PAUSE` in `docs/runtime/lane_allocation.json` because it has a persisted SR alarm pause. Refreshed C11/C12 state after allocation change. Full profile deployability audit reports `2/2 CONTROLLED_LIVE_PILOT_CANDIDATE`, hard issues `{}`, institutional-language candidates `0`. Full adversarial stress gate exits `GO` with no hard blockers/silences.
- **Residual warnings:** FDR audit warnings remain portfolio-accounting warnings (847 raw active, 353 unique streams, 527 families; 2 non-profile rows would fail current K). Chain integrity reports `SUSPECT` with `critical=0` and warnings only. Both active lanes have reviewed SR `watch` alarms, short-history warning (`years_tested=6`), MNQ event-tail slippage debt (`PENDING_EVENT_TAIL`), and weak DSR cross-check. These are controlled-pilot warnings, not full-production clearance.
- **Verification:** `pytest tests/test_tools/test_adversarial_stress_gate.py tests/test_trading_app/test_deployability.py tests/test_tools/test_live_readiness_report.py tests/test_tools/test_backfill_deployability_evidence.py tests/test_pipeline/test_check_drift_ws2.py::TestValidatedSetupsWriterAllowlist -q` = 40 passed. `ruff check` on touched files passed. `pipeline/check_drift.py` = no drift, 123 OK, 19 advisory. `scripts/tools/adversarial_stress_gate.py --profile topstep_50k_mnq_auto --format text --timeout-seconds 900` = GO.

## Next Steps — Active
1. Treat current MNQ book as controlled live pilot only: 1 contract, no institutional/full-production language, no scaling until SR watch recheck, chain warnings, and event-tail debt are explicitly reviewed.
2. If moving to broker execution, run the normal live preflight immediately before session start; do not rely on this handoff as live-market freshness.
3. Continue PR48 MES q45_exec bridge / MGC shadow-only closeout / Track D MNQ COMEX_SETTLE Gate 0 runner only after preserving the controlled-pilot deployability state above.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
