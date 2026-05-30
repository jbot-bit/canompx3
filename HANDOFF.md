# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13. Added a read-only Chordia unlock batch planner at `scripts/tools/chordia_unlock_batch.py`; generated `artifacts/research/chordia_unlock_batch_2026_05_30/` with 100 missing-verdict candidates for `topstep_50k_mnq_auto` (priority counts `{0: 14, 4: 1, 5: 85}`). Added planned read-only lane bench state surface at `scripts/tools/lane_bench_state.py`; generated `artifacts/research/lane_bench_state_2026_05_30/` over 848 shelf rows with state counts `{ALLOCATOR_ELIGIBLE_BENCH: 14, EXACT_LANE_READY_FOR_REPLAY: 708, KILLED: 12, LIVE_ACTIVE: 3, PARKED: 111}`. These tools do not mutate live allocation, `validated_setups`, or `chordia_audit_log.yaml`; they separate live-active, allocator-eligible bench, strict-replay queue, killed, and parked rows so throughput automation can target the right next action without weakening gates.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-30
- **Commit:** 1cc7f4a1 — fix(live): ralph iter 213 — lifecycle-block silent-fail + readiness effective-copies
- **Files changed:** 8 files
  - `scripts/run_live_session.py`
  - `scripts/tools/live_readiness_report.py`
  - `scripts/tools/refresh_control_state.py`
  - `tests/test_scripts/test_run_live_session_preflight.py`
  - `tests/test_tools/test_live_readiness_report.py`
  - `tests/test_tools/test_refresh_control_state.py`
  - `tests/test_trading_app/test_session_orchestrator.py`
  - `trading_app/live/session_orchestrator.py`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
