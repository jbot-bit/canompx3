---
task: >
  Restore the missing daily forward-paper accrual automation and harden it.
  Root cause: NO project scheduled task is registered (schtasks /query shows
  zero canompx tasks), which is why paper_trades froze at 2026-04-24 (33-day
  gap, closed manually this session via --sync). Append
  `python -m trading_app.paper_trade_logger --sync` to the canonical refresh
  chain (scripts/daily_refresh.bat) so the forward-paper clock self-maintains
  through the 2026-10-07 deployment gate, then register the daily task in a
  CRASH-SAFE configuration (fixed-time daily 07:30 Brisbane, NOT ONSTART/
  ONLOGON) per operator constraint "don't fuck up my startup, lots of crashes"
  (ref i9-14900HX instability mitigation in memory). Doctrine 6-month
  forward-paper lock (pre_registered_criteria.md Amendment 2.6) UNCHANGED.
mode: IMPLEMENTATION
scope_lock:
  - scripts/daily_refresh.bat
blast_radius: >
  scripts/daily_refresh.bat — append ONE line to the existing refresh chain
  (paper_trade_logger --sync) after refresh_data builds outcomes. The .bat is
  run by Windows Task Scheduler. Reads orb_outcomes + daily_features (read
  only); writes paper_trades (gold.db) incremental + idempotent (DELETE-before-
  insert per strategy_id, OOS boundary assertion at paper_trade_logger.py:285).
  Does NOT touch: live trading path, live_trades, validated_setups, allocator,
  any deployment-gate logic, startup/boot sequence. Task registration is a
  schtasks system-state change (separate operator-gated step, fixed-time daily,
  crash-safe — a missed run self-heals on the next run's MAX(trading_day)+1
  cursor). MS schtasks has no native catch-up, which is fine: the sync cursor
  is the catch-up mechanism.
---

## Context
- 33-day paper_trades gap closed manually this session (`--sync`, +282 rows, latest now 2026-05-25).
- 9 orphaned lane strategy_ids correctly preserved as frozen forward history (NOT deleted).
- Only 1 of 4 current live lanes overlapped the stale set (lanes rebalanced) — `--sync` (not full backfill) is the correct flag so orphans stay frozen.

## Why a fixed-time daily task is crash-safe
- 07:30 Brisbane is a clock time, after all US sessions close — NOT tied to startup/login.
- If PC is asleep/crashed at 07:30, plain schtasks skips that run (MS docs: no native start-when-available). Next run's per-lane `MAX(trading_day)+1` cursor backfills the missed days from orb_outcomes. No corruption possible.
- Strictly LESS startup impact than the project's original intent (already a fixed-time daily task, never ONSTART).

## Acceptance
1. daily_refresh.bat contains the --sync line after refresh_data, before the completion echo.
2. `git diff` shows exactly one added invocation, logged to logs\daily_refresh.log like its siblings.
3. check_drift.py passes.
4. Task registration held for explicit operator go (system-state change).
