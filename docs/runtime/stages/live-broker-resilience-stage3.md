---
task: Live-broker resilience hardening — Stage 3 (broker-state-unknown kill-switch SLA)
mode: IMPLEMENTATION
stage: 3
total_stages: 5
worktree: C:/Users/joshd/canompx3/.worktrees/live-broker-resilience
branch: feat/live-broker-resilience
---

## Scope Lock

- trading_app/live/session_orchestrator.py
- tests/test_trading_app/test_equity_age_watchdog.py
- docs/runtime/stages/live-broker-resilience-stage3.md

## Blast Radius

- trading_app/live/session_orchestrator.py — adds EQUITY_AGE_SLA constant + a new branch in the existing `_watchdog()` coroutine. Calls the already-shipped `query_equity_with_age()` from Stage 1. Existing feed-dead branch unchanged. New code path is purely additive (defends a new failure mode the feed-watchdog cannot see: broker accepts our connection but stops answering equity reads with positions open).
- tests/test_trading_app/test_equity_age_watchdog.py — new file. Mocks positions, drives the SLA branch directly with a synthesized stale `EquityReading` and asserts kill-switch + `_emergency_flatten()` interaction. Does NOT exercise full asyncio sleep.
- Reads: live broker APIs via existing client (no new endpoints). Writes: SessionSafetyState (existing `kill_switch_fired` flag).
- Not touched: http_client, projectx/*, tradovate/*, bot_dashboard, circuit_breaker (Stage 4), drift checks (Stage 5).

## Why this stage

Stages 1+2 closed: (a) every transient HTTP failure is now retried-with-deadline, (b) every order place is idempotent. What is NOT yet closed: the case where the broker stops responding to *reads* (equity/positions) while we have a live position open. The retries inside the HTTP client will surface the failure as `BrokerHTTPError`, but no orchestrator code today acts on that.

Tonight's TopStepX incident is the proof case — equity went `Balance —` for 90+ seconds with the bot still nominally "connected". Stages 1+2 make the HTTP fail faster and louder; Stage 3 makes the orchestrator act when those reads stay stale past the SLA.

The kill-switch already exists (`KILL_SWITCH_TIMEOUT = 300s` feed-dead branch). Stage 3 adds the *broker-state-unknown* branch: if `query_equity_with_age(account).age_s > EQUITY_AGE_SLA` AND `_positions.active_positions()` is non-empty, fire the same kill switch.

## Done criteria

- `EQUITY_AGE_SLA` constant added with a comment explaining the chosen value (≥ ProjectX retry budget + grace, ≤ feed-dead timeout).
- `_watchdog()` branches: if signal-only OR no order_router OR no active positions → skip equity check (cheap path); else read `query_equity_with_age()`, observe `age_s`, fire kill switch if stale.
- Equity-fetch errors (BrokerHTTPError) inside the watchdog are caught with logging + `record_failure`-style behaviour and treated as "broker unreachable past retry budget" → also fire kill switch (the explicit motivation for the SLA).
- New test file covers: (1) fresh equity → no kill switch, (2) stale equity past SLA with active position → kill switch fires, (3) stale equity with NO active position → no kill switch (don't flatten nothing), (4) BrokerHTTPError raised → kill switch fires.
- `pipeline/check_drift.py` green.
- Targeted pytest green.
- Self-review pass (code-review skill) before commit.
