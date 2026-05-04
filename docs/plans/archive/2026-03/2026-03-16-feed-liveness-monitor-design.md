---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Feed Liveness Monitor — Design

**Date:** 2026-03-16
**Status:** 4TP — design complete, implementing
**Problem:** pysignalr connection can silently die (connected but no data), causing missed trades overnight with no alert
**Priority:** #1 blocker for unattended automation

## Problem

The ProjectX SignalR data feed can enter a "connected but silent" state where:
- WebSocket connection is technically alive (pings succeed)
- No market data arrives
- No bars produced, no ORBs formed, no trades placed
- Nobody knows until morning

This happens when: server-side subscription drops silently, network route degrades at app level, or SignalR hub reconnects but loses subscription state.

Tradovate feed has partial detection (`_LIVENESS_TIMEOUT = 60s`) but only logs a warning — no reconnect, no alert.

## Solution

Add liveness monitoring to both feed implementations:
1. Track `_last_data_at` on every quote/trade received
2. Check staleness in the existing stop-file watcher loop (every 2.5s)
3. After 2 consecutive stale periods (90s each = ~3 minutes) → force reconnect
4. Fire `on_stale` callback to orchestrator → push alert notification

## Files Changed

| File | Change |
|------|--------|
| `trading_app/live/projectx/data_feed.py` | Add `_last_data_at`, stale detection in watcher, `_force_reconnect` flag |
| `trading_app/live/tradovate/data_feed.py` | Elevate existing liveness warning to reconnect + alert |
| `trading_app/live/broker_base.py` | Add optional `on_stale` callback to BrokerFeed |
| `trading_app/live/session_orchestrator.py` | Wire `on_stale` to `_notify()` for push alerts |
| `tests/test_trading_app/test_projectx_feed.py` | 4 new tests: stale detection, reset, callback, reconnect |

## Key Design Decisions

- **90s timeout:** 1.5 bar periods. During active trading, zero ticks for 90s is abnormal.
- **2 consecutive stale checks before reconnect:** Prevents single-check false positives (e.g., thin market moment).
- **`_force_reconnect` separate from `_stop_requested`:** Reconnect resumes the loop; stop exits it.
- **`on_stale` callback (not hardcoded alert):** Keeps feed layer broker-agnostic per existing ABC pattern.

## Risks

1. False positive during exchange maintenance — harmless (reconnect, resubscribe, wait)
2. Reconnect storm if server is down — bounded by existing `_MAX_RECONNECTS = 20` + exponential backoff
3. Race between stale-reconnect and stop-file — `_stop_requested` takes priority
