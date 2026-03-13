# Webhook Order Deduplication

**Date:** 2026-03-13
**Status:** Approved
**Triggered by:** PLAN_codex audit — webhook endpoint has zero replay/dedup protection

## Problem

If TradingView fires the same alert twice within seconds (network retry, double-fire, or replay),
both orders reach the broker and fill. No layer stops the duplicate:
- Rate limiter is timestamp-based (2 requests 500ms apart both pass)
- No idempotency key in request model
- Order routers send no `clientOrderId` to broker
- PositionTracker only protects the engine path, not webhook

Result: 2x position size, 2x capital at risk, no recovery mechanism.

## Design

In-memory TTL dedup cache in `webhook_server.py`. Key = `(instrument, direction, action)`.
If identical request arrives within configurable window (default 10s), return cached response
with `status="deduplicated"` instead of placing a new order.

### Dedup Key

`f"{instrument}:{direction}:{action}"` — e.g. `"MGC:long:entry"`

Excludes `entry_model`, `qty`, `entry_price` from key because:
- TradingView shouldn't fire both E1 and E2 for the same signal
- Qty changes within 10s would be a config error, not a legitimate trade
- Entry price changes (E2 stop) within 10s are the same signal

### Flow

```
POST /trade
  → HMAC check (existing)
  → Dedup check (NEW) — if cached + within window → return cached response
  → Rate limit (existing)
  → Contract resolution (existing)
  → Place order (existing)
  → Cache response (NEW) — store key → (timestamp, response)
```

### Configuration

`WEBHOOK_DEDUP_SECONDS` env var, default 10. Set to 0 to disable.

### Edge Cases

- Server restart: cache clears — safe (alerts are bar-close driven, minutes apart)
- Exit after entry: different `action` = different key, not deduped
- Reverse direction: different `direction` = different key, not deduped
- Legitimate rapid trades: 10s window is shorter than any ORB bar (1 min minimum)

### Files Changed

- `trading_app/live/webhook_server.py` — dedup cache + check + prune (~25 lines)
- `tests/test_trading_app/test_webhook_server.py` — 3 new tests

### Not In Scope (V1)

- `clientOrderId` on broker orders (V2 — requires broker API research)
- PositionTracker integration (separate architecture, separate code path)
- Persistent dedup store (Redis/DB — unnecessary for in-process single-instance server)
