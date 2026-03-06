# 4TP Design: Live Fill Tracking + Orphan Blocking

**Date:** 2026-03-07
**Trigger:** Bloomey review HIGH-1 (phantom fill prices) + HIGH-2 (orphan positions log-only)
**Scope:** `trading_app/live/` — no pipeline, no schema, no entry model changes

## Problem Statement

### HIGH-1: Phantom Fill Prices
The live session uses ExecutionEngine's simulated `event.price` for all P&L tracking.
Neither broker router returns actual fill prices — both return only `order_id` + `status`.
PerformanceMonitor, CUSUM, and RiskManager all operate on fictional prices.

### HIGH-2: Orphaned Positions Not Blocked
On startup, `positions.query_open()` detects orphans and logs CRITICAL, but execution
continues. RiskManager has zero visibility into broker-side orphans. After any crash,
untracked exposure accumulates.

## Design

### HIGH-2: Orphan Blocking (3 lines)
- If orphans detected and `force_orphans=False` (default): raise `RuntimeError`
- New `--force-orphans` CLI flag to acknowledge and continue
- Zero behavior change when no orphans exist

### HIGH-1: Fill Price Tracking (Layer 1+2)

**Layer 1: Capture fill from submit response**
- Both broker `submit()` methods already get JSON back. Market orders may include
  fill price in response. Extract it when present, return as `fill_price: float | None`.
- `_entry_prices` changes from `dict[str, float]` to `dict[str, dict]` storing
  `{engine_price, fill_price, slippage}`.
- Log slippage on every fill where both prices are known.
- `_record_exit` uses `fill_price` when available, falls back to `engine_price`.

**Layer 2: EOD slippage summary**
- `daily_summary()` includes total slippage points across all trades.
- `TradeRecord` gets `slippage_pts` field for tracking.

**Layer 3 (FUTURE — not this PR):**
- Subscribe to ProjectX User Hub / Tradovate WebSocket for real-time fill events.
- Required for E2 stop entries where fill happens minutes after submission.

## Files Changed

| File | Change |
|------|--------|
| `session_orchestrator.py` | Orphan blocking, fill price tracking in _entry_prices, slippage logging |
| `scripts/run_live_session.py` | `--force-orphans` CLI flag |
| `broker_base.py` | Document submit() return includes optional fill_price |
| `projectx/order_router.py` | Extract fill_price from response |
| `tradovate/order_router.py` | Extract fill_price from response, extend OrderResult |
| `performance_monitor.py` | Add slippage_pts to TradeRecord, slippage in summary |
| `tests/test_trading_app/test_session_orchestrator.py` | New — orphan blocking + fill tracking tests |

## Risks
- Broker APIs may not return fill_price on submit → None fallback, zero behavior change
- E2 stops won't have fill at submit time → expected, Layer 3 future fix
- `_entry_prices` type change is contained to session_orchestrator.py (3 methods)

## Rollback
Revert commit. No schema, no DB, no pipeline impact.
