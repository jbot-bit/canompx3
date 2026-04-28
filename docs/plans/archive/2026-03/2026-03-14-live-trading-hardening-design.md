---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Live Trading Pre-Go-Live Hardening — Design Doc

**Date:** 2026-03-14
**Source:** Full infrastructure audit (Mar 14) — 1 CRITICAL, 3 HIGH, 4 MEDIUM findings
**Scope:** Fix all CRITICAL and HIGH findings before real money trading
**Blast radius:** 5 files modified, 3 files created (8 total)

---

## Goal

Harden the live trading infrastructure to survive process crashes, broker API failures, and operator errors — closing the gaps between "works in paper trading" and "safe for real money."

## Findings (from audit)

| # | Severity | Finding | File(s) |
|---|----------|---------|---------|
| 1 | CRITICAL | No persistent trade journal — all trade records in memory, crash = data loss | performance_monitor.py, session_orchestrator.py |
| 2 | HIGH | No exit order retry — single HTTP failure = position left open | tradovate/order_router.py, projectx/order_router.py |
| 3 | HIGH | pysignalr feed won't stop gracefully — only process kill works | projectx/data_feed.py |
| 4 | HIGH | Tradovate data feed has zero test coverage | tradovate/data_feed.py |
| 5 | HIGH | Webhook bypasses all risk management | webhook_server.py |

---

## Fix 1: Persistent Trade Journal (CRITICAL)

### Design Decision: Separate DB
Use `live_journal.db` (not `gold.db`) to avoid write contention with pipeline rebuilds. TradeJournal class takes `db_path` parameter. Fail-open: journal write failure logs CRITICAL but NEVER blocks trading loop.

### Schema
```sql
CREATE TABLE IF NOT EXISTS live_trades (
    trade_id       TEXT PRIMARY KEY,
    trading_day    DATE NOT NULL,
    instrument     TEXT NOT NULL,
    strategy_id    TEXT NOT NULL,
    direction      TEXT NOT NULL,
    entry_model    TEXT NOT NULL,
    entry_price    DOUBLE,
    exit_price     DOUBLE,
    fill_entry     DOUBLE,
    fill_exit      DOUBLE,
    actual_r       DOUBLE,
    expected_r     DOUBLE,
    slippage_pts   DOUBLE,
    pnl_dollars    DOUBLE,
    exit_reason    TEXT,
    cusum_alarm    BOOLEAN DEFAULT FALSE,
    broker         TEXT,
    order_id_entry TEXT,
    order_id_exit  TEXT,
    session_mode   TEXT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT now()
);
```

### Interface: TradeJournal class
New file: `trading_app/live/trade_journal.py` (~80 lines)

```python
class TradeJournal:
    def __init__(self, db_path: Path, mode: str = "live"):
        """Open DuckDB connection. Creates table if not exists."""

    def record_entry(self, trade_id, trading_day, instrument, strategy_id,
                     direction, entry_model, entry_price, fill_entry,
                     broker, order_id_entry) -> None:
        """INSERT partial row on entry. Fail-open."""

    def record_exit(self, trade_id, exit_price, fill_exit, actual_r,
                    expected_r, slippage_pts, pnl_dollars, exit_reason,
                    order_id_exit, cusum_alarm=False) -> None:
        """UPDATE existing row on exit. Fail-open."""

    def daily_summary(self, trading_day) -> dict:
        """Query completed trades for EOD report."""

    def close(self) -> None:
        """Close DB connection."""
```

### Wiring
- `SessionOrchestrator.__init__()`: create `TradeJournal` instance
- `_handle_event()` ENTRY branch: call `journal.record_entry()` after broker submit
- `_record_exit()`: call `journal.record_exit()` after computing R
- `_emergency_flatten()`: call `journal.record_exit()` with `exit_reason='kill_switch'`
- `post_session()`: call `journal.close()`

### Test strategy
`tests/test_trading_app/test_trade_journal.py` (~120 lines):
- Insert entry, update exit, verify round-trip
- Daily summary aggregation
- Partial row (entry without exit) survives crash
- Duplicate trade_id raises or updates gracefully
- Fail-open: DB error doesn't propagate

---

## Fix 2: Exit Order Retry (HIGH)

### Design
Add `_submit_exit_with_retry()` to `SessionOrchestrator`. 3 attempts, linear backoff (1s, 2s, 3s). Only for EXIT/SCRATCH, never for ENTRY.

```python
async def _submit_exit_with_retry(self, spec, strategy_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = self.router.submit(spec)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 1.0 * (attempt + 1)
                log.warning("Exit retry %d/%d for %s: %s",
                           attempt + 1, max_retries, strategy_id, e)
                await asyncio.sleep(wait)
            else:
                log.critical("EXIT FAILED after %d retries for %s: %s",
                            max_retries, strategy_id, e)
                await self._notify(
                    f"MANUAL CLOSE REQUIRED: {strategy_id} exit failed "
                    f"after {max_retries} retries"
                )
                raise
```

### Wiring
Replace `self.router.submit(exit_spec)` with `await self._submit_exit_with_retry(exit_spec, strategy_id)` in:
- EXIT branch (~line 850)
- SCRATCH branch
- Rollover close loop (~line 475)

### Test strategy
Add to `test_session_orchestrator.py`:
- Mock router fails twice, succeeds third → verify retry count
- Mock router fails all 3 → verify CRITICAL log + notification
- Verify ENTRY path is NOT retried

---

## Fix 3: pysignalr Graceful Stop (HIGH)

### Root cause
`_stop_file_watcher()` sets flag but never stops `client.run()`.

### Fix
Store the asyncio task running `client.run()` and cancel it from the stop watcher:

```python
# In _run_pysignalr():
client_task = asyncio.create_task(client.run())
stop_task = asyncio.create_task(self._stop_file_watcher(client_task))
try:
    await client_task
except asyncio.CancelledError:
    log.info("pysignalr client cancelled by stop file")
    self._stop_requested = True
finally:
    stop_task.cancel()

# In _stop_file_watcher(client_task):
async def _stop_file_watcher(self, client_task=None):
    while not self._stop_requested:
        await asyncio.sleep(2.5)
        if _STOP_FILE.exists():
            log.info("Stop file detected")
            self._stop_requested = True
            if client_task and not client_task.done():
                client_task.cancel()
            return
```

### Test strategy
Add to `test_projectx_feed.py`:
- Create stop file → verify client task cancelled
- Verify clean exit (no hanging coroutines)

---

## Fix 4: Tradovate Data Feed Tests (HIGH)

### New file: `tests/test_trading_app/test_tradovate_feed.py` (~150 lines)

10 test cases:
1. `test_handle_frame_extracts_bid_price`
2. `test_handle_frame_falls_back_to_price`
3. `test_handle_frame_no_price_logs_warning`
4. `test_bar_completion_calls_on_bar`
5. `test_heartbeat_sends_empty_array`
6. `test_stop_file_sets_flag`
7. `test_reconnect_exponential_backoff`
8. `test_max_reconnects_exhausted`
9. `test_liveness_warning_after_60s`
10. `test_flush_emits_partial_bar`

Mock `websockets.connect()`, inject fake frames, verify callbacks.

---

## Fix 5: Webhook Risk Hardening (HIGH)

### Changes to `webhook_server.py`

1. **Circuit breaker** — import `CircuitBreaker`, create module-level instance, check before submit, record success/failure
2. **Instrument allowlist** — import `ACTIVE_ORB_INSTRUMENTS`, reject unknown instruments with 400
3. **Qty cap** — `MAX_QTY = int(os.getenv("WEBHOOK_MAX_QTY", "5"))`, reject above with 400
4. **Broker factory** — replace hardcoded `TradovateOrderRouter` with `create_router()` from `broker_factory`
5. **Fix OrderResult bug** — line 242 uses `.get()` on dataclass, should use `getattr()`

### NOT adding
Full RiskManager, PositionTracker, daily loss limit — webhook is intentionally stateless. Circuit breaker + allowlist + qty cap provide defense-in-depth.

### Test strategy
Add to `test_webhook_server.py`:
- Circuit breaker blocks after 5 failures
- Unknown instrument rejected (400)
- Qty > MAX_QTY rejected (400)
- Circuit breaker resets on success

---

## Implementation Order

1. **Fix 1: Trade Journal** (CRITICAL, ~half day) — schema + class + wiring + tests
2. **Fix 2: Exit Retry** (~2 hours) — one new method + 3 call site changes + tests
3. **Fix 3: pysignalr Stop** (~1 hour) — 3 line changes + test
4. **Fix 4: Tradovate Feed Tests** (~half day) — new test file only
5. **Fix 5: Webhook Risk** (~3 hours) — refactor + guards + tests

Each fix is independently committable. No cross-dependencies between fixes.

---

## Rollback Plan

| Fix | Rollback |
|-----|----------|
| 1 | Drop `live_trades` table, remove journal from orchestrator (3 call sites) |
| 2 | Replace `_submit_exit_with_retry()` with direct `router.submit()` |
| 3 | Revert data_feed.py to previous `_stop_file_watcher` pattern |
| 4 | Delete test file |
| 5 | Revert webhook_server.py |

---

## Risk: DB Contention (Fix 1)

Using separate `live_journal.db` eliminates contention with `gold.db` pipeline writes. The journal DB is append-only during sessions, read-only for analysis. No FK constraints to pipeline tables. If consolidation into `gold.db` is desired later, a migration script can copy rows.

## Risk: Duplicate Exit Orders (Fix 2)

PositionTracker already prevents duplicate exit transitions (PENDING_EXIT → reject). The retry only re-submits to the broker if the HTTP call failed — the position state hasn't changed. If the first attempt actually succeeded at the broker but the HTTP response was lost (timeout), the retry would submit a duplicate. Mitigation: the broker's own idempotency on order_id, plus PositionTracker state prevents the engine from processing the trade twice.
