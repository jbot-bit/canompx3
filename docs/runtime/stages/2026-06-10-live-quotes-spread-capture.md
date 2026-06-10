---
task: "Defect B — capture live bid/ask spread to a separate live_quotes table to validate the backtest cost model (off by default, fail-open, never touches the capital path)"
mode: IMPLEMENTATION
scope_lock:
  - pipeline/init_db.py
  - trading_app/live/spread_accumulator.py
  - trading_app/live/quote_persister.py
  - trading_app/live/projectx/data_feed.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_spread_accumulator.py
  - tests/test_trading_app/test_quote_persister.py
  - tests/test_trading_app/test_projectx_spread_capture.py
---

## Blast Radius

- `pipeline/init_db.py` — additive: +`LIVE_QUOTES_SCHEMA` constant + 1 `con.execute` call. `bars_1m` DDL UNTOUCHED. No existing row/column affected. New table is `CREATE TABLE IF NOT EXISTS` (idempotent).
- `trading_app/live/spread_accumulator.py` — NEW file, zero callers when flag off. Thread-safe (signalrcore fires on a foreign thread). Mirrors `BarAggregator` minute-roll.
- `trading_app/live/quote_persister.py` — NEW file, clones `BarPersister`. Writes ONLY to `live_quotes` (separate table). Fail-open (CRITICAL-log, return 0). Idempotent DELETE+INSERT.
- `trading_app/live/projectx/data_feed.py` — additive: `parse_bid_ask` (pure), gated `CANOMPX_CAPTURE_SPREAD` flag in `__init__`, a shared `_capture_spread(bid, ask, now)` helper invoked from BOTH `_on_quote` (async/pysignalr) AND `_on_quote_sync`/`_apply_tick_state` (foreign-thread/signalrcore), and accumulator flush inside existing `flush()`. The spread branch is its own `try/except Exception` and CANNOT raise into the bar/order path. Existing `parse_quote`/`_cum_to_delta`/bar lines UNCHANGED.
- `trading_app/live/session_orchestrator.py` — additive, all gated behind the flag: `QuotePersister` ctor near `:763`, `on_quote_minute=` kwarg in feed ctor at `:4125`, new `_on_quote_minute` method (mirrors `:1958`), flush AFTER the bar flush at `:4452`. Zero construction/wiring when flag off → capital path byte-identical to today.
- Reads: `pipeline.cost_model.COST_SPECS` (validation query only, future), `pipeline.paths.GOLD_DB_PATH`. Writes: `live_quotes` only, at session shutdown, single-writer (same seam as bars).
- Capital-path risk: LOW. Off by default; own exception block; separate table; flushes AFTER bars (bars are the capital artifact, flush first).

## Design corrections over the approved plan (verified in code 2026-06-10, HEAD 99235992)

1. **Both quote paths covered.** `data_feed.py` has `_on_quote` (pysignalr, async) AND `_on_quote_sync` (signalrcore, foreign thread). The approved plan wired only the former → `live_quotes` would be SILENTLY empty on a signalrcore session. Fix: shared `_capture_spread` helper called from both; sync path is thread-safe because the accumulator owns a lock.
2. **No async callback.** `on_quote_minute` is a PLAIN synchronous, thread-safe call (like `BarPersister.append`), so the foreign-thread sync path can invoke it directly. No `await`, no `call_soon_threadsafe` needed for the append (the persister list is lock-guarded).
3. **Accumulator lock.** `SpreadAccumulator.add` carries a `threading.Lock` (same reason `BarAggregator` does).
4. **Flush the last partial minute.** `DataFeed.flush()` flushes the accumulator's in-progress minute too, else the final minute of every session is lost.

## Crossed/locked guard
`ask <= bid` or missing bid/ask → DROP the tick, exclude from `n_quotes`, debug-log. Coercing to 0 would understate measured friction (the dangerous direction).

## Verification gates
- Unit: SpreadAccumulator (drop/avg/min/max/close/roll), QuotePersister (idempotent, fail-open).
- Isolation (load-bearing): flag off → neither object built, no spread branch; flag on → raised exception in spread branch does NOT propagate into bar/order path.
- `python pipeline/check_drift.py` exit 0; ruff clean (show output).
- Adversarial-audit gate: `evidence-auditor` on `data_feed.py` diff before done (`adversarial-audit-gate.md` — trading_app/live/, judgment change, new write path).
- Owed: live behavioral proof on an active feed (flag ON, confirm `live_quotes` populates, avg_spread near modelled one-way values).
