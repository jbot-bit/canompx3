---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Live Trading Infrastructure Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all CRITICAL/HIGH/MEDIUM live trading audit findings from Mar 14 — persistent trade journal, exit retry, pysignalr stop, webhook risk, bar aggregator, rollover, bid price.

**Architecture:** All changes isolated to `trading_app/live/`. No schema changes to existing tables (live_trades is new). Each task is one commit.

**Tech Stack:** Python, DuckDB, asyncio, requests, FastAPI, pytest

---

## Task 1: Persistent Trade Journal — Schema (CRITICAL)

**Files:**
- Modify: `pipeline/init_db.py` — ✅ ALREADY DONE (live_trades table added)
- Test: `tests/test_pipeline/test_schema.py`

**Step 1: Add test**
```python
def test_live_trades_table_exists(tmp_path):
    from pipeline.init_db import init_db
    init_db(tmp_path / "test.db")
    con = duckdb.connect(str(tmp_path / "test.db"), read_only=True)
    tables = [t[0] for t in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]
    con.close()
    assert "live_trades" in tables
```

**Step 2: Run**
```bash
python -m pytest tests/test_pipeline/test_schema.py -x -q -k "live_trades"
```
Expected: PASS (table already exists from init_db change)

**Step 3: Commit**
```bash
git add pipeline/init_db.py tests/test_pipeline/test_schema.py
git commit -m "feat: add live_trades table to schema (persistent trade journal)"
```

---

## Task 2: Persistent Trade Journal — PerformanceMonitor (CRITICAL)

**Files:**
- Modify: `trading_app/live/performance_monitor.py`
- Modify: `trading_app/live/session_orchestrator.py` line 195
- Create: `tests/test_trading_app/test_performance_monitor.py`

**Context:** `_trades` list is in-memory only. Crash = day's data lost. Fix: write each trade to `live_trades` table on `record_trade()`.

**Step 1: Write failing test**
```python
# tests/test_trading_app/test_performance_monitor.py
import duckdb
from datetime import date, datetime, UTC
from unittest.mock import MagicMock
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord

def _make_strategy():
    s = MagicMock()
    s.strategy_id = "MGC_TEST_E1"
    s.expectancy_r = 0.1
    s.win_rate = 0.55
    s.rr_target = 2.0
    return s

def test_record_trade_writes_to_db(tmp_path):
    db = tmp_path / "test.db"
    con = duckdb.connect(str(db))
    con.execute("""CREATE TABLE live_trades (
        id INTEGER, strategy_id TEXT, trading_day DATE, direction TEXT,
        entry_price DOUBLE, exit_price DOUBLE, actual_r DOUBLE,
        expected_r DOUBLE, slippage_pts DOUBLE, recorded_at TIMESTAMPTZ
    )""")
    con.close()

    monitor = PerformanceMonitor([_make_strategy()], db_path=db)
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))

    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("SELECT strategy_id FROM live_trades").fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "MGC_TEST_E1"

def test_record_trade_no_db_doesnt_crash():
    monitor = PerformanceMonitor([_make_strategy()], db_path=None)
    monitor.record_trade(TradeRecord(
        strategy_id="MGC_TEST_E1", trading_day=date(2026, 3, 15),
        direction="long", entry_price=3000.0, exit_price=3004.0,
        actual_r=0.8, expected_r=0.1,
    ))
```

**Step 2: Run to verify FAIL**
```bash
python -m pytest tests/test_trading_app/test_performance_monitor.py -xvs
```
Expected: FAIL — PerformanceMonitor takes no kwarg `db_path`

**Step 3: Modify `performance_monitor.py`**

Add imports at top:
```python
import duckdb
from pathlib import Path
```

Change `__init__` signature:
```python
def __init__(self, strategies: list[PortfolioStrategy], db_path: Path | None = None):
    self._db_path = db_path
    # ... rest unchanged ...
```

In `record_trade()`, after `self._trades.append(record)`:
```python
        if self._db_path is not None:
            try:
                with duckdb.connect(str(self._db_path)) as con:
                    con.execute(
                        """INSERT INTO live_trades
                           (strategy_id, trading_day, direction, entry_price,
                            exit_price, actual_r, expected_r, slippage_pts, recorded_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [record.strategy_id, record.trading_day, record.direction,
                         record.entry_price, record.exit_price, record.actual_r,
                         record.expected_r, record.slippage_pts, record.timestamp],
                    )
            except duckdb.Error as e:
                log.error("Trade journal write failed for %s: %s", record.strategy_id, e)
```

**Step 4: Update `session_orchestrator.py` line 195**

Change:
```python
self.monitor = PerformanceMonitor(self.portfolio.strategies)
```
To:
```python
self.monitor = PerformanceMonitor(self.portfolio.strategies, db_path=GOLD_DB_PATH)
```
(`GOLD_DB_PATH` is already imported at line 27.)

**Step 5: Run to verify PASS**
```bash
python -m pytest tests/test_trading_app/test_performance_monitor.py -xvs
python -m pytest tests/test_trading_app/test_session_orchestrator.py -x -q
python pipeline/check_drift.py
```

**Step 6: Commit**
```bash
git add trading_app/live/performance_monitor.py trading_app/live/session_orchestrator.py tests/test_trading_app/test_performance_monitor.py
git commit -m "feat: persist trade records to live_trades on every fill (CRITICAL)"
```

---

## Task 3: Exit Order Retry via BrokerRouter base (HIGH)

**Files:**
- Modify: `trading_app/live/broker_base.py`
- Modify: `trading_app/live/session_orchestrator.py` line 846
- Test: `tests/test_trading_app/test_broker_base.py`

**Context:** Single `submit()` call on exit. One 5xx = position left open. Fix: add `submit_exit()` concrete method to `BrokerRouter` base — wraps `submit()` with 3 retries on 5xx/network. Both Tradovate and ProjectX inherit it free. Update orchestrator EXIT branch to call `submit_exit`.

**Step 1: Write failing test**
```python
# tests/test_trading_app/test_broker_base.py — add:
import requests
from unittest.mock import MagicMock, patch

def test_submit_exit_retries_on_5xx():
    from trading_app.live.broker_base import BrokerRouter

    class ConcreteRouter(BrokerRouter):
        def build_order_spec(self, *a, **kw): return {}
        def submit(self, spec): ...  # overridden by mock
        def build_exit_spec(self, *a, **kw): return {}
        def cancel(self, order_id): pass
        def supports_native_brackets(self): return False

    router = ConcreteRouter(account_id=1, auth=None)
    call_count = 0

    def fake_submit(spec):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            err = requests.HTTPError()
            err.response = MagicMock(status_code=503)
            raise err
        return {"order_id": 42, "status": "submitted"}

    router.submit = fake_submit
    result = router.submit_exit({}, max_retries=3, backoff_s=0)
    assert result["order_id"] == 42
    assert call_count == 3

def test_submit_exit_no_retry_on_4xx():
    # 4xx = our fault, must not retry
    ...
```

**Step 2: Run to verify FAIL** (`BrokerRouter` has no `submit_exit`)

**Step 3: Add `submit_exit()` to `broker_base.py`** after `query_order_status`:
```python
    def submit_exit(self, spec: dict, max_retries: int = 3, backoff_s: float = 1.0) -> dict:
        """Submit exit order with retry on 5xx/network errors.

        Exits are always market orders — retry is safe (no double-fill risk).
        4xx errors are not retried (client error, fix the request).
        """
        import time as _time
        import requests as _requests

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return self.submit(spec)
            except _requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status < 500:
                    raise  # 4xx = don't retry
                last_exc = e
                log.warning(
                    "Exit order attempt %d/%d failed (HTTP %s) — retrying in %.0fs",
                    attempt + 1, max_retries, status, backoff_s,
                )
            except _requests.RequestException as e:
                last_exc = e
                log.warning("Exit order attempt %d/%d network error: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                _time.sleep(backoff_s)
        raise RuntimeError(f"Exit order failed after {max_retries} attempts: {last_exc}")
```

Add `import logging; log = logging.getLogger(__name__)` at top of `broker_base.py` if not present.

**Step 4: Update `session_orchestrator.py` line 846**

Change:
```python
result = await loop.run_in_executor(None, self.order_router.submit, exit_spec)
```
To:
```python
result = await loop.run_in_executor(None, self.order_router.submit_exit, exit_spec)
```

**Step 5: Run to verify PASS + gates**
```bash
python -m pytest tests/test_trading_app/test_broker_base.py -xvs
python -m pytest tests/test_trading_app/test_session_orchestrator.py -x -q
python pipeline/check_drift.py
```

**Step 6: Commit**
```bash
git add trading_app/live/broker_base.py trading_app/live/session_orchestrator.py tests/test_trading_app/test_broker_base.py
git commit -m "feat: add submit_exit() with 3-attempt retry to BrokerRouter base (HIGH)"
```

---

## Task 4: Fix pysignalr Stop (HIGH)

**Files:**
- Modify: `trading_app/live/projectx/data_feed.py` lines 132-140
- Test: `tests/test_trading_app/test_projectx_feed.py`

**Context:** `_stop_file_watcher()` sets flag and returns, but `await client.run()` keeps blocking. Fix: wrap `client.run()` in a task, use `asyncio.wait(FIRST_COMPLETED)` so the stop flag cancels the run.

**Step 1: Write failing test**
```python
# In test_projectx_feed.py — add:
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

async def test_stop_flag_exits_pysignalr_run():
    from trading_app.live.projectx.data_feed import ProjectXDataFeed
    feed = ProjectXDataFeed.__new__(ProjectXDataFeed)
    feed._stop_requested = False
    feed._on_quote = lambda *a: None
    feed._on_trade = lambda *a: None
    feed._on_connected_async = AsyncMock()
    feed.auth = MagicMock()
    feed.auth.get_token.return_value = "tok"

    async def forever():
        await asyncio.sleep(10)

    mock_client = MagicMock()
    mock_client.run = forever
    mock_client.on = MagicMock()
    mock_client.on_open = MagicMock()

    async def set_stop():
        await asyncio.sleep(0.05)
        feed._stop_requested = True

    with patch("pysignalr.client.SignalRClient", return_value=mock_client):
        await asyncio.wait_for(
            asyncio.gather(feed._run_pysignalr("MGCJ5"), set_stop()),
            timeout=1.0,
        )
```

**Step 2: Run to verify FAIL** (times out — `client.run()` never cancelled)

**Step 3: Fix `_run_pysignalr` lines 132-140**

Replace:
```python
                stop_task = asyncio.create_task(self._stop_file_watcher())
                try:
                    await client.run()
                finally:
                    stop_task.cancel()

                log.info("Feed closed cleanly for %s", symbol)
                return
```
With:
```python
                run_task = asyncio.create_task(client.run())
                stop_task = asyncio.create_task(self._stop_file_watcher())
                try:
                    await asyncio.wait(
                        [run_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for t in (run_task, stop_task):
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

                if self._stop_requested:
                    log.info("Feed stopped cleanly via stop flag for %s", symbol)
                    return
                log.info("Feed closed cleanly for %s", symbol)
                return
```

**Step 4: Run to verify PASS + gates + commit**
```bash
python -m pytest tests/test_trading_app/test_projectx_feed.py -x -q
python pipeline/check_drift.py
git add trading_app/live/projectx/data_feed.py tests/test_trading_app/test_projectx_feed.py
git commit -m "fix: cancel client.run() when stop flag set in pysignalr backend (HIGH)"
```

---

## Task 5: Webhook Position Limit Guard (HIGH)

**Files:**
- Modify: `trading_app/live/webhook_server.py`
- Test: `tests/test_trading_app/test_webhook_server.py`

**Context:** Webhook bypasses all risk management — no position limits. Fix: module-level `_OPEN_POSITIONS` dict, cap at `MAX_OPEN_POSITIONS` (env, default 1). Entry blocked when at cap. Exit always allowed.

**Step 1: Write failing test**
```python
# In test_webhook_server.py — add:
def test_entry_blocked_when_position_open(client):
    from trading_app.live import webhook_server as ws
    ws._OPEN_POSITIONS.clear()
    ws._OPEN_POSITIONS["MGC"] = 1

    resp = client.post("/trade", json={
        "instrument": "MGC", "direction": "long", "action": "entry",
        "qty": 1, "secret": ws.WEBHOOK_SECRET,
    })
    assert resp.status_code == 429
    assert "position limit" in resp.json()["detail"].lower()
```

**Step 2: Run to verify FAIL**

**Step 3: Add to `webhook_server.py`** after rate-limit constants:
```python
MAX_OPEN_POSITIONS = int(os.environ.get("WEBHOOK_MAX_POSITIONS", "1"))
_OPEN_POSITIONS: dict[str, int] = {}

def _check_position_limit(req: "TradeRequest") -> None:
    if req.action != "entry":
        return
    if _OPEN_POSITIONS.get(req.instrument, 0) >= MAX_OPEN_POSITIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Position limit: {req.instrument} already has {_OPEN_POSITIONS.get(req.instrument, 0)} open position(s)",
        )
```

In `/trade` route, after `_check_rate_limit()`:
```python
    _check_position_limit(req)
```

After successful order submission, update counter:
```python
    if req.action == "entry":
        _OPEN_POSITIONS[req.instrument] = _OPEN_POSITIONS.get(req.instrument, 0) + 1
    elif req.action == "exit":
        _OPEN_POSITIONS[req.instrument] = max(0, _OPEN_POSITIONS.get(req.instrument, 0) - 1)
```

**Step 4: Run to verify PASS + gates + commit**
```bash
python -m pytest tests/test_trading_app/test_webhook_server.py -x -q
python pipeline/check_drift.py
git add trading_app/live/webhook_server.py tests/test_trading_app/test_webhook_server.py
git commit -m "feat: add per-instrument position limit to webhook server (HIGH)"
```

---

## Task 6: BarAggregator Out-of-Order Tick Protection (MEDIUM)

**Files:**
- Modify: `trading_app/live/bar_aggregator.py`
- Test: `tests/test_trading_app/test_bar_aggregator.py`

**Step 1: Write failing test**
```python
def test_out_of_order_tick_dropped():
    from datetime import datetime, UTC
    from trading_app.live.bar_aggregator import BarAggregator
    agg = BarAggregator()
    t1 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 3, 15, 10, 1, 0, tzinfo=UTC)
    t_old = datetime(2026, 3, 15, 9, 59, 0, tzinfo=UTC)

    agg.on_tick(3000.0, 1, t1)
    agg.on_tick(3001.0, 1, t2)  # opens 10:01 bar
    result = agg.on_tick(2900.0, 1, t_old)  # out-of-order — must drop
    assert result is None
    assert agg._current.low > 2900.0  # 2900 not incorporated
```

**Step 2: Run to verify FAIL**

**Step 3: Add guard to `bar_aggregator.py` in `on_tick()`**

After `tick_minute = ts.replace(second=0, microsecond=0)`, add:
```python
        if self._bar_minute is not None and tick_minute < self._bar_minute:
            log.warning("Dropped out-of-order tick: %s < current bar %s", tick_minute, self._bar_minute)
            return None
```

Add at top: `import logging; log = logging.getLogger(__name__)`

**Step 4: Run + commit**
```bash
python -m pytest tests/test_trading_app/test_bar_aggregator.py -x -q
git add trading_app/live/bar_aggregator.py tests/test_trading_app/test_bar_aggregator.py
git commit -m "fix: drop out-of-order ticks in BarAggregator (MEDIUM)"
```

---

## Task 7: Rollover Close — Add Notify on Failure (MEDIUM)

**Files:**
- Modify: `trading_app/live/session_orchestrator.py` (2 locations)

**Context:** Task 3 already adds retry at the broker level. This task adds `_notify()` to the two rollover catch blocks that currently only `log.error` silently — so operator gets alerted when an EOD close fails after retries.

**Location 1** — `_on_rollover` (around line 477):
Change:
```python
            except Exception as e:
                log.error(
                    "Rollover close failed for %s (%s) — position may remain open: %s",
                    event.strategy_id, event.event_type, e,
                )
```
To:
```python
            except Exception as e:
                msg = f"ROLLOVER CLOSE FAILED: {event.strategy_id} ({event.event_type}) — position may remain open: {e}"
                log.error(msg)
                self._notify(msg)
```

**Location 2** — `_close_all` (around line 1180):
Same change pattern — add `self._notify(msg)` after `log.error`.

**Step: Run + commit**
```bash
python -m pytest tests/test_trading_app/test_session_orchestrator.py -x -q
python pipeline/check_drift.py
git add trading_app/live/session_orchestrator.py
git commit -m "fix: notify operator when rollover close fails (MEDIUM)"
```

---

## Task 8: Tradovate Feed — Prefer Trade Price Over bidPrice (MEDIUM)

**Files:**
- Modify: `trading_app/live/tradovate/data_feed.py` line 194
- Create: `tests/test_trading_app/test_tradovate_data_feed.py`

**Step 1: Write failing test**
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock

async def test_handle_frame_prefers_trade_price():
    from trading_app.live.tradovate.data_feed import TradovateDataFeed
    feed = TradovateDataFeed.__new__(TradovateDataFeed)
    feed._agg = MagicMock()
    feed._agg.on_tick.return_value = None
    feed._last_quote_at = None
    feed._quote_count = 0
    feed.on_bar = AsyncMock()

    frame = {"d": {"quotes": [{"price": 3001.0, "bidPrice": 2999.0}]}}
    await feed._handle_frame(frame, "MGCJ5")

    used_price = feed._agg.on_tick.call_args[0][0]
    assert used_price == 3001.0  # trade price, not bid
```

**Step 2: Run to verify FAIL**

**Step 3: Fix `_handle_frame` in `tradovate/data_feed.py`**

Change:
```python
            price = q.get("bidPrice")
            if price is None:
                price = q.get("price")
```
To:
```python
            price = q.get("price") or q.get("midPrice") or q.get("bidPrice")
```

**Step 4: Run + commit**
```bash
python -m pytest tests/test_trading_app/test_tradovate_data_feed.py -xvs
git add trading_app/live/tradovate/data_feed.py tests/test_trading_app/test_tradovate_data_feed.py
git commit -m "fix: prefer trade price over bidPrice in Tradovate data feed (MEDIUM)"
```

---

## Execution Order

1 → 2 (schema before writes) → 3 → 4 → 5 → 6 → 7 → 8

All other tasks independent once schema is done.
