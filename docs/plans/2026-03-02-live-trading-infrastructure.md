# Live Trading Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing backtesting/simulation engine to live markets via Tradovate, with automated daily data backfill and CUSUM-based monitoring.

**Architecture:** Five sequential phases: (A) automated daily data pipeline, (B) Tradovate live data feed → real-time bar builder, (C) order router mapping ExecutionEngine events to Tradovate REST calls, (D) monitoring & alerting, (E) session orchestrator that ties it all together. Each phase is independently testable and deployable.

**Tech Stack:** Python, Tradovate REST + WebSocket API, DuckDB (gold.db), existing ExecutionEngine/RiskManager/LiveConfig, `websockets` library, Windows Task Scheduler for automation.

---

## Background — What Exists vs What's Missing

### Already built (DO NOT REBUILD)
- `trading_app/execution_engine.py` — bar-by-bar state machine. Key API:
  - `on_trading_day_start(trading_day: date)` — must call once at start of each trading day
  - `on_bar(bar: dict) -> list[TradeEvent]` — bar dict needs keys: `ts_utc, open, high, low, close, volume`
  - `TradeEvent` fields: `event_type` ("ENTRY"/"EXIT"/"SCRATCH"/"REJECT"), `strategy_id`, `timestamp`, `price`, `direction`, `contracts`, `reason`
  - `TradeEvent.price` = entry fill price on ENTRY events, exit fill price on EXIT events
- `trading_app/risk_manager.py` — `RiskLimits`, `RiskManager`. RiskLimits fields: `max_daily_loss_r`, `max_concurrent_positions`, `max_per_orb_positions`, `max_daily_trades`, `drawdown_warning_r`
- `trading_app/live_config.py` — `build_live_portfolio(db_path, instrument, ...) -> tuple[Portfolio, list[str]]` — returns (Portfolio, notes). NOT just Portfolio.
- `trading_app/portfolio.py` — `Portfolio` (has `.strategies: list[PortfolioStrategy]`, `.max_daily_loss_r`, `.max_concurrent_positions`). `PortfolioStrategy` (has `.strategy_id`, `.expectancy_r`, `.entry_model`, `.instrument`)
- `trading_app/market_state.py` — `MarketState`, `OrbSnapshot`. `from_trading_day()` reads from DB (historical). `update_orb()` for live updates.
- `pipeline/dst.py` — `DYNAMIC_ORB_RESOLVERS: dict[str, Callable[[date], tuple[int, int]]]`. Each resolver returns `(hour, minute)` in **Brisbane time** (UTC+10, no DST). To get UTC: subtract 10 hours.
- `pipeline/asset_configs.py` — `ACTIVE_ORB_INSTRUMENTS: list[str]` (["M2K","MBT","MES","MGC","MNQ"]). `ASSET_CONFIGS: dict[str, dict]` — dict of dicts (not objects). Key `"source_symbol"` does NOT exist in the current config dicts; all instruments use their own symbol for ingestion.
- `pipeline/paths.py` — `GOLD_DB_PATH: Path`
- `pipeline/ingest_dbn.py`, `pipeline/build_bars_5m.py`, `pipeline/build_daily_features.py` — all support `--instrument`, `--start`, `--end` flags
- `trading_app/outcome_builder.py` — supports `--instrument`, `--start`, `--end`, `--force`
- `trading_app/paper_trader.py` — historical replay, NOT live

### Missing (this plan builds)
```
Live data source (Tradovate WS)
    ↓
Real-time bar aggregator (ticks → 1-min bars, ts_utc key)   ← Phase B
    ↓
Live MarketState (ORB detection from live bars)              ← Phase B
    ↓
ExecutionEngine.on_bar() ← EXISTS (takes bar dict with ts_utc)
    ↓
Order router → Tradovate REST (maps event_type→order)        ← Phase C
    ↓
Fill handler → side-table (strategy_id → entry price)        ← Phase C
    ↓
Live P&L + CUSUM monitoring (per PortfolioStrategy)          ← Phase D
    ↓
Daily session orchestrator                                    ← Phase E
    ↓
Automated daily backfill (EOD)                               ← Phase A
```

---

## TopstepX / Tradovate Notes

- **TopstepX**: Automated trading is **explicitly allowed** on Combine and Funded accounts. No separate API — routes through Tradovate. Prohibited strategies are HFT/manipulation/account-stacking, not systematic ORB.
- **Tradovate**: Full REST + WebSocket API. Market data: `md/subscribeQuote`. Orders: `order/placeOrder`. Auth: OAuth access token (~15-min TTL, renewable).
- **Tradovate market data WebSocket**: `wss://md-demo.tradovate.com/v1/websocket` (demo) / `wss://md.tradovate.com/v1/websocket` (live)
- **Contract symbols**: MGC=`MGCM6`, MNQ=`MNQM6`, etc. Roll quarterly — use `/contract/find` to get current front-month.
- **Rate limits**: Use subscriptions not polling. Heartbeat every 2.5s required.

---

## Phase A: Automated Daily Data Backfill

**Goal:** Each evening after markets close, automatically bring gold.db up to date through outcomes. No auto-trigger of discovery/validation — that's monthly/manual.

**Files to create:**
- `pipeline/daily_backfill.py`
- `tests/test_pipeline/test_daily_backfill.py`

**Files to read first:**
- `pipeline/paths.py` — GOLD_DB_PATH
- `pipeline/asset_configs.py` — ACTIVE_ORB_INSTRUMENTS

---

### Task A1: Freshness check utility

**Step 1: Write failing test**

```python
# tests/test_pipeline/test_daily_backfill.py
import duckdb, tempfile, os
from datetime import date
from pipeline.daily_backfill import get_last_ingested_date, is_up_to_date

def _make_db(rows=None):
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    con = duckdb.connect(f.name)
    con.execute("CREATE TABLE bars_1m (ts_event TIMESTAMPTZ, symbol VARCHAR)")
    if rows:
        for row in rows:
            con.execute("INSERT INTO bars_1m VALUES (?, ?)", row)
    con.close()
    return f.name

def test_get_last_ingested_date_empty():
    db = _make_db()
    try:
        assert get_last_ingested_date(db, "MGC") is None
    finally:
        os.unlink(db)

def test_get_last_ingested_date_with_data():
    db = _make_db([("2026-02-28 15:00:00+00", "MGC")])
    try:
        result = get_last_ingested_date(db, "MGC")
        assert result.date().isoformat() == "2026-02-28"
    finally:
        os.unlink(db)

def test_is_up_to_date_true():
    db = _make_db([("2026-03-01 15:00:00+00", "MGC")])
    try:
        assert is_up_to_date(db, "MGC", date(2026, 3, 1)) is True
    finally:
        os.unlink(db)

def test_is_up_to_date_false():
    db = _make_db([("2026-02-28 15:00:00+00", "MGC")])
    try:
        assert is_up_to_date(db, "MGC", date(2026, 3, 1)) is False
    finally:
        os.unlink(db)
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_pipeline/test_daily_backfill.py -x -q
```
Expected: `ImportError: cannot import name 'get_last_ingested_date'`

**Step 3: Implement**

```python
# pipeline/daily_backfill.py
"""
Nightly data backfill: ingest new bars → 5m bars → daily features → outcomes.
Does NOT trigger strategy discovery or validation (those are periodic/manual).

Usage:
    python pipeline/daily_backfill.py --instrument MGC
    python pipeline/daily_backfill.py  # all active instruments
"""
import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH


def get_last_ingested_date(db_path: str, symbol: str):
    """Return the datetime of the most recent bar for symbol, or None."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            "SELECT MAX(ts_event) FROM bars_1m WHERE symbol = ?", [symbol]
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    finally:
        con.close()


def is_up_to_date(db_path: str, symbol: str, as_of: date) -> bool:
    """True if bars_1m has data through as_of date for symbol."""
    last = get_last_ingested_date(db_path, symbol)
    if last is None:
        return False
    return last.date() >= as_of


def _run(cmd: list[str], desc: str) -> None:
    """Run subprocess, raise on non-zero exit."""
    print(f"\n▶ {desc}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {desc} (exit {result.returncode})")
    print(f"✓ {desc}")
```

**Step 4: Run tests**
```bash
python -m pytest tests/test_pipeline/test_daily_backfill.py -x -q
python pipeline/check_drift.py
```

**Step 5: Commit**
```bash
git add pipeline/daily_backfill.py tests/test_pipeline/test_daily_backfill.py
git commit -m "feat: daily_backfill freshness check utilities"
```

---

### Task A2: Backfill orchestration

**Step 1: Write failing test**

```python
# append to tests/test_pipeline/test_daily_backfill.py
from unittest.mock import patch
from pipeline.daily_backfill import run_backfill_for_instrument

def test_run_backfill_skips_when_current():
    db = _make_db([("2026-03-01 15:00:00+00", "MGC")])
    try:
        with patch("pipeline.daily_backfill._run") as mock_run:
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
            mock_run.assert_not_called()
    finally:
        os.unlink(db)

def test_run_backfill_calls_pipeline_when_stale():
    db = _make_db()  # empty
    try:
        with patch("pipeline.daily_backfill._run") as mock_run:
            run_backfill_for_instrument("MGC", db_path=db, as_of=date(2026, 3, 1))
            assert mock_run.call_count >= 4  # ingest, 5m, daily_features, outcomes
    finally:
        os.unlink(db)
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_pipeline/test_daily_backfill.py::test_run_backfill_skips_when_current -x -q
```

**Step 3: Implement** — add to `pipeline/daily_backfill.py`:

```python
def run_backfill_for_instrument(
    instrument: str,
    db_path: str | None = None,
    as_of: date | None = None,
) -> None:
    """Run incremental backfill for one instrument. Idempotent."""
    db = db_path or str(GOLD_DB_PATH)
    target = as_of or (date.today() - timedelta(days=1))  # last complete session

    if is_up_to_date(db, instrument, target):
        print(f"✓ {instrument}: already current through {target}, skipping")
        return

    last = get_last_ingested_date(db, instrument)
    start = (last.date() + timedelta(days=1)).isoformat() if last else "2021-01-01"
    end = target.isoformat()
    print(f"▶ {instrument}: backfilling {start} → {end}")

    py = sys.executable
    _run([py, "pipeline/ingest_dbn.py",
          "--instrument", instrument, "--start", start, "--end", end],
         f"ingest_dbn {instrument}")
    _run([py, "pipeline/build_bars_5m.py",
          "--instrument", instrument, "--start", start, "--end", end],
         f"build_bars_5m {instrument}")
    _run([py, "pipeline/build_daily_features.py",
          "--instrument", instrument, "--start", start, "--end", end],
         f"build_daily_features {instrument}")
    # Outcomes: incremental only. DO NOT trigger discovery/validation.
    _run([py, "trading_app/outcome_builder.py",
          "--instrument", instrument, "--start", start, "--end", end],
         f"outcome_builder {instrument}")


def main():
    parser = argparse.ArgumentParser(description="Nightly data backfill")
    parser.add_argument("--instrument", help="Single instrument (default: all active)")
    parser.add_argument("--db-path", help="Override DB path")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else list(ACTIVE_ORB_INSTRUMENTS)
    db = args.db_path or str(GOLD_DB_PATH)
    errors = []
    for inst in instruments:
        try:
            run_backfill_for_instrument(inst, db_path=db)
        except Exception as e:
            print(f"✗ {inst}: {e}")
            errors.append(inst)
    if errors:
        print(f"\n✗ Backfill failed for: {errors}")
        sys.exit(1)
    print("\n✓ Backfill complete")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests + drift check**
```bash
python -m pytest tests/test_pipeline/test_daily_backfill.py -x -q
python pipeline/check_drift.py
```

**Step 5: Commit**
```bash
git add pipeline/daily_backfill.py tests/test_pipeline/test_daily_backfill.py
git commit -m "feat: daily incremental backfill orchestrator (no auto-discovery)"
```

---

### Task A3: Windows Task Scheduler setup

```python
# scripts/setup_daily_backfill.py
"""
Register Windows Task Scheduler job to run daily backfill at 7:00 AM Brisbane.
Brisbane = UTC+10, no DST. Run once: python scripts/setup_daily_backfill.py
"""
import subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
PYTHON = sys.executable
SCRIPT = PROJECT / "pipeline" / "daily_backfill.py"
TASK_NAME = "canompx3-daily-backfill"

cmd = ["schtasks", "/create", "/f",
       "/tn", TASK_NAME,
       "/tr", f'"{PYTHON}" "{SCRIPT}"',
       "/sc", "daily", "/st", "07:00", "/rl", "HIGHEST"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"✓ Created '{TASK_NAME}'" if r.returncode == 0 else f"✗ {r.stderr}")
```

```bash
git add scripts/setup_daily_backfill.py
git commit -m "feat: Windows Task Scheduler setup for daily 7am backfill"
```

---

## Phase B: Tradovate Live Data Feed

**Goal:** Connect to Tradovate WebSocket, receive real-time quotes, aggregate into 1-minute OHLCV bars with `ts_utc` key (matching what ExecutionEngine.on_bar() expects), build live `MarketState`.

**Files to create:**
- `trading_app/live/__init__.py`
- `trading_app/live/tradovate_auth.py`
- `trading_app/live/bar_aggregator.py`
- `trading_app/live/data_feed.py`
- `trading_app/live/live_market_state.py`
- `tests/test_trading_app/test_bar_aggregator.py`
- `tests/test_trading_app/test_live_market_state.py`

**Files to read first:**
- `trading_app/execution_engine.py` lines 260-290 — `on_bar()` and `on_trading_day_start()` signatures
- `pipeline/dst.py` — `DYNAMIC_ORB_RESOLVERS` structure (each entry: `callable(date) -> (hour, minute)` in Brisbane time)

---

### Task B1: Auth module

**Files:** Create `trading_app/live/__init__.py` (empty) and `trading_app/live/tradovate_auth.py`

Add to `.env`:
```
TRADOVATE_USER=your_email
TRADOVATE_PASS=your_password
TRADOVATE_APP_ID=Sample App
TRADOVATE_APP_VERSION=1.0
TRADOVATE_CID=your_cid
TRADOVATE_SEC=your_secret
```

```python
# trading_app/live/tradovate_auth.py
"""
Tradovate OAuth token management. Auto-renews before expiry.
Reads credentials from .env (loaded via python-dotenv).
"""
import os, time
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

LIVE_BASE = "https://live.tradovate.com/v1"
DEMO_BASE = "https://demo.tradovate.com/v1"


class TradovateAuth:
    def __init__(self, demo: bool = True):
        self.base = DEMO_BASE if demo else LIVE_BASE
        self._token: str | None = None
        self._expires_at: float = 0

    def get_token(self) -> str:
        if time.time() < self._expires_at - 60:
            return self._token
        return self._refresh()

    def _refresh(self) -> str:
        resp = requests.post(f"{self.base}/auth/accesstokenrequest", json={
            "name": os.environ["TRADOVATE_USER"],
            "password": os.environ["TRADOVATE_PASS"],
            "appId": os.environ["TRADOVATE_APP_ID"],
            "appVersion": os.environ.get("TRADOVATE_APP_VERSION", "1.0"),
            "cid": int(os.environ["TRADOVATE_CID"]),
            "sec": os.environ["TRADOVATE_SEC"],
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token = data["accessToken"]
        exp = datetime.fromisoformat(data["expirationTime"].replace("Z", "+00:00"))
        self._expires_at = exp.timestamp()
        return self._token

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}
```

```bash
git add trading_app/live/ && git commit -m "feat: Tradovate auth module with auto-renewal"
```

---

### Task B2: 1-minute bar aggregator

**IMPORTANT:** The bar dict must use key `ts_utc` (not `ts_event`) — that is what `ExecutionEngine.on_bar()` reads.

**Step 1: Write failing tests**

```python
# tests/test_trading_app/test_bar_aggregator.py
from datetime import datetime, timezone
from trading_app.live.bar_aggregator import BarAggregator, Bar

def _ts(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 3, 3, 10, minute, second, tzinfo=timezone.utc)

def test_first_tick_opens_bar_returns_none():
    agg = BarAggregator()
    assert agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 5)) is None

def test_tick_crossing_minute_boundary_closes_previous_bar():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 5))
    agg.on_tick(price=2001.0, volume=2, ts=_ts(0, 30))
    agg.on_tick(price=1999.0, volume=1, ts=_ts(0, 59))
    completed = agg.on_tick(price=2002.0, volume=1, ts=_ts(1, 1))
    assert completed is not None
    assert completed.open == 2000.0
    assert completed.high == 2001.0
    assert completed.low == 1999.0
    assert completed.close == 1999.0
    assert completed.volume == 4

def test_bar_ts_utc_is_minute_start():
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(5, 3))
    completed = agg.on_tick(price=2001.0, volume=1, ts=_ts(6, 0))
    assert completed.ts_utc.minute == 5
    assert completed.ts_utc.second == 0

def test_bar_as_dict_has_ts_utc_key():
    """ExecutionEngine.on_bar() requires key 'ts_utc'."""
    agg = BarAggregator()
    agg.on_tick(price=2000.0, volume=1, ts=_ts(0, 0))
    bar = agg.on_tick(price=2001.0, volume=1, ts=_ts(1, 0))
    d = bar.as_dict()
    assert "ts_utc" in d
    assert "ts_event" not in d  # wrong key name
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_trading_app/test_bar_aggregator.py -x -q
```

**Step 3: Implement**

```python
# trading_app/live/bar_aggregator.py
"""
Aggregates real-time ticks into 1-minute OHLCV bars.
Bar.ts_utc = start of minute in UTC (matches ExecutionEngine.on_bar() key requirement).
"""
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Bar:
    ts_utc: datetime   # UTC, truncated to minute start — key name matches on_bar() expectation
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str = ""

    def as_dict(self) -> dict:
        """Return dict suitable for ExecutionEngine.on_bar()."""
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BarAggregator:
    def __init__(self):
        self._current: Bar | None = None
        self._bar_minute: datetime | None = None

    def on_tick(self, price: float, volume: int, ts: datetime) -> Bar | None:
        """Process one tick. Returns completed Bar when minute boundary crossed."""
        tick_minute = ts.replace(second=0, microsecond=0)

        if self._current is None:
            self._open_bar(price, volume, tick_minute)
            return None

        if tick_minute == self._bar_minute:
            self._current.high = max(self._current.high, price)
            self._current.low = min(self._current.low, price)
            self._current.close = price
            self._current.volume += volume
            return None

        completed = self._current
        self._open_bar(price, volume, tick_minute)
        return completed

    def _open_bar(self, price: float, volume: int, minute: datetime) -> None:
        self._bar_minute = minute
        self._current = Bar(
            ts_utc=minute,
            open=price, high=price, low=price, close=price, volume=volume,
        )

    def flush(self) -> Bar | None:
        """Force-close current bar (call at session end)."""
        bar = self._current
        self._current = None
        self._bar_minute = None
        return bar
```

**Step 4: Run tests**
```bash
python -m pytest tests/test_trading_app/test_bar_aggregator.py -x -q
```

**Step 5: Commit**
```bash
git add trading_app/live/bar_aggregator.py tests/test_trading_app/test_bar_aggregator.py
git commit -m "feat: 1-min bar aggregator producing ts_utc key for ExecutionEngine"
```

---

### Task B3: Tradovate WebSocket data feed

```python
# trading_app/live/data_feed.py
"""
Connects to Tradovate market data WebSocket.
Produces 1-minute bars via BarAggregator.
Calls on_bar(bar: Bar) callback for each completed bar.
"""
import asyncio, json
from collections.abc import Callable
from datetime import datetime, timezone
import websockets
from .bar_aggregator import Bar, BarAggregator
from .tradovate_auth import TradovateAuth

MD_WS_LIVE = "wss://md.tradovate.com/v1/websocket"
MD_WS_DEMO = "wss://md-demo.tradovate.com/v1/websocket"


class DataFeed:
    def __init__(self, auth: TradovateAuth, on_bar: Callable[[Bar], None], demo: bool = True):
        self.auth = auth
        self.on_bar = on_bar
        self.demo = demo
        self._agg = BarAggregator()

    async def run(self, symbol: str) -> None:
        url = MD_WS_DEMO if self.demo else MD_WS_LIVE
        async with websockets.connect(url) as ws:
            # Auth handshake
            await ws.send(json.dumps({
                "url": "auth/accesstokenrequest",
                "body": {"token": self.auth.get_token()},
            }))
            resp = json.loads(await ws.recv())
            if resp.get("s") != 200:
                raise RuntimeError(f"Auth failed: {resp}")

            # Subscribe to real-time quotes
            await ws.send(json.dumps({"url": "md/subscribeQuote", "body": {"symbol": symbol}}))

            async def heartbeat():
                while True:
                    await asyncio.sleep(2.5)
                    await ws.send("[]")
            asyncio.create_task(heartbeat())

            async for message in ws:
                if not message or message == "[]":
                    continue
                try:
                    frames = json.loads(message)
                except json.JSONDecodeError:
                    continue
                for frame in (frames if isinstance(frames, list) else [frames]):
                    self._handle_frame(frame, symbol)

    def _handle_frame(self, frame: dict, symbol: str) -> None:
        if not isinstance(frame, dict):
            return
        for q in frame.get("d", {}).get("quotes", []):
            price = q.get("bidPrice") or q.get("price")
            if price is None:
                continue
            bar = self._agg.on_tick(float(price), 1, datetime.now(timezone.utc))
            if bar is not None:
                bar.symbol = symbol
                self.on_bar(bar)
```

```bash
git add trading_app/live/data_feed.py
git commit -m "feat: Tradovate WebSocket data feed with bar aggregation"
```

---

### Task B4: Live ORB builder (from live bars)

**CRITICAL NOTE on SESSION_CATALOG:** `DYNAMIC_ORB_RESOLVERS[label](date)` returns `(hour, minute)` in **Brisbane local time** (UTC+10, no DST). Must convert to UTC datetime before comparing against bar `ts_utc`.

**Step 1: Write failing tests**

```python
# tests/test_trading_app/test_live_market_state.py
from datetime import date, datetime, timezone
from trading_app.live.bar_aggregator import Bar
from trading_app.live.live_market_state import LiveORBBuilder

def _bar(hour_utc: int, minute_utc: int, high: float, low: float, symbol="MGC") -> Bar:
    ts = datetime(2026, 3, 3, hour_utc, minute_utc, 0, tzinfo=timezone.utc)
    mid = (high + low) / 2
    return Bar(ts_utc=ts, open=mid, high=high, low=low, close=mid, volume=10, symbol=symbol)

def test_orb_incomplete_before_enough_bars():
    # CME_REOPEN Brisbane 09:00 = UTC 23:00 previous day
    # For trading_day 2026-03-03, CME_REOPEN starts at 2026-03-02 23:00 UTC
    builder = LiveORBBuilder(instrument="MGC", trading_day=date(2026, 3, 3))
    # Add 4 bars (need 5 for 5-min ORB)
    for m in range(4):
        builder.on_bar(_bar(23, m, high=2001.0 + m, low=1999.0 - m))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is None

def test_orb_complete_after_five_bars():
    builder = LiveORBBuilder(instrument="MGC", trading_day=date(2026, 3, 3))
    for m in range(5):
        builder.on_bar(_bar(23, m, high=2001.0 + m, low=1999.0 - m))
    orb = builder.get_orb("CME_REOPEN", orb_minutes=5)
    assert orb is not None
    assert orb.high == 2005.0  # max of (2001,2002,2003,2004,2005)
    assert orb.low == 1999.0   # min of (1999,1998,1997,1996,1995)
    assert orb.complete is True
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_trading_app/test_live_market_state.py -x -q
```

**Step 3: Implement**

```python
# trading_app/live/live_market_state.py
"""
Builds ORB ranges incrementally from live 1-minute bars.
Converts Brisbane session times to UTC for bar comparisons.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from pipeline.dst import DYNAMIC_ORB_RESOLVERS
from trading_app.live.bar_aggregator import Bar

_BRISBANE = ZoneInfo("Australia/Brisbane")
_UTC = timezone.utc


def _session_start_utc(session_label: str, trading_day: date) -> Optional[datetime]:
    """
    Get session start as UTC datetime for a given Brisbane trading_day.
    DYNAMIC_ORB_RESOLVERS[label](date) returns (hour, minute) in Brisbane local time.
    Brisbane = UTC+10, no DST.
    """
    resolver = DYNAMIC_ORB_RESOLVERS.get(session_label)
    if resolver is None:
        return None
    try:
        bris_h, bris_m = resolver(trading_day)
    except Exception:
        return None

    # Determine calendar date in Brisbane.
    # Sessions before 09:00 Brisbane belong to the PREVIOUS Brisbane date
    # (e.g. CME_REOPEN at 09:00 Brisbane = that same trading_day)
    # Sessions that cross midnight in Brisbane are on trading_day itself.
    # trading_day is already the Brisbane trading day, so we use it directly.
    # For sessions like NYSE_OPEN (00:30 Brisbane), the calendar date is trading_day
    # (not trading_day - 1), because midnight-crossing sessions are assigned to
    # the trading_day they belong to per the pipeline convention.
    bris_dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                       bris_h, bris_m, 0, tzinfo=_BRISBANE)
    return bris_dt.astimezone(_UTC)


@dataclass
class LiveORB:
    high: float
    low: float
    size: float
    bars_count: int
    complete: bool


class LiveORBBuilder:
    def __init__(self, instrument: str, trading_day: date):
        self.instrument = instrument
        self.trading_day = trading_day
        self._bars: list[Bar] = []

    def on_bar(self, bar: Bar) -> None:
        self._bars.append(bar)

    def get_orb(self, session_label: str, orb_minutes: int) -> Optional[LiveORB]:
        """Return completed ORB or None if not enough bars yet."""
        start_utc = _session_start_utc(session_label, self.trading_day)
        if start_utc is None:
            return None

        end_utc = start_utc + timedelta(minutes=orb_minutes)
        orb_bars = [b for b in self._bars if start_utc <= b.ts_utc < end_utc]

        if len(orb_bars) < orb_minutes:
            return None

        high = max(b.high for b in orb_bars)
        low = min(b.low for b in orb_bars)
        return LiveORB(high=high, low=low, size=round(high - low, 4),
                       bars_count=len(orb_bars), complete=True)

    def get_bars_since(self, session_label: str) -> list[Bar]:
        start = _session_start_utc(session_label, self.trading_day)
        if start is None:
            return []
        return [b for b in self._bars if b.ts_utc >= start]
```

**Step 4: Run tests**
```bash
python -m pytest tests/test_trading_app/test_live_market_state.py -x -q
```

**Step 5: Commit**
```bash
git add trading_app/live/live_market_state.py tests/test_trading_app/test_live_market_state.py
git commit -m "feat: live ORB builder with correct Brisbane→UTC session conversion"
```

---

### Task B5: Front-month contract resolver

```python
# trading_app/live/contract_resolver.py
"""Resolves instrument shortname to current Tradovate front-month symbol."""
import requests
from .tradovate_auth import TradovateAuth

DEMO_BASE = "https://demo.tradovate.com/v1"
LIVE_BASE = "https://live.tradovate.com/v1"

# Tradovate product names (base symbol, no expiry code)
PRODUCT_MAP = {"MGC": "MGC", "MNQ": "MNQ", "MES": "MES", "M2K": "M2K"}


def resolve_front_month(instrument: str, auth: TradovateAuth, demo: bool = True) -> str:
    """Return current front-month symbol, e.g. 'MGCM6'."""
    base = DEMO_BASE if demo else LIVE_BASE
    resp = requests.get(f"{base}/contract/find",
                        params={"name": PRODUCT_MAP[instrument]},
                        headers=auth.headers(), timeout=5)
    resp.raise_for_status()
    contracts = resp.json()
    if not contracts:
        raise RuntimeError(f"No contracts found for {instrument}")
    front = sorted(contracts, key=lambda c: c.get("expirationDate", ""))[0]
    return front["name"]
```

```bash
git add trading_app/live/contract_resolver.py
git commit -m "feat: front-month contract resolver from Tradovate /contract/find"
```

---

## Phase C: Order Router

**Goal:** Map `TradeEvent` objects from `ExecutionEngine` to actual Tradovate orders.

**Key facts about TradeEvent (verified):**
- `event_type`: `"ENTRY"` or `"EXIT"` (not TradeState enum — that's internal to engine)
- `price`: entry fill price on ENTRY events; exit fill price on EXIT events
- `strategy_id`: use to look up strategy details from `Portfolio.strategies`
- No `entry_model`, `pnl_r`, `expectancy_r` fields on TradeEvent — look these up from `PortfolioStrategy`

**Files to create:**
- `trading_app/live/order_router.py`
- `tests/test_trading_app/test_order_router.py`

---

### Task C1: Order router

**Step 1: Write failing tests**

```python
# tests/test_trading_app/test_order_router.py
from trading_app.live.order_router import OrderRouter, OrderSpec

def test_e1_long_generates_market_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="long", entry_model="E1",
        entry_price=2000.0, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Market"
    assert spec.action == "Buy"
    assert spec.stop_price is None

def test_e2_short_generates_stop_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="short", entry_model="E2",
        entry_price=1999.5, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Stop"
    assert spec.action == "Sell"
    assert spec.stop_price == 1999.5

def test_e3_raises_not_supported():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    import pytest
    with pytest.raises(ValueError, match="E3"):
        router.build_order_spec(
            direction="long", entry_model="E3",
            entry_price=2000.0, symbol="MGCM6", qty=1,
        )
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_trading_app/test_order_router.py -x -q
```

**Step 3: Implement**

```python
# trading_app/live/order_router.py
"""
Maps ExecutionEngine TradeEvent.event_type → Tradovate REST orders.

IMPORTANT: TradeEvent fields are: event_type, strategy_id, timestamp, price,
           direction, contracts, reason. There is NO entry_model, pnl_r, etc.
           entry_model must be looked up from Portfolio.strategies by strategy_id.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import requests
from .tradovate_auth import TradovateAuth

log = logging.getLogger(__name__)

LIVE_BASE = "https://live.tradovate.com/v1"
DEMO_BASE = "https://demo.tradovate.com/v1"


@dataclass
class OrderSpec:
    action: str              # "Buy" | "Sell"
    order_type: str          # "Market" | "Stop"
    symbol: str
    qty: int
    account_id: int
    stop_price: Optional[float] = None


@dataclass
class OrderResult:
    order_id: int
    status: str


class OrderRouter:
    def __init__(self, account_id: int, auth: Optional[TradovateAuth], demo: bool = True):
        self.account_id = account_id
        self.auth = auth
        self.base = DEMO_BASE if demo else LIVE_BASE

    def build_order_spec(
        self,
        direction: str,       # "long" | "short"
        entry_model: str,     # "E1" | "E2" — E3 not supported live (no timeout mechanism)
        entry_price: float,   # from TradeEvent.price on ENTRY events
        symbol: str,
        qty: int = 1,
    ) -> OrderSpec:
        action = "Buy" if direction == "long" else "Sell"
        if entry_model == "E1":
            return OrderSpec(action=action, order_type="Market",
                             symbol=symbol, qty=qty, account_id=self.account_id)
        elif entry_model == "E2":
            return OrderSpec(action=action, order_type="Stop", symbol=symbol, qty=qty,
                             account_id=self.account_id, stop_price=entry_price)
        else:
            raise ValueError(f"Entry model '{entry_model}' not supported for live trading. "
                             f"E3 has no timeout mechanism — use E1 or E2 only.")

    def submit(self, spec: OrderSpec) -> OrderResult:
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit live orders without TradovateAuth")
        body = {
            "accountId": spec.account_id,
            "action": spec.action,
            "symbol": spec.symbol,
            "orderQty": spec.qty,
            "orderType": spec.order_type,
            "isAutomated": True,
        }
        if spec.stop_price is not None:
            body["stopPrice"] = spec.stop_price
        resp = requests.post(f"{self.base}/order/placeOrder", json=body,
                             headers=self.auth.headers(), timeout=5)
        resp.raise_for_status()
        data = resp.json()
        order_id = data.get("orderId", -1)
        log.info("Order placed: %s %s qty=%d → orderId=%d",
                 spec.action, spec.symbol, spec.qty, order_id)
        return OrderResult(order_id=order_id, status="submitted")

    def cancel(self, order_id: int) -> None:
        if self.auth is None:
            return
        requests.post(f"{self.base}/order/cancelOrder",
                      json={"orderId": order_id},
                      headers=self.auth.headers(), timeout=5)
```

**Step 4: Run tests**
```bash
python -m pytest tests/test_trading_app/test_order_router.py -x -q
```

**Step 5: Commit**
```bash
git add trading_app/live/order_router.py tests/test_trading_app/test_order_router.py
git commit -m "feat: Tradovate order router (E1=market, E2=stop-market, E3 blocked)"
```

---

## Phase D: Monitoring & Alerting

**Goal:** Track live P&L per strategy vs backtest expectations using CUSUM drift detection.

**Key fact:** `PerformanceMonitor` takes `list[PortfolioStrategy]` (from `Portfolio.strategies`) NOT `list[LiveStrategySpec]`. `PortfolioStrategy` has `strategy_id` and `expectancy_r`. `LiveStrategySpec` does not.

---

### Task D1: CUSUM drift detector

**Step 1: Write failing tests**

```python
# tests/test_trading_app/test_cusum_monitor.py
from trading_app.live.cusum_monitor import CUSUMMonitor

def test_no_alarm_on_good_performance():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(10):
        monitor.update(+0.5)
    assert not monitor.alarm_triggered

def test_alarm_triggered_on_persistent_losses():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(20):
        monitor.update(-1.0)
    assert monitor.alarm_triggered

def test_drift_severity_positive_when_losing():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(5):
        monitor.update(-1.0)
    assert monitor.drift_severity > 0

def test_clear_resets_state():
    monitor = CUSUMMonitor(expected_r=0.3, std_r=1.0, threshold=4.0)
    for _ in range(20):
        monitor.update(-1.0)
    monitor.clear()
    assert not monitor.alarm_triggered
    assert monitor.cusum_neg == 0.0
```

**Step 2: Run to verify failure**
```bash
python -m pytest tests/test_trading_app/test_cusum_monitor.py -x -q
```

**Step 3: Implement**

```python
# trading_app/live/cusum_monitor.py
"""
CUSUM control chart for detecting strategy performance drift.
Reference: arXiv:1509.01570 "Real-time financial surveillance via quickest change-point detection"

One CUSUMMonitor per strategy. Raises alarm when cumulative downward deviation
from expected R exceeds threshold * std_r.
threshold=4.0 is ~4 standard deviations of accumulated drift before alarm.
"""
from dataclasses import dataclass, field


@dataclass
class CUSUMMonitor:
    expected_r: float    # Expected R per trade from backtest (e.g. 0.30)
    std_r: float         # Std deviation of R outcomes (use 1.0 as conservative default)
    threshold: float     # Alarm threshold in std units (e.g. 4.0)

    cusum_pos: float = field(default=0.0, init=False)
    cusum_neg: float = field(default=0.0, init=False)
    alarm_triggered: bool = field(default=False, init=False)
    n_trades: int = field(default=0, init=False)

    def update(self, actual_r: float) -> bool:
        """Process one trade result. Returns True if alarm just triggered."""
        self.n_trades += 1
        z = (actual_r - self.expected_r) / self.std_r
        self.cusum_neg = min(0.0, self.cusum_neg + z)   # lower: tracks losses
        self.cusum_pos = max(0.0, self.cusum_pos + z)   # upper: tracks outperformance

        if -self.cusum_neg > self.threshold and not self.alarm_triggered:
            self.alarm_triggered = True
            return True
        return False

    def clear(self) -> None:
        """Reset after investigation."""
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.alarm_triggered = False

    @property
    def drift_severity(self) -> float:
        """How many std devs below expected (positive = underperforming)."""
        return -self.cusum_neg
```

**Step 4: Run tests**
```bash
python -m pytest tests/test_trading_app/test_cusum_monitor.py -x -q
```

**Step 5: Commit**
```bash
git add trading_app/live/cusum_monitor.py tests/test_trading_app/test_cusum_monitor.py
git commit -m "feat: CUSUM drift detector for live strategy monitoring"
```

---

### Task D2: Performance monitor

```python
# trading_app/live/performance_monitor.py
"""
Per-strategy live P&L tracking with CUSUM drift detection.
Takes list[PortfolioStrategy] (from Portfolio.strategies) — NOT LiveStrategySpec.
PortfolioStrategy has strategy_id and expectancy_r. LiveStrategySpec does not.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from trading_app.portfolio import PortfolioStrategy
from .cusum_monitor import CUSUMMonitor

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    strategy_id: str
    trading_day: date
    direction: str
    entry_price: float
    exit_price: float
    actual_r: float
    expected_r: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMonitor:
    def __init__(self, strategies: list[PortfolioStrategy]):
        """
        strategies: Portfolio.strategies (list[PortfolioStrategy]).
        Each PortfolioStrategy has .strategy_id and .expectancy_r.
        """
        self._monitors: dict[str, CUSUMMonitor] = {
            s.strategy_id: CUSUMMonitor(
                expected_r=s.expectancy_r,
                std_r=1.0,
                threshold=4.0,
            )
            for s in strategies
        }
        self._strategies: dict[str, PortfolioStrategy] = {
            s.strategy_id: s for s in strategies
        }
        self._trades: list[TradeRecord] = []
        self._daily_r: dict[str, float] = {}

    def record_trade(self, record: TradeRecord) -> Optional[str]:
        """Record completed trade. Returns alert string if CUSUM alarm triggered."""
        self._trades.append(record)
        self._daily_r[record.strategy_id] = (
            self._daily_r.get(record.strategy_id, 0.0) + record.actual_r
        )
        monitor = self._monitors.get(record.strategy_id)
        if monitor and monitor.update(record.actual_r):
            msg = (f"⚠ CUSUM ALARM: {record.strategy_id} "
                   f"drift={monitor.drift_severity:.2f}σ after {monitor.n_trades} trades")
            log.warning(msg)
            return msg
        return None

    def daily_summary(self) -> dict:
        return {
            "date": date.today().isoformat(),
            "total_r": sum(self._daily_r.values()),
            "by_strategy": dict(self._daily_r),
            "n_trades": len(self._trades),
            "alarms": [sid for sid, m in self._monitors.items() if m.alarm_triggered],
        }

    def reset_daily(self) -> None:
        self._daily_r.clear()
```

```bash
git add trading_app/live/performance_monitor.py
git commit -m "feat: PerformanceMonitor with CUSUM, uses PortfolioStrategy (not LiveStrategySpec)"
```

---

## Phase E: Session Orchestrator

**Goal:** Tie everything together into a runnable live session.

**Critical API notes (verified):**
1. `build_live_portfolio()` returns `tuple[Portfolio, list[str]]` — MUST unpack
2. `engine.on_bar(bar_dict)` — one argument only. Bar dict needs `ts_utc` key.
3. `engine.on_trading_day_start(trading_day)` — must call before first bar of each day
4. `TradeEvent.event_type` — `"ENTRY"` or `"EXIT"` (not TradeState enum)
5. `TradeEvent.price` — entry or exit fill price (depends on event_type)
6. `TradeEvent` has NO `entry_model`, `pnl_r`, `expectancy_r` — look these up from Portfolio.strategies
7. `entry_model` for order routing: look up `PortfolioStrategy.entry_model` by `event.strategy_id`

---

### Task E1: Session orchestrator

```python
# trading_app/live/session_orchestrator.py
"""
Live trading session orchestrator.
DataFeed → BarAggregator → ExecutionEngine → OrderRouter → PerformanceMonitor.

VERIFIED API NOTES:
- build_live_portfolio() returns (Portfolio, notes) — unpack the tuple
- engine.on_bar(bar_dict) — bar_dict must have 'ts_utc' key, not 'ts_event'
- engine.on_trading_day_start(date) — call before first bar of day
- TradeEvent.event_type: "ENTRY" or "EXIT" (not TradeState enum)
- TradeEvent.price: entry fill on ENTRY, exit fill on EXIT
- entry_model: look up from self._strategy_map[event.strategy_id].entry_model
"""
import asyncio
import logging
from datetime import date
from pathlib import Path

from trading_app.live.tradovate_auth import TradovateAuth
from trading_app.live.data_feed import DataFeed
from trading_app.live.bar_aggregator import Bar
from trading_app.live.live_market_state import LiveORBBuilder
from trading_app.live.order_router import OrderRouter
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord
from trading_app.live.contract_resolver import resolve_front_month
from trading_app.execution_engine import ExecutionEngine
from trading_app.risk_manager import RiskManager, RiskLimits
from trading_app.live_config import build_live_portfolio
from trading_app.portfolio import PortfolioStrategy
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from pipeline.daily_backfill import run_backfill_for_instrument

log = logging.getLogger(__name__)


class SessionOrchestrator:
    def __init__(self, instrument: str, demo: bool = True, account_id: int = 0):
        self.instrument = instrument
        self.demo = demo
        self.trading_day = date.today()

        self.auth = TradovateAuth(demo=demo)

        # build_live_portfolio returns (Portfolio, list[str]) — unpack the tuple
        self.portfolio, notes = build_live_portfolio(
            db_path=GOLD_DB_PATH, instrument=instrument
        )
        for note in notes:
            log.info("live_config note: %s", note)

        if not self.portfolio.strategies:
            raise RuntimeError(f"No active strategies for {instrument}")

        # Strategy lookup map for resolving entry_model from strategy_id on TradeEvents
        self._strategy_map: dict[str, PortfolioStrategy] = {
            s.strategy_id: s for s in self.portfolio.strategies
        }

        # Execution stack
        cost = get_cost_spec(instrument)
        risk_limits = RiskLimits(
            max_daily_loss_r=self.portfolio.max_daily_loss_r,
            max_concurrent_positions=self.portfolio.max_concurrent_positions,
        )
        self.risk_mgr = RiskManager(risk_limits)
        self.engine = ExecutionEngine(
            portfolio=self.portfolio,
            cost_spec=cost,
            risk_manager=self.risk_mgr,
            live_session_costs=True,
        )

        # Live infrastructure
        self.orb_builder = LiveORBBuilder(instrument, self.trading_day)
        self.order_router = OrderRouter(account_id=account_id, auth=self.auth, demo=demo)
        # PerformanceMonitor takes list[PortfolioStrategy] (has strategy_id + expectancy_r)
        self.monitor = PerformanceMonitor(self.portfolio.strategies)

        # Entry price side-table: strategy_id → entry fill price (for pnl_r calculation on exit)
        self._entry_prices: dict[str, float] = {}

        # Resolve front-month contract symbol
        self.contract_symbol = resolve_front_month(instrument, self.auth, demo=demo)
        log.info("Session ready: %s → %s (%s)", instrument, self.contract_symbol,
                 "DEMO" if demo else "LIVE")

        # Signal engine that a new trading day is starting
        self.engine.on_trading_day_start(self.trading_day)

    def _on_bar(self, bar: Bar) -> None:
        """Called for each completed 1-minute bar from DataFeed."""
        self.orb_builder.on_bar(bar)

        # Bar dict must use key 'ts_utc' — that is what ExecutionEngine.on_bar() reads
        bar_dict = {
            "ts_utc": bar.ts_utc,   # NOT ts_event
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        # on_bar() takes ONE argument (bar dict), returns list[TradeEvent]
        events = self.engine.on_bar(bar_dict)

        for event in events:
            self._handle_event(event)

    def _handle_event(self, event) -> None:
        """
        Handle a TradeEvent from ExecutionEngine.
        TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
        event.price = entry fill on ENTRY events, exit fill on EXIT events
        There is NO entry_model, pnl_r, or expectancy_r on TradeEvent.
        """
        strategy = self._strategy_map.get(event.strategy_id)
        if strategy is None:
            log.warning("Unknown strategy_id: %s", event.strategy_id)
            return

        if event.event_type == "ENTRY":
            # Store entry price for pnl calculation on exit
            self._entry_prices[event.strategy_id] = event.price

            # Look up entry_model from PortfolioStrategy (not TradeEvent)
            spec = self.order_router.build_order_spec(
                direction=event.direction,
                entry_model=strategy.entry_model,  # from PortfolioStrategy, not TradeEvent
                entry_price=event.price,           # TradeEvent.price on ENTRY = fill price
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            result = self.order_router.submit(spec)
            log.info("ENTRY order: %s %s @ %.2f → orderId=%d",
                     event.strategy_id, event.direction, event.price, result.order_id)

        elif event.event_type == "EXIT":
            entry_price = self._entry_prices.pop(event.strategy_id, event.price)
            exit_price = event.price  # TradeEvent.price on EXIT = fill price

            # Compute actual R from prices (engine doesn't put pnl_r on TradeEvent)
            risk_pts = strategy.median_risk_points or 10.0
            direction_sign = 1.0 if event.direction == "long" else -1.0
            gross_pts = direction_sign * (exit_price - entry_price)
            actual_r = gross_pts / risk_pts if risk_pts > 0 else 0.0

            record = TradeRecord(
                strategy_id=event.strategy_id,
                trading_day=self.trading_day,
                direction=event.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                actual_r=actual_r,
                expected_r=strategy.expectancy_r,  # from PortfolioStrategy
            )
            alert = self.monitor.record_trade(record)
            if alert:
                log.warning(alert)
                # TODO Phase F: send to Slack / email

        elif event.event_type == "REJECT":
            log.info("REJECT: %s — %s", event.strategy_id, event.reason)

    async def run(self) -> None:
        feed = DataFeed(self.auth, on_bar=self._on_bar, demo=self.demo)
        log.info("Starting live feed: %s", self.contract_symbol)
        await feed.run(self.contract_symbol)

    def post_session(self) -> None:
        """EOD: log summary, run incremental backfill."""
        summary = self.monitor.daily_summary()
        log.info("EOD summary: %s", summary)
        try:
            run_backfill_for_instrument(self.instrument)
        except Exception as e:
            log.error("EOD backfill failed: %s", e)
```

```python
# scripts/run_live_session.py
"""Entry point for a live trading session."""
import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from trading_app.live.session_orchestrator import SessionOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--demo", action="store_true", default=True)
    parser.add_argument("--live", action="store_true",
                        help="REAL MONEY — requires typing CONFIRM")
    parser.add_argument("--account-id", type=int, default=0)
    args = parser.parse_args()

    if args.live:
        confirm = input("⚠ LIVE MODE with real money. Type CONFIRM to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            return
        demo = False
    else:
        demo = True

    session = SessionOrchestrator(
        instrument=args.instrument, demo=demo, account_id=args.account_id
    )
    try:
        asyncio.run(session.run())
    finally:
        session.post_session()


if __name__ == "__main__":
    main()
```

```bash
git add trading_app/live/session_orchestrator.py scripts/run_live_session.py
git commit -m "feat: live session orchestrator with verified API usage"
```

---

## Phase F: Integration Verification

### Task F1: Demo dry-run

```bash
# 1. Set .env with Tradovate demo credentials
# 2. Run against demo environment
python scripts/run_live_session.py --instrument MGC --demo --account-id <your_demo_account_id>

# Verify in logs:
# ✓ "Session ready: MGC → MGCM6 (DEMO)"
# ✓ Bars flowing in each minute
# ✓ "ENTRY order: ... → orderId=..."
# ✓ Orders visible in Tradovate demo web UI
```

### Task F2: Run full test suite

```bash
python -m pytest tests/ -x -q
python pipeline/check_drift.py
```
Expected: all existing 1,867 tests pass + new tests pass.

### Task F3: TopstepX wiring

TopstepX uses Tradovate. Use your TopstepX Tradovate account credentials in `.env`. No code changes.

---

## Discovery / Validation — Re-run Policy

| Step | Frequency | Trigger |
|------|-----------|---------|
| ingest → 5m → daily_features → outcomes | Daily automated | Task Scheduler 7am |
| strategy_discovery + strategy_validator | Monthly/quarterly | Manual, intentional |
| build_edge_families | After validator | Manual |
| live_config.py rebuild | After validator | Manual |

New bars do NOT invalidate existing validated strategies. One month of new outcomes barely moves multi-year validation windows. Re-run discovery quarterly or when you suspect regime change.

---

## Dependencies to Install

```bash
pip install websockets requests python-dotenv
```
Add to `requirements.txt`.

---

## Files Created Summary

```
pipeline/
  daily_backfill.py
scripts/
  setup_daily_backfill.py
  run_live_session.py
trading_app/live/
  __init__.py
  tradovate_auth.py
  bar_aggregator.py
  data_feed.py
  live_market_state.py
  contract_resolver.py
  order_router.py
  cusum_monitor.py
  performance_monitor.py
  session_orchestrator.py
tests/test_trading_app/
  test_bar_aggregator.py
  test_live_market_state.py
  test_order_router.py
  test_cusum_monitor.py
tests/test_pipeline/
  test_daily_backfill.py
```

---

## Pre-Live Checklist

Before switching `--demo` to `--live`:

- [ ] Demo dry-run completed — orders appear in Tradovate demo UI
- [ ] CUSUM alarm tested with synthetic losing sequence
- [ ] Daily loss limit set conservatively (start at 3R)
- [ ] Max concurrent positions = 1 per instrument
- [ ] TopstepX position limits checked (Combine daily loss limit)
- [ ] EOD backfill verified running on Task Scheduler
- [ ] 2+ weeks paper trading before live
- [ ] Manual kill: Ctrl+C → post_session() runs cleanly
