# Go-Live Deployment Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Get from "code complete" to "placing real trades" in the shortest safe path.

**Architecture:** The entire live trading stack is already built (12 modules, ~3,000 lines, reviewed, bug-fixed today). Three execution modes exist: `--signal-only` (console alerts, no orders), `--demo` (paper trading on Tradovate), `--live` (real money). The ONLY blocker is Tradovate credentials. Zero new code required for core functionality.

**Tech Stack:** Python, Tradovate REST + WebSocket, DuckDB (gold.db), existing ExecutionEngine/SessionOrchestrator/DataFeed/OrderRouter, `websockets` library.

---

## Background — What Already Exists

### Signal Alerter (ALREADY BUILT)
`scripts/run_live_session.py --signal-only --instrument MGC` does everything:
- Connects to Tradovate market data WebSocket (read-only)
- Builds 1m bars via `bar_aggregator.py`
- Feeds bars to `ExecutionEngine` which detects ORB + breaks
- Logs `⚡ SIGNAL [strategy_id]: BUY MGCM6 @ 2950.00 ← trade this manually`
- Writes all signals to `live_signals.jsonl`
- Defaults to signal-only if no mode flag given (safest default)
- **Still needs Tradovate creds** — even market data requires auth token

### Demo/Live Modes (ALREADY BUILT)
- `--demo` connects to `demo.tradovateapi.com`, places paper orders via `OrderRouter`
- `--live` requires typing "CONFIRM", connects to `live.tradovateapi.com`
- Exit/scratch events auto-submit closing orders
- `post_session()` runs on Ctrl+C for cleanup

### Infrastructure (ALREADY BUILT)
- `TradovateAuth` — OAuth token mgmt, auto-renewal
- `DataFeed` — WS quote subscription + heartbeat (2.5s)
- `BarAggregator` — tick → 1m OHLCV bars
- `ContractResolver` — front-month lookup via REST
- `OrderRouter` — E1→Market, E2→Stop, exits→Market
- `CUSUMMonitor` — drift detection per strategy
- `PerformanceMonitor` — live P&L tracking
- `SessionOrchestrator` — wires everything together
- `daily_backfill.py` — EOD automated pipeline refresh
- `setup_daily_backfill.py` — Windows Task Scheduler at 7am Brisbane

### Critical Bug Fixed Today (Mar 6)
`daily_features_row=None` in live mode caused ALL fail-closed filters to silently reject every trade. Fixed: `_build_daily_features_row()` now queries gold.db for yesterday's features as proxy.

---

## Tradovate API Reference

### Base URLs
| Environment | REST | WebSocket | Market Data |
|------------|------|-----------|-------------|
| **Demo** | `https://demo.tradovateapi.com/v1` | `wss://demo.tradovateapi.com/v1/websocket` | `wss://md.tradovateapi.com/v1/websocket` |
| **Live** | `https://live.tradovateapi.com/v1` | `wss://live.tradovateapi.com/v1/websocket` | `wss://md.tradovateapi.com/v1/websocket` |

### Authentication
POST `/auth/accesstokenrequest`:
```json
{
  "name": "<TRADOVATE_USER>",
  "password": "<TRADOVATE_PASS>",
  "appId": "<TRADOVATE_APP_ID>",
  "appVersion": "<TRADOVATE_APP_VERSION>",
  "cid": "<TRADOVATE_CID>",
  "sec": "<TRADOVATE_SEC>",
  "deviceId": "<UUID>"
}
```
Returns: `accessToken` (REST), `mdAccessToken` (market data WS), `expirationTime`.

### Order Placement
POST `/order/placeorder`:
```json
{
  "accountSpec": "<username>",
  "accountId": 12345,
  "action": "Buy",
  "symbol": "MGCM6",
  "orderQty": 1,
  "orderType": "Stop",
  "stopPrice": 2950.0,
  "isAutomated": true,
  "timeInForce": "Day"
}
```
**CRITICAL:** `isAutomated: true` required for algorithmic orders (exchange policy).

### Rate Limits
- Generous for normal use (per-second/minute/hour)
- Auth requests have stricter limits + `p-ticket`/`p-time` penalty
- 1 simultaneous connection per customer (default)

---

## Implementation Tasks

### Task 0: Get Tradovate Demo Credentials

**Files:**
- Modify: `.env` (fill 6 empty TRADOVATE_* vars)

**Step 1: Sign up for Tradovate demo account**

Go to https://trader.tradovate.com/welcome and create a demo account.
After signup, go to Application Settings to find your API credentials.

**Step 2: Get CID and SEC**

In the Tradovate Trader app: Settings → API Access → create new API key.
This gives you:
- `CID` (numeric client ID)
- `SEC` (UUID secret)

**Step 3: Fill `.env`**

```bash
# In .env file:
TRADOVATE_USER=your_username
TRADOVATE_PASS=your_password
TRADOVATE_APP_ID=ORB_Trading
TRADOVATE_APP_VERSION=1.0
TRADOVATE_CID=your_cid_number
TRADOVATE_SEC=your_sec_uuid
```

**Step 4: Verify auth works**

Run:
```bash
python -c "
from trading_app.live.tradovate_auth import TradovateAuth
auth = TradovateAuth(demo=True)
token = auth.get_token()
print(f'Token acquired: {token[:20]}...')
print(f'Headers: {auth.headers()}')
"
```
Expected: Token string printed, no errors.

**Step 5: Commit (DO NOT commit .env — it's in .gitignore)**

No commit needed. Credentials stay local.

---

### Task 1: First Signal-Only Session

**Files:** None to create or modify — all code exists.

**Step 1: Run signal-only mode**

```bash
python scripts/run_live_session.py --instrument MGC --signal-only
```

Expected output:
```
╔══════════════════════════════════════════╗
║   MODE: SIGNAL ONLY — no orders placed   ║
║   Watch for ⚡ SIGNAL lines in the log   ║
║   Trade manually on Tradovate/TV          ║
╚══════════════════════════════════════════╝
   Instrument: MGC
Session ready: MGC → MGCM6 (SIGNAL-ONLY)
Starting live feed: MGCM6
```

**Step 2: Verify market data flows**

Watch the console. You should see:
- 1m bar formation logs (every minute during market hours)
- ORB detection when a session window opens
- If a break occurs: `⚡ SIGNAL [strategy_id]: BUY/SELL MGCM6 @ price`

**Step 3: Check signals file**

```bash
cat live_signals.jsonl
```

Should contain SESSION_START record and any SIGNAL_ENTRY/SIGNAL_EXIT events.

**Step 4: Test clean shutdown**

Press Ctrl+C. Verify `post_session()` runs and prints session summary.

---

### Task 2: First Demo Order

**Files:** None to create or modify.

**Prerequisite:** Task 0 complete (credentials working).

**Step 1: Find your demo account ID**

```bash
python -c "
from trading_app.live.tradovate_auth import TradovateAuth
from trading_app.live.contract_resolver import resolve_account_id
auth = TradovateAuth(demo=True)
acct_id = resolve_account_id(auth, demo=True)
print(f'Demo account ID: {acct_id}')
"
```

**Step 2: Run demo mode**

```bash
python scripts/run_live_session.py --instrument MGC --demo
```

Expected: Same as signal-only but ENTRY events now submit orders to Tradovate demo.

**Step 3: Verify in Tradovate Trader**

Open the Tradovate Trader app (demo environment). When a signal fires:
- Order should appear in the Orders panel
- For E2 entries: Stop order at the break level
- For exits: Market order closing the position

**Step 4: Verify exit fires**

Wait for a stop hit or target hit. Confirm:
- EXIT event logged in console
- Closing order submitted to Tradovate
- Position closed in Tradovate Trader

**Step 5: Verify post_session cleanup**

Press Ctrl+C. Confirm no orphaned positions in Tradovate.

---

### Task 3: EOD Backfill Automation

**Files:** None to create — script exists.

**Step 1: Test daily backfill manually**

```bash
python pipeline/daily_backfill.py
```

Expected: Ingests today's bars, rebuilds 5m bars, rebuilds daily_features.

**Step 2: Set up Windows Task Scheduler**

```bash
python scripts/setup_daily_backfill.py
```

This creates a scheduled task to run at 7:00 AM Brisbane (before market open).

**Step 3: Verify next morning**

Check Task Scheduler → confirm task ran → check gold.db has fresh data.

---

### Task 4: Paper Trading Campaign (2+ Weeks)

**Files:** None.

**Step 1: Daily routine**

1. 30 min before target session: `python scripts/run_live_session.py --instrument MGC --demo`
2. Monitor console for signals
3. After session: check `live_signals.jsonl` for all events
4. Compare demo fills vs backtest expectations

**Step 2: Track metrics**

After each session, note:
- Number of signals fired vs expected
- Fill prices vs backtest entry prices
- Any filter rejections (logged as REJECT events)
- CUSUM alerts (if any)

**Step 3: Multi-instrument expansion**

After 1 week on MGC, add MNQ:
```bash
# Terminal 1:
python scripts/run_live_session.py --instrument MGC --demo
# Terminal 2:
python scripts/run_live_session.py --instrument MNQ --demo
```

Note: Each instance uses a separate WebSocket connection. Stay within connection limits.

**Step 4: Go/No-Go decision**

After 2+ weeks, review:
- [ ] Signals match backtest expectations (>80% alignment)
- [ ] No orphaned positions after shutdown
- [ ] CUSUM not alarming on any strategy
- [ ] Daily backfill running reliably
- [ ] Confident in the system's behavior

---

### Task 5: Go Live

**Files:**
- Modify: `.env` (switch to live credentials)

**Step 1: Switch credentials**

Update `.env` with live account credentials (different CID/SEC from demo).

**Step 2: Conservative risk limits**

Verify in `trading_app/live_config.py`:
- `max_daily_loss_r = 3.0` (3R daily loss limit)
- `max_concurrent_positions = 1` (one position at a time)

**Step 3: Single instrument start**

```bash
python scripts/run_live_session.py --instrument MGC --live
```

Type `CONFIRM` when prompted.

**Step 4: Monitor first live session**

Stay at the screen for the entire session. Watch for:
- Correct entry signals
- Orders appearing in Tradovate (live account)
- Stop/target levels matching expectations
- Clean exit on target or stop

**Step 5: Scale up**

After 1 month of consistent results:
- Add second instrument
- Consider increasing position size
- Review CUSUM for strategy drift

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| No broker-side stop-loss (V1) | Engine manages stops internally. Conservative sizing. |
| WS disconnect = unprotected position | Session terminates. Manual monitoring during V1. |
| Sync HTTP blocks event loop | ~100ms, logged if >1s. Acceptable for V1. |
| Demo fills != live fills | Expected. Demo validates logic, not execution quality. |
| Single connection limit | One instance per environment. |

## Validation Checklist

- [ ] Tradovate demo credentials in .env (all 6 vars populated)
- [ ] Auth token acquired successfully
- [ ] Signal-only session connects and receives bars
- [ ] ORB detected at session open matches manual observation
- [ ] At least 1 demo order appears in Tradovate Trader UI
- [ ] Exit order fires on stop or target hit
- [ ] Ctrl+C triggers clean post_session()
- [ ] Daily backfill on Task Scheduler verified
- [ ] 2+ weeks demo trading before live
- [ ] CUSUM not alarming before go-live
