# Go-Live Deployment Plan

**Date:** 2026-03-06
**Goal:** Get from "code complete" to "placing real trades" in the shortest safe path.
**Status:** Infrastructure merged to main. Filter bug fixed today. Zero credentials configured.

---

## Orient: What Exists vs What's Missing

### Already Built (DO NOT REBUILD)
- `trading_app/live/` — 12 modules, ~3,000 lines, fully reviewed and bug-fixed
- `trading_app/execution_engine.py` — bar-by-bar state machine
- `trading_app/live_config.py` — 26 live portfolio specs across 10 sessions
- `scripts/run_live_session.py` — CLI entry point (`--demo` / `--live`)
- `pipeline/daily_backfill.py` — EOD automated pipeline refresh
- `scripts/setup_daily_backfill.py` — Windows Task Scheduler 7am
- 125+ active LONDON_METALS strategies, 448 tests passing, 61 drift checks

### What's Missing (Deployment, Not Code)
1. **Tradovate credentials** — `.env` has empty `TRADOVATE_USER`, `TRADOVATE_PASS`, `TRADOVATE_CID`, `TRADOVATE_SEC`
2. **Demo account dry-run** — never executed end-to-end
3. **Signal alerter** — no way to get "TRADE NOW" notifications without running the full live system
4. **Paper trading validation** — 2+ weeks recommended before real money

---

## Design: Two-Track Deployment

### Track A: Signal Alerter (Immediate — enables manual trading)

Lightweight script that:
1. Queries gold.db for today's daily_features (filter eligibility)
2. Connects to Tradovate market data WebSocket (read-only, no order placement)
3. Builds 1m bars in real-time via `bar_aggregator.py`
4. Feeds bars to `ExecutionEngine` which detects ORB formation + breaks
5. On TradeEvent(ENTRY), emits console alert + optional Windows toast notification
6. Shows: instrument, session, direction, entry price, stop, target (RR), filter status

**No orders placed.** Human reads alert, places order manually in Tradovate Trader.

Key: Reuses existing `DataFeed`, `BarAggregator`, `ExecutionEngine`, `LiveConfig`.
New code: ~50-line wrapper script + notification function.

Still needs Tradovate credentials for market data WebSocket, but read-only — zero risk.

### Track B: Full Demo Automation (This Weekend)

1. Get Tradovate demo credentials (sign up at trader.tradovate.com)
2. Fill `.env` with 6 credentials
3. Run `python scripts/run_live_session.py --instrument MGC --demo --account-id <ID>`
4. Verify orders appear in Tradovate demo UI
5. Run for 2+ weeks across multiple sessions
6. Compare demo fills vs backtest expectations

### Track C: Go Live (After 2+ Weeks Demo)

1. Switch to live credentials
2. Set conservative risk limits: daily_loss=3R, max_concurrent=1
3. Start with single instrument (MGC or MNQ)
4. Monitor via CUSUM + performance dashboard

---

## Tradovate API Reference (Extracted from Docs)

### Base URLs
| Environment | REST | WebSocket | Market Data |
|------------|------|-----------|-------------|
| **Demo** | `https://demo.tradovateapi.com/v1` | `wss://demo.tradovateapi.com/v1/websocket` | `wss://md.tradovateapi.com/v1/websocket` |
| **Live** | `https://live.tradovateapi.com/v1` | `wss://live.tradovateapi.com/v1/websocket` | `wss://md.tradovateapi.com/v1/websocket` |
| **Replay** | — | `wss://replay.tradovateapi.com/v1/websocket` | — |

### Authentication
POST `/auth/accesstokenrequest` with JSON body:
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
Returns: `accessToken`, `mdAccessToken`, `expirationTime`.
Use `Bearer <accessToken>` in Authorization header for REST.
Use `mdAccessToken` for market data WebSocket auth.
Renew via GET `/auth/renewaccesstoken`.

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
Order types: `Market`, `Limit`, `Stop`, `StopLimit`, `MIT`, `TrailingStop`, `TrailingStopLimit`
**CRITICAL:** `isAutomated: true` is REQUIRED for algorithmic orders (exchange policy).

### Market Data (WebSocket)
- `md/subscribeQuote` — `{"symbol": "MGCM6"}` — real-time quotes
- `md/getChart` — 1m bars via `MinuteBar` with `elementSize: 1`
- `md/cancelChart` — unsubscribe with `subscriptionId`
- Heartbeat required every 2.5 seconds or connection drops

### Rate Limits
- Per-second, per-minute, per-hour limits (generous for normal use)
- Auth requests have stricter limits + time penalty (`p-ticket` / `p-time`)
- 429 = Too Many Requests, wait and retry
- 1 simultaneous connection per customer (default; more available by subscription)

### Contract Resolution
- GET `/product/find?name=MGC` — find product by name
- GET `/contract/find?name=MGCM6` — find specific contract
- GET `/contract/suggest?t=MGC&l=10` — suggest contracts

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| No broker-side stop-loss (V1) | Engine manages stops internally. Conservative position sizing. |
| WS disconnect = unprotected position | Session terminates on disconnect. Manual monitoring during V1. |
| Sync HTTP blocks event loop | ~100ms latency, logged if >1s. Acceptable for V1. |
| Demo fills != live fills | Expected. Demo validates logic, not execution quality. |
| Single connection limit | One instance per environment. Don't run demo + live simultaneously. |

---

## Validation Criteria

- [ ] Tradovate demo credentials in .env (all 6 vars populated)
- [ ] `python scripts/run_live_session.py --instrument MGC --demo` starts without error
- [ ] Market data WebSocket connects and receives quotes
- [ ] ORB detected at session open matches manual observation
- [ ] At least 1 demo order appears in Tradovate Trader UI
- [ ] CUSUM alarm fires on synthetic bad sequence (existing test)
- [ ] Daily loss limit triggers at 3R (test with demo)
- [ ] EOD backfill runs on Windows Task Scheduler at 7am
- [ ] 2+ weeks demo trading before live
- [ ] Manual Ctrl+C triggers clean post_session()

---

## Implementation Tasks

### Phase 1: Credentials & First Connection (30 min)
1. Sign up for Tradovate demo account at trader.tradovate.com
2. Get API credentials (CID + SEC from Application Settings)
3. Fill `.env` with all 6 TRADOVATE_* vars
4. Test auth: `python -c "from trading_app.live.tradovate_auth import TradovateAuth; ..."`

### Phase 2: Signal Alerter Script (1-2 hours)
1. Create `scripts/run_signal_alerter.py`
   - Reuses DataFeed, BarAggregator, ExecutionEngine, LiveConfig
   - Console output: "SIGNAL: MGC LONDON_METALS LONG @ 2950.0, Stop 2940.0, Target 2975.0 (RR 2.5)"
   - Optional: Windows toast notification via `win10toast` or `plyer`
   - No order submission — read-only market data only
2. Test with a live session (any active session)

### Phase 3: Full Demo Dry-Run (1-2 hours)
1. Run `python scripts/run_live_session.py --instrument MGC --demo --account-id <ID>`
2. Wait for session to fire
3. Verify order appears in Tradovate demo UI
4. Verify stop/target are correct
5. Verify exit order fires on stop or target hit
6. Verify post_session() cleanup

### Phase 4: Paper Trading Campaign (2+ weeks)
1. Run signal alerter during manual trading sessions
2. Run demo automation on parallel sessions
3. Log discrepancies between backtest expectations and demo fills
4. Monitor CUSUM for strategy drift

### Phase 5: Go Live
1. Switch `.env` to live credentials
2. Set risk limits: daily_loss=3R, max_concurrent=1, single instrument
3. Start with smallest position size
4. Scale up after 1 month of consistent results
