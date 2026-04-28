---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Multi-Broker Deployment Plan

**Created:** 2026-04-03
**Status:** READY FOR APPROVAL
**Goal:** Maximum profit extraction across all auto-friendly prop firms
**Base:** 9 regime-gated lanes, $10,352/yr per account at 1ct (filtered sim 2022-2025, max DD $2,423)

## What Already Exists

| Component | Status | File |
|-----------|--------|------|
| Session orchestrator | BUILT | `trading_app/live/session_orchestrator.py` |
| Copy order router (1→N accounts) | BUILT | `trading_app/live/copy_order_router.py` |
| Multi-instrument runner | BUILT | `trading_app/live/multi_runner.py` |
| Broker abstraction (5 ABCs) | BUILT | `trading_app/live/broker_base.py` |
| ProjectX integration (TopStep) | BUILT | `trading_app/live/projectx/` (1,227 lines) |
| Prop profiles | BUILT | `trading_app/prop_profiles.py` |
| Regime gate allocator | BUILT | `trading_app/lane_allocator.py` |
| Bot dashboard | BUILT | `trading_app/live/bot_dashboard.py` |
| Telegram alerts | RUNNING | `scripts/infra/telegram_feed.py` |

## What's Needed

### Broker Integrations Required

| Firm | Platform | API Type | Accounts | Complexity |
|------|----------|----------|----------|------------|
| **TopStep** | ProjectX | REST + WebSocket | 5 | **DONE** |
| **Tradeify** | Tradovate | REST + WebSocket | 5 | MEDIUM — same pattern as ProjectX |
| **MFFU** | Tradovate | REST + WebSocket | 5 | SAME as Tradeify (same API) |
| **Bulenox** | Rithmic | Protobuf + WebSocket | 3 | HARD — different protocol entirely |

### Tradovate API (verified from official docs)

```
Auth:    POST https://live.tradovateapi.com/v1/auth/accesstokenrequest
Demo:    POST https://demo.tradovateapi.com/v1/auth/accesstokenrequest
Orders:  POST /v1/order/placeorder {accountId, action, symbol, orderQty, orderType, price, isAutomated: true}
Bracket: POST /v1/order/placeOSO {bracket1: {action, orderType, price}}
Cancel:  POST /v1/order/cancelorder {orderId}
WS:      wss://live.tradovateapi.com/v1/websocket (authorize with accessToken)
         wss://md.tradovateapi.com/v1/websocket (market data)
```

Key differences from ProjectX:
- Auth: username+password+cid+sec (vs ProjectX apiKey)
- Orders: `placeorder` endpoint (vs ProjectX `Order/Limit`, `Order/StopMarket`)
- Brackets: `placeOSO` (vs ProjectX native bracket)
- `isAutomated: true` REQUIRED for bot orders
- Returns `mdAccessToken` separately for market data

### Rithmic API (for Bulenox)

- Protocol Buffers over WebSocket (NOT REST)
- Python library exists: `pyrithmic` (GitHub: jacksonwoody/pyrithmic)
- Significantly different architecture from REST APIs
- **Recommendation: DEFER to Phase 3. Use 3rd party copier (Replikanto/FutuCopy) for Bulenox.**

## Architecture

```
                    ┌─────────────────┐
                    │  Lane Allocator  │
                    │  (regime gate)   │
                    └────────┬────────┘
                             │ 9 lanes
                    ┌────────┴────────┐
                    │ SessionOrchestrator │
                    │ (per instrument)    │
                    └────────┬────────┘
                             │ trade signals
                    ┌────────┴────────┐
                    │ BrokerDispatcher │  ← NEW: routes to multiple brokers
                    └────────┬────────┘
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────┴──────┐ ┌────┴─────┐ ┌──────┴────────┐
    │ CopyOrderRouter│ │CopyOrder │ │ 3rd party     │
    │ (ProjectX x5)  │ │(Tradovate│ │ copier        │
    │ TopStep        │ │ x10)     │ │ (Bulenox x3)  │
    └────────────────┘ │Tradeify  │ └───────────────┘
                       │+ MFFU    │
                       └──────────┘
```

### New Component: BrokerDispatcher

Sits between SessionOrchestrator and CopyOrderRouters. When a trade signal fires:
1. Build order spec from the signal
2. Fan out to ALL active CopyOrderRouters (one per firm)
3. Each CopyOrderRouter fans out to N accounts within that firm
4. Return primary result, log shadow results

This is a thin layer — ~50 lines wrapping multiple CopyOrderRouters.

## Implementation Stages

### Stage 1: Tradovate Integration (Tradeify + MFFU)

**Files to create:**
```
trading_app/live/tradovate/__init__.py
trading_app/live/tradovate/auth.py          # TradovateAuth(BrokerAuth)
trading_app/live/tradovate/order_router.py  # TradovateOrderRouter(BrokerRouter)
trading_app/live/tradovate/data_feed.py     # TradovateFeed(BrokerFeed)
trading_app/live/tradovate/contracts.py     # TradovateContracts(BrokerContracts)
trading_app/live/tradovate/positions.py     # TradovatePositions(BrokerPositions)
```

**Pattern:** Mirror `projectx/` structure exactly. Same ABCs, same method signatures.

**Auth mapping:**
```python
# ProjectX
POST /api/Auth/loginKey {userName, apiKey} → {token}

# Tradovate
POST /v1/auth/accesstokenrequest {name, password, cid, sec, deviceId} → {accessToken, mdAccessToken}
```

**Order mapping:**
```python
# ProjectX
POST /api/Order/StopMarket {accountId, contractId, action, qty, stopPrice}

# Tradovate
POST /v1/order/placeorder {accountId, symbol, action, orderQty, orderType: "Stop", stopPrice, isAutomated: true}
```

**Bracket mapping:**
```python
# ProjectX: separate bracket endpoint
POST /api/Order/Bracket {entryOrderId, stopLoss, takeProfit}

# Tradovate: OSO (Order Sends Order)
POST /v1/order/placeOSO {initial order + bracket1: {stop} + bracket2: {target}}
```

**Effort:** ~500 lines across 5 files. 2-3 hours. Same patterns as ProjectX.

### Stage 2: BrokerDispatcher + Multi-Firm Profiles

**Files to create/modify:**
```
trading_app/live/broker_dispatcher.py       # NEW: multi-broker signal routing
trading_app/prop_profiles.py                # ADD: TopStep x5, Tradeify x5, MFFU x5 profiles
trading_app/live/broker_factory.py          # MODIFY: add Tradovate broker creation
```

**Effort:** ~200 lines new, ~50 lines modified. 1 hour.

### Stage 3: Bulenox via 3rd Party Copier

**Not code — configuration:**
- Install Replikanto or FutuCopy on the Bulenox NinjaTrader instances
- Configure to copy from TopStep master account
- No new code needed — external tool handles the routing

**Effort:** 30 minutes setup per account.

### Stage 4: IBKR Integration (future)

**Files to create:**
```
trading_app/live/ibkr/__init__.py
trading_app/live/ibkr/auth.py
trading_app/live/ibkr/order_router.py
trading_app/live/ibkr/data_feed.py
trading_app/live/ibkr/contracts.py
trading_app/live/ibkr/positions.py
```

**Uses:** `ib_insync` Python library (well-documented, mature).
**Effort:** ~400 lines. 2 hours.

## Profit Targets

| Phase | Firm | Accounts | NET/yr | Timeline |
|-------|------|----------|--------|----------|
| 1a | TopStep (ProjectX — DONE) | 5 | $40,644 | NOW (eval pass needed) |
| 1b | Tradovate integration | 0 | $0 | 1 week build |
| 2a | Tradeify (Tradovate) | 5 | $43,044 | +2 weeks (build + eval) |
| 2b | MFFU (Tradovate) | 5 | $40,584 | +3 weeks (eval) |
| 3 | Bulenox (3rd party copier) | 3 | $23,738 | +1 month |
| 4 | IBKR (self-funded) | 1 @2ct | $20,704 | +6 months |
| **TOTAL** | | **19+1** | **$168,714/yr** | |

## Risk Management

- **DD budget:** $2,423 max DD (filtered sim) vs $3K limit (TopStep/Bulenox) = $577 headroom
- **Tradeify/MFFU DD:** $2,500 — TIGHT ($77 headroom). Monitor closely first month.
- **Account blow rate:** ~4%/yr at $3K DD. Budget 1 reset/yr across all accounts = ~$100-200/yr.
- **Single point of failure:** Data feed. If Databento goes down, all accounts stop. Mitigation: daily refresh at 07:30, bot only needs gold.db (local).
- **Regime gate lag:** 6-month trailing. Won't catch sudden regime breaks within 1 month. Accepted risk — backtest shows this still works (6/6 years positive).

## Acceptance Criteria

- [ ] Tradovate auth works (demo + live)
- [ ] Tradovate placeorder works (market, stop, bracket)
- [ ] CopyOrderRouter works with Tradovate (N accounts)
- [ ] BrokerDispatcher routes to ProjectX + Tradovate simultaneously
- [ ] All 9 lanes fire correctly on demo across both brokers
- [ ] Prop profiles for TopStep x5, Tradeify x5, MFFU x5
- [ ] Drift check passes
- [ ] Existing ProjectX integration unaffected (no regression)
