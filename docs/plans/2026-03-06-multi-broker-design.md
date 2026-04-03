# Multi-Broker Live Trading — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Support both TopstepX (ProjectX API) and Tradovate as interchangeable brokers, selectable via `.env` config.

**Architecture:** Broker abstraction layer with 5 ABCs (Auth, DataFeed, OrderRouter, ContractResolver, Positions) + factory pattern. Existing Tradovate code refactored into a submodule; new ProjectX module added alongside. SessionOrchestrator becomes broker-agnostic.

**Tech Stack:** Python, `requests` (REST), `websockets` (Tradovate WS), `signalrcore` (ProjectX SignalR), DuckDB (gold.db), existing ExecutionEngine/SessionOrchestrator.

---

## Background

### Why Multi-Broker?

Tradovate API requires a personal live funded account ($1,000+ equity) + $25/mo API Access subscription. Apex prop firm accounts cannot get direct API access (confirmed via Tradovate community forums, Oct 2024 - Nov 2025). TopstepX (ProjectX) explicitly supports API access for prop firm traders at $29/mo ($14.50 with code "topstep").

### What Already Exists

12 modules in `trading_app/live/` (~3,000 lines), all hardcoded to Tradovate:
- `tradovate_auth.py` — OAuth token management
- `data_feed.py` — Tradovate WebSocket market data
- `order_router.py` — Tradovate REST order placement
- `contract_resolver.py` — Tradovate contract lookup + account discovery
- `session_orchestrator.py` — Wires everything together (broker-agnostic logic already here)
- `bar_aggregator.py` — Tick → 1m OHLCV (broker-independent)
- `execution_engine.py` — Strategy evaluation (broker-independent)
- `cusum_monitor.py`, `performance_monitor.py`, `live_market_state.py` — All broker-independent

### ProjectX Gateway API (from official docs)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/Auth/loginKey` | POST | Auth with `{userName, apiKey}` → JWT token |
| `/api/Auth/validate` | POST | Refresh expired token (24h lifetime) |
| `/api/Account/search` | POST | List accounts `{onlyActiveAccounts: true}` |
| `/api/Contract/available` | POST | List tradeable contracts `{live: false}` |
| `/api/Order/place` | POST | Place order with brackets |
| `/api/Order/cancel` | POST | Cancel order |
| `/api/Position/searchOpen` | POST | Open positions |
| `/api/Position/close` | POST | Close position |

**Base URL:** `https://api.thefuturesdesk.projectx.com`
**Market Hub:** `https://rtc.thefuturesdesk.projectx.com/hubs/market` (SignalR/WebSocket)
**User Hub:** `https://rtc.thefuturesdesk.projectx.com/hubs/user` (SignalR/WebSocket)
**Rate limits:** 200 req/60s general, 50 req/30s for history bars

**Order type enums:** `1`=Limit, `2`=Market, `4`=Stop, `5`=TrailingStop
**Side enums:** `0`=Bid (buy), `1`=Ask (sell)
**Contract IDs:** `CON.F.US.MGC.M26` format (vs Tradovate `MGCM6`)

**Real-time data:** SignalR library with `SubscribeContractQuotes` and `SubscribeContractTrades` methods. Events: `GatewayQuote` (lastPrice, bestBid, bestAsk, OHLV), `GatewayTrade` (price, volume, type).

**Native brackets:** `stopLossBracket: {ticks, type}` and `takeProfitBracket: {ticks, type}` on order placement. This is a significant advantage over Tradovate where the engine must manage stops separately.

---

## Broker Comparison

| Aspect | Tradovate | ProjectX (TopstepX) |
|--------|-----------|---------------------|
| Auth | OAuth: user+pass+CID+SEC → accessToken | API key: user+apiKey → JWT (24h) |
| Token refresh | Auto before expiry (60s buffer) | POST /api/Auth/validate |
| Market data | Raw WebSocket + JSON + 2.5s heartbeat | SignalR WebSocket (auto-reconnect) |
| Contract format | `MGCM6` (name string) | `CON.F.US.MGC.M26` (structured ID) |
| Order types | `"Market"/"Stop"` strings | `2/4` integer enums |
| Order sides | `"Buy"/"Sell"` strings | `0/1` integer enums |
| Brackets | Not native — engine manages stops | Native stopLoss/takeProfit brackets |
| Account lookup | GET /account/list | POST /api/Account/search |
| Heartbeat | Manual 2.5s ping required | SignalR manages automatically |
| Prop firm support | NO (requires personal funded account) | YES (designed for prop firms) |
| Cost | $25/mo + $1,000 deposit | $14.50/mo (with code "topstep") |

---

## Architecture

### File Structure

```
trading_app/live/
  broker_base.py           # ABCs: BrokerAuth, BrokerFeed, BrokerRouter, BrokerContracts
  broker_factory.py        # create_broker() factory
  tradovate/               # Existing code, refactored
    __init__.py
    auth.py                # TradovateAuth (from tradovate_auth.py)
    data_feed.py           # TradovateDataFeed (from data_feed.py)
    order_router.py        # TradovateOrderRouter (from order_router.py)
    contract_resolver.py   # TradovateContracts (from contract_resolver.py)
  projectx/                # NEW
    __init__.py
    auth.py                # ProjectXAuth
    data_feed.py           # ProjectXDataFeed (SignalR)
    order_router.py        # ProjectXOrderRouter
    contract_resolver.py   # ProjectXContracts
  bar_aggregator.py        # UNCHANGED (broker-independent)
  session_orchestrator.py  # MODIFIED (use ABCs instead of concrete classes)
  # Keep old files as thin re-exports for backwards compat during transition
```

### Abstract Base Classes

```python
# broker_base.py
from abc import ABC, abstractmethod

class BrokerAuth(ABC):
    @abstractmethod
    def get_token(self) -> str: ...
    @abstractmethod
    def headers(self) -> dict: ...
    @abstractmethod
    def refresh_if_needed(self) -> None: ...  # Proactive token refresh (ProjectX: /validate, Tradovate: pre-expiry)

class BrokerFeed(ABC):
    @abstractmethod
    async def run(self, symbol: str) -> None: ...
    @abstractmethod
    def flush(self, symbol: str = "") -> Bar | None: ...

class BrokerRouter(ABC):
    @abstractmethod
    def build_order_spec(self, direction, entry_model, entry_price, symbol, qty) -> OrderSpec: ...
    @abstractmethod
    def submit(self, spec: OrderSpec) -> OrderResult: ...
    @abstractmethod
    def build_exit_spec(self, direction, symbol, qty) -> OrderSpec: ...
    @abstractmethod
    def cancel(self, order_id: int) -> None: ...
    @abstractmethod
    def supports_native_brackets(self) -> bool: ...  # ProjectX: True, Tradovate: False

class BrokerContracts(ABC):
    @abstractmethod
    def resolve_account_id(self) -> int: ...
    @abstractmethod
    def resolve_front_month(self, instrument: str) -> str: ...

class BrokerPositions(ABC):
    """Query broker for open positions — critical for crash recovery + EOD reconciliation."""
    @abstractmethod
    def query_open(self, account_id: int) -> list[dict]: ...
    # Returns [{contract_id, side, size, avg_price}]
```

### Position Reconciliation (M2.5 Review — P0)

On session start, call `BrokerPositions.query_open()` to detect orphaned positions from previous crashes. On `post_session()`, compare engine state vs broker state. Log discrepancies as CRITICAL. This prevents capital loss from untracked open positions.

### Native Bracket Orders (M2.5 Review — P1)

ProjectX supports `stopLossBracket` and `takeProfitBracket` on the `/api/Order/place` request — one API call for entry + stop + target. `ProjectXOrderRouter.supports_native_brackets()` returns True. When True, SessionOrchestrator can submit entry with brackets in a single call instead of managing stops separately via the engine. Tradovate returns False — engine continues managing stops.

### Factory

```python
# broker_factory.py
def create_broker(name: str, demo: bool = True) -> tuple[BrokerAuth, BrokerFeed, BrokerRouter, BrokerContracts]:
    if name == "projectx":
        from .projectx.auth import ProjectXAuth
        # ... etc
    elif name == "tradovate":
        from .tradovate.auth import TradovateAuth
        # ... etc
    else:
        raise ValueError(f"Unknown broker: {name}")
```

### Contract ID Mapping

ProjectX uses structured IDs like `CON.F.US.MGC.M26`. We need a mapping:

```python
# ProjectX symbol mapping — MUST BE VERIFIED against /api/Contract/available
# These are educated guesses from the API docs examples. Do NOT hardcode until confirmed.
INSTRUMENT_TO_SYMBOL = {
    "MGC": "F.US.MGC",    # Micro Gold — VERIFY
    "MNQ": "F.US.ENQ",    # Micro Nasdaq — VERIFY (could be F.US.MNQ)
    "MES": "F.US.EP",     # Micro S&P — VERIFY (could be F.US.MES)
    "M2K": "F.US.RTY",    # Micro Russell — VERIFY (could be F.US.M2K)
}
```

**CRITICAL (M2.5 Review — P1):** The exact symbol IDs MUST be confirmed by calling `/api/Contract/available` during first auth. The mapping above is from API doc examples which show full-size contracts (ES→EP, NQ→ENQ, RTY). Micro contracts may have different symbol IDs. Implementation Task 1 must include a discovery step that fetches and logs all available contracts.

### SignalR Data Feed

ProjectX uses Microsoft SignalR for real-time data. Python library: `signalrcore` (or `pysignalr`).

```python
# Simplified ProjectX feed flow
hub = HubConnectionBuilder()
    .with_url(f"{MARKET_HUB}?access_token={token}")
    .build()

hub.on("GatewayQuote", handle_quote)
hub.on("GatewayTrade", handle_trade)
hub.start()
hub.invoke("SubscribeContractQuotes", contract_id)
```

Each `GatewayQuote` provides `lastPrice`, `bestBid`, `bestAsk` — fed into the same `BarAggregator` we already have.

---

## .env Configuration

```env
# Broker selection
BROKER=projectx              # "projectx" or "tradovate"

# ProjectX / TopstepX credentials
PROJECTX_USER=your_username
PROJECTX_API_KEY=your_api_key

# Tradovate credentials — SET IN .env, NEVER COMMIT VALUES
TRADOVATE_USER=<your_tradovate_username>
TRADOVATE_PASS=<your_tradovate_api_password>
TRADOVATE_APP_ID=<your_app_id>
TRADOVATE_APP_VERSION=1.0
TRADOVATE_CID=<your_cid>
TRADOVATE_SEC=<your_api_secret>
```

### CLI Override

```bash
# Uses BROKER from .env
python scripts/run_live_session.py --instrument MGC --signal-only

# Override broker for this run
python scripts/run_live_session.py --instrument MGC --broker projectx --signal-only
python scripts/run_live_session.py --instrument MGC --broker tradovate --demo
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SignalR library compatibility on Windows | Test `signalrcore` first; fallback to `pysignalr` |
| ProjectX contract IDs unknown for our instruments | First auth call fetches available contracts, logs mapping |
| Token expiry mid-session (24h) | Validate endpoint called proactively; auto-refresh on 401 |
| Different error formats between brokers | Normalize to common exceptions in each implementation |
| Backwards compatibility during transition | Old import paths re-export from new locations |
| No ProjectX sandbox for testing | Use signal-only mode first; then demo with small size |

## Testing

- Unit tests for each ABC implementation (mock HTTP responses)
- Integration test: auth → list accounts → list contracts (live API, ProjectX)
- Signal-only session test: connect feed → receive bars → generate signals (no orders)
- Position reconciliation test: simulate crash mid-session, verify orphan detection on restart
- Drift check: ensure `BROKER` env var is set and valid
- Drift check: ensure all `trading_app/live/*.py` files use ABCs not concrete broker classes

## Dependencies

- `signalrcore` or `pysignalr` — for ProjectX SignalR market data (MUST test on Windows first — known compat issues)
- All other dependencies already in project (`requests`, `websockets`, `python-dotenv`)

## M2.5 Design Review — Triage Summary (Mar 6 2026)

12 findings. 5 TRUE, 3 PARTIALLY TRUE, 2 FALSE POSITIVE, 1 WORTH EXPLORING.

**Incorporated into design:**
- P0: `BrokerPositions` ABC added (position reconciliation on start + EOD)
- P0: Proactive token refresh (`refresh_if_needed()` on BrokerAuth)
- P1: Native bracket support (`supports_native_brackets()` on BrokerRouter)
- P1: Contract ID mapping marked MUST VERIFY (not hardcoded)
- P1: Thin re-export shims required in migration plan

**Rejected:**
- Exception hierarchy — over-engineering for 2 brokers
- Historical contract lookup — not needed for live trading
