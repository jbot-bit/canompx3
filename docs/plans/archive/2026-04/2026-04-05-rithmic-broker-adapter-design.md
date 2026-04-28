---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Rithmic Broker Adapter — Design Document V2

**Date:** 2026-04-05
**Status:** V2 FINAL — verified against async_rithmic 1.5.9 source
**Author:** Claude (autonomous session)

## Purpose

One Rithmic API integration unlocks two durable prop firm lanes:
- **Bulenox:** 3 simultaneous Master accounts, 11 total, no forced conversion, bots allowed
- **Elite Trader Funding:** 100% profit split, bots+copy allowed, Rithmic platform

Unlike TopStep Express (forced Live at 3-5 payouts), Bulenox has NO forced conversion.
This is the long-term automated scaling lane.

## Architecture

### Scope: Order-Only Adapter (like Tradovate)

```
ProjectX DataFeed (market data)
    |  1-minute bars
BarAggregator -> ExecutionEngine
    |  TradeEvent
BrokerDispatcher
    +-- ProjectXOrderRouter (TopStep - primary)
    +-- RithmicOrderRouter (Bulenox - secondary, order-only)
```

- `feed_class: None` -- uses ProjectX master feed for market data
- Order routing via Rithmic ORDER_PLANT (Protocol Buffer over WebSocket)
- Server-side brackets (stops/targets survive client crash)

Future: Add Rithmic TICKER_PLANT for standalone operation without ProjectX.

## Verified async_rithmic 1.5.9 API Surface

### Connection
```
RithmicClient(user, password, system_name, app_name, app_version, url, manual_or_auto=AUTO)
await client.connect(plants=[ORDER_PLANT, PNL_PLANT])  # skip TICKER for order-only
await client.disconnect()
```

### Order Submission (submit_order handles ALL order types including brackets)
- Market: `submit_order(order_id, symbol, exchange, qty, TransactionType.BUY, OrderType.MARKET)`
- Stop: `submit_order(..., OrderType.STOP_MARKET, trigger_price=X)`
- With bracket: add `stop_ticks=N, target_ticks=M` -- auto-switches to template 330
- Brackets are ATOMIC (entry + bracket in one call, server-side)

### Order Enums
- OrderType: LIMIT=1, MARKET=2, STOP_LIMIT=3, STOP_MARKET=4, MARKET_IF_TOUCHED=5, LIMIT_IF_TOUCHED=6
- TransactionType: BUY=1, SELL=2
- OrderDuration: DAY=1, GTC=2, IOC=3, FOK=4
- OrderPlacement: MANUAL=1, AUTO=2

### Order Management
- `cancel_order(order_id=... or basket_id=...)`
- `cancel_all_orders(account_id=...)`
- `modify_order(order_id=..., stop_ticks=..., target_ticks=...)`
- `list_orders(account_id=...)`
- `exit_position(symbol=..., exchange=..., account_id=...)` -- market flatten
- `get_order(order_id=... or basket_id=...)`

### Positions / PnL
- `list_positions(account_id=...)` -- open positions
- `list_account_summary(account_id=...)` -- equity/balance
- `subscribe_to_pnl_updates()` -- real-time PnL callbacks

### Accounts
- `client.accounts` -- auto-populated on login (list of account objects)
- Each account has `.account_id` (string)
- `client.fcm_id`, `client.ib_id` -- from login info

### Contracts
- `get_front_month_contract(symbol, exchange)` -- e.g. ("MES", "CME") -> "MESM6"

### Events (callback-based)
- `on_exchange_order_notification` -- fill/reject/cancel (types: STATUS=1, FILL=5, REJECT=6, CANCEL=3)
- `on_bracket_update` -- bracket modifications
- `on_connected`, `on_disconnected`

## Key Design Decisions (V2 refined)

| Decision | Rationale |
|----------|-----------|
| async_rithmic 1.5.9 | Production-stable (Feb 2026), 146 commits, auto-reconnect, MIT |
| OrderPlacement.AUTO | Required for prop firm automated trading |
| Atomic brackets via submit_order kwargs | stop_ticks + target_ticks on entry order = server-side brackets |
| Event-driven fill tracking | on_exchange_order_notification caches fills; query_order_status reads cache |
| exit_position() for kill switch | One-call emergency flatten via Rithmic server |
| Background asyncio thread | Daemon thread runs RithmicClient event loop; sync methods use run_coroutine_threadsafe |
| account_id string->int | Rithmic uses string IDs; convert via int() for ABC compatibility |
| Connect ORDER_PLANT + PNL_PLANT only | Skip TICKER (ProjectX handles data) and HISTORY (not needed live) |
| Prop profiles active=False | Deploy code, test on paper, then activate |

## Implementation Stages

### Stage 1: Foundation (auth + contracts + factory + __init__)

**NEW: trading_app/live/rithmic/__init__.py**
- Re-export public classes

**NEW: trading_app/live/rithmic/auth.py -- RithmicAuth(BrokerAuth)**
- Env vars: RITHMIC_USER, RITHMIC_PASSWORD, RITHMIC_SYSTEM_NAME, RITHMIC_APP_NAME, RITHMIC_GATEWAY
- Creates RithmicClient with OrderPlacement.AUTO
- Spawns daemon thread with asyncio event loop
- connect() on first use (lazy), connects ORDER_PLANT + PNL_PLANT only
- is_healthy property tracks connection state
- get_token() returns placeholder (Rithmic is connection-based, not token-based)
- headers() returns empty dict
- refresh_if_needed() triggers reconnect check
- Exposes run_async(coro) method to bridge async->sync via run_coroutine_threadsafe
- Exposes client property for direct access by other Rithmic components

**NEW: trading_app/live/rithmic/contracts.py -- RithmicContracts(BrokerContracts)**
- resolve_account_id() -> int: first account from client.accounts, int() conversion
- resolve_all_account_ids() -> list[tuple[int, str]]: all accounts for copy trading
- resolve_front_month(instrument) -> str: get_front_month_contract(root, "CME")
- Instrument mapping: {"MES": "MES", "MNQ": "MNQ", "MGC": "MGC"} (root symbols match)

**MODIFIED: trading_app/live/broker_factory.py**
- Add "rithmic" to VALID_BROKERS
- New elif branch: lazy-import Rithmic components, return with feed_class=None

### Stage 2: Order Router (core trading logic)

**NEW: trading_app/live/rithmic/order_router.py -- RithmicOrderRouter(BrokerRouter)**

Constructor:
- account_id (int), auth (RithmicAuth), tick_size, exchange ("CME")
- Stores _rithmic_account_id as string version
- Price collar check (same 0.5% logic as ProjectX/Tradovate)
- _order_cache dict for tracking basket_id -> fill_price mappings
- Registers on_exchange_order_notification callback to update cache

build_order_spec(direction, entry_model, entry_price, symbol, qty):
- E1 -> dict with order_type=MARKET, transaction_type from direction
- E2 -> dict with order_type=STOP_MARKET, trigger_price=entry_price
- Attaches _intent dict for BrokerDispatcher cross-broker routing
- Returns broker-agnostic spec dict

submit(spec):
- Price collar check on trigger_price (if present)
- Generates unique order_id string (timestamp + random)
- Extracts order parameters from spec
- Calls auth.run_async(client.submit_order(...)) to bridge async->sync
- If bracket fields present, includes stop_ticks and target_ticks
- Logs full payload before and response after (audit trail)
- Returns {order_id, status, fill_price} matching standard format
- Tracks basket_id from response for future queries

build_exit_spec(direction, symbol, qty):
- Market order in opposite direction
- Attaches _exit_intent for BrokerDispatcher

cancel(order_id):
- Calls cancel_order via async bridge

supports_native_brackets() -> True:
- Rithmic has SERVER-SIDE brackets

build_bracket_spec(direction, symbol, entry_price, stop_price, target_price, qty):
- Calculates stop_ticks and target_ticks from prices and tick_size
- Returns dict with stop_ticks, target_ticks

merge_bracket_into_entry(entry_spec, bracket_spec):
- Merges bracket fields into spec (stop_ticks, target_ticks become kwargs to submit_order)

query_order_status(order_id):
- Reads from _order_cache (populated by exchange notification callback)
- Fallback: calls get_order via async bridge

query_open_orders():
- Calls list_orders via async bridge

cancel_bracket_orders(contract_id):
- Lists orders, cancels bracket legs matching contract

### Stage 3: Positions + Profiles

**NEW: trading_app/live/rithmic/positions.py -- RithmicPositions(BrokerPositions)**
- query_open(account_id) -> list[dict]: calls list_positions, formats to standard
- query_equity(account_id) -> float: calls list_account_summary, extracts balance

**MODIFIED: trading_app/prop_profiles.py**
- Add "elite" PropFirmSpec (conservative defaults, needs verification)
- Add "bulenox_50k" AccountProfile (3 copies, same lanes as topstep_50k_mnq_auto, active=False)

### Stage 4: Tests

**NEW: tests/test_trading_app/test_rithmic_router.py**
Unit tests (no network, mock async_rithmic):
- build_order_spec E1 market: correct OrderType.MARKET + TransactionType
- build_order_spec E2 stop: correct OrderType.STOP_MARKET + trigger_price
- build_order_spec invalid entry model: raises ValueError
- build_exit_spec: direction reversal correct
- build_bracket_spec: tick calculation for long and short
- merge_bracket_into_entry: stop_ticks/target_ticks present in merged spec
- Price collar: rejects >0.5% deviation, passes within tolerance
- broker_factory: "rithmic" in VALID_BROKERS
- broker_factory: create_broker_components("rithmic") returns correct types

## Blast Radius

- 5 new files: rithmic/{__init__, auth, order_router, contracts, positions}.py
- 1 new test file: test_rithmic_router.py
- 2 modified files: broker_factory.py (add rithmic branch), prop_profiles.py (add profiles)
- Zero pipeline changes, zero schema changes, zero DB changes
- One-way dependency preserved (trading_app only)

## Failure Modes

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| async_rithmic breaks | Low | Pin v1.5.9, pyrithmic fallback |
| No Rithmic credentials yet | High | Paper trading env, profile active=False |
| Async->sync deadlock | Medium | Dedicated daemon thread, 10s timeout on bridge calls |
| Connection drop w/ open position | Low | Server-side brackets protect position |
| Contract roll mid-session | Very Low | get_front_month_contract() + cache |
| Account ID not purely numeric | Low | Try int(), fallback to hash |
| Conformance test not passed | Medium | All code works against paper trading; conformance is operational not code |

## Rollback

All changes additive. Delete new files + revert factory/profiles = zero impact on existing paths.
