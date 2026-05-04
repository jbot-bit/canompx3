# ProjectX Gateway API — Canonical Reference
**Source:** https://gateway.docs.projectx.com  
**Fetched:** 2026-03-25  
**Platform:** TopstepX (The Futures Desk)  
**Purpose:** Ground-truth reference for canompx3 compliance audits.  
All code must be verified against this document before going live.

---

## BASE URLS

| Purpose | URL |
|---|---|
| REST API | `https://api.thefuturesdesk.projectx.com` |
| User Hub (SignalR) | `https://rtc.thefuturesdesk.projectx.com/hubs/user` |
| Market Hub (SignalR) | `https://rtc.thefuturesdesk.projectx.com/hubs/market` |

> **Critical:** These are the TopstepX-specific URLs. Other prop firms using ProjectX
> have different base URLs. Never use `api.topstepx.com` or `api.projectx.com` — those
> are wrong for this platform.

---

## AUTHENTICATION

### Login with API Key
`POST /api/Auth/loginKey`

**Request:**
```json
{
  "userName": "string",
  "apiKey": "string"
}
```

**Response:**
```json
{
  "token": "your_session_token_here",
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

**Rules:**
- Token is valid for **24 hours**
- Store token securely, use as Bearer token on all subsequent requests
- Check `success == true` AND `errorCode == 0` before using the token

---

### Validate / Refresh Token
`POST /api/Auth/validate`

No request body required (token is read from Authorization header).

**Response:**
```json
{
  "success": true,
  "errorCode": 0,
  "errorMessage": null,
  "newToken": "NEW_TOKEN"
}
```

**Rules:**
- Call this before token expires (recommended: refresh at 23h, 1h before expiry)
- Use `newToken` from response going forward
- All SignalR streams must also receive the new token via their token factory

---

## RATE LIMITS

| Endpoint | Limit |
|---|---|
| `POST /api/History/retrieveBars` | 50 requests / 30 seconds |
| All other endpoints | 200 requests / 60 seconds |

**If exceeded:** HTTP `429 Too Many Requests`. Back off and retry.

---

## ACCOUNTS

### Search Accounts
`POST /api/Account/search`

**Request:**
```json
{
  "onlyActiveAccounts": true
}
```

**Response:**
```json
{
  "accounts": [
    {
      "id": 1,
      "name": "TEST_ACCOUNT_1",
      "balance": 50000,
      "canTrade": true,
      "isVisible": true
    }
  ],
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

**Account fields:**

| Field | Type | Description |
|---|---|---|
| id | integer | Account ID — use this in all other API calls |
| name | string | Display name |
| balance | number | Current account balance |
| canTrade | bool | Whether account can currently trade |
| isVisible | bool | Whether account is visible in platform |

> **Note:** The `simulated` field does NOT appear in the REST search response.
> It appears in the SignalR `GatewayUserAccount` event. Use `canTrade` to
> gate trading, not `simulated`.

---

## ORDERS

### Place an Order
`POST /api/Order/place`

**Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| accountId | integer | REQUIRED | Account ID |
| contractId | string | REQUIRED | Contract ID (e.g. `CON.F.US.MNQ.M26`) |
| type | integer | REQUIRED | Order type (see enum below) |
| side | integer | REQUIRED | 0=Bid(buy), 1=Ask(sell) |
| size | integer | REQUIRED | Number of contracts |
| limitPrice | decimal | optional | Limit price (nullable) |
| stopPrice | decimal | optional | Stop trigger price (nullable) |
| trailPrice | decimal | optional | Trail amount (nullable) |
| customTag | string | optional | **Must be unique per account** |
| stopLossBracket | object | optional | See bracket spec below |
| takeProfitBracket | object | optional | See bracket spec below |

**Bracket object spec:**

| Field | Type | Required | Description |
|---|---|---|---|
| ticks | integer | REQUIRED | Distance in ticks from fill price |
| type | integer | REQUIRED | Order type for the bracket leg |

**Response:**
```json
{
  "orderId": 9056,
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

> **CRITICAL — Bracket architecture:**
> The response returns **ONE orderId** — the entry order ID only.
> Bracket legs (SL and TP) do **NOT** get separate IDs in the place response.
> They appear as separate open orders in `/api/Order/searchOpen` AFTER the
> entry order fills. To verify brackets exist, call searchOpen after fill
> and look for orders you didn't place directly (filter by type and price).

---

### Cancel an Order
`POST /api/Order/cancel`

**Request:**
```json
{
  "accountId": 465,
  "orderId": 26974
}
```

**Response:**
```json
{
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

> **Critical:** Both `accountId` AND `orderId` are required. Sending only `orderId` will fail.

---

### Search Open Orders
`POST /api/Order/searchOpen`

**Request:**
```json
{
  "accountId": 212
}
```

**Response:**
```json
{
  "orders": [
    {
      "id": 26970,
      "accountId": 212,
      "contractId": "CON.F.US.EP.M25",
      "creationTimestamp": "2025-04-21T19:45:52.105808+00:00",
      "updateTimestamp": "2025-04-21T19:45:52.105808+00:00",
      "status": 1,
      "type": 4,
      "side": 1,
      "size": 1,
      "limitPrice": null,
      "stopPrice": 5138.000000000,
      "filledPrice": null
    }
  ],
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

**Order fields:**

| Field | Type | Description |
|---|---|---|
| id | integer | Order ID |
| accountId | integer | Account ID |
| contractId | string | Contract ID |
| creationTimestamp | string | ISO 8601 UTC timestamp |
| updateTimestamp | string | ISO 8601 UTC timestamp |
| status | integer | OrderStatus enum |
| type | integer | OrderType enum |
| side | integer | 0=Bid, 1=Ask |
| size | integer | Contracts |
| limitPrice | decimal/null | Limit price |
| stopPrice | decimal/null | Stop price |
| filledPrice | decimal/null | Fill price (null if not filled) |

> **Note:** There is NO `customTag` field in the searchOpen response.
> Bracket verification must use type + stopPrice + creationTimestamp,
> not customTag.

---

### Search Orders (Historical)
`POST /api/Order/search`

Searches historical orders (not just open ones). Use for post-session reconciliation.

---

### Modify an Order
`POST /api/Order/modify`

Used to change price/size of an existing open order.

---

## POSITIONS

### Search Open Positions
`POST /api/Position/searchOpen`

**Request:**
```json
{
  "accountId": 536
}
```

**Response:**
```json
{
  "positions": [
    {
      "id": 6124,
      "accountId": 536,
      "contractId": "CON.F.US.GMET.J25",
      "creationTimestamp": "2025-04-21T19:52:32.175721+00:00",
      "type": 1,
      "size": 2,
      "averagePrice": 1575.750000000
    }
  ],
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

**Position fields:**

| Field | Type | Description |
|---|---|---|
| id | integer | Position ID |
| accountId | integer | Account ID |
| contractId | string | Contract ID |
| creationTimestamp | string | ISO 8601 UTC |
| type | integer | PositionType enum: 1=Long, 2=Short |
| size | integer | Number of contracts |
| averagePrice | decimal | Average fill price |

> **CRITICAL:** There is **NO unrealizedPnL field** in the position response.
> Mark-to-market P&L must be computed manually:
> `unrealized_pnl = (current_price - averagePrice) * size * tick_value`
> For short positions: `(averagePrice - current_price) * size * tick_value`

---

### Close Position (by contract)
`POST /api/Position/closeContract`

**Request:**
```json
{
  "accountId": 536,
  "contractId": "CON.F.US.GMET.J25"
}
```

**Response:**
```json
{
  "success": true,
  "errorCode": 0,
  "errorMessage": null
}
```

> Closes the entire position for that contract at market.
> Use for emergency flatten / session-end close.

---

### Partially Close Position
`POST /api/Position/closeContractPartial`

Closes a specified number of contracts. Used for scaling out.

---

## ENUMS — OFFICIAL VALUES

### OrderType
```
0 = Unknown
1 = Limit
2 = Market
3 = StopLimit
4 = Stop         ← Stop-market (use for E2 entries and SL brackets)
5 = TrailingStop
6 = JoinBid
7 = JoinAsk
```

> **Common mistake:** Using type=3 (StopLimit) instead of type=4 (Stop).
> StopLimit requires BOTH a stop price AND a limit price.
> Stop (type=4) is a pure stop-market — triggers at stop price, fills at market.

### OrderSide
```
0 = Bid  (buy / long)
1 = Ask  (sell / short)
```

### OrderStatus
```
0 = None
1 = Open
2 = Filled
3 = Cancelled
4 = Expired
5 = Rejected     ← DISTINCT from Cancelled — must handle separately
6 = Pending
```

> **Critical:** A Rejected order (5) must be handled separately from Cancelled (3).
> A rejected E2 stop-market entry is NOT the same as a cancelled one.
> If only checking for status=3, a rejected entry creates a ghost trade.

### PositionType
```
0 = Undefined
1 = Long
2 = Short
```

> **Common mistake:** Treating Long=0, Short=1. Actual values are Long=1, Short=2.

### TradeLogType
```
0 = Buy
1 = Sell
```

### DomType
```
0  = Unknown
1  = Ask
2  = Bid
3  = BestAsk
4  = BestBid
5  = Trade
6  = Reset
7  = Low
8  = High
9  = NewBestBid
10 = NewBestAsk
11 = Fill
```

---

## SIGNALR — REALTIME

### Connection Setup

```python
# Python (pysignalr — preferred)
from pysignalr.client import SignalRClient

user_hub_url = "https://rtc.thefuturesdesk.projectx.com/hubs/user"
client = SignalRClient(
    url=f"{user_hub_url}?access_token={jwt_token}",
    skip_negotiation=True
)
```

**Rules:**
- `skipNegotiation: True` — required
- Transport: WebSockets only
- Token passed as query param: `?access_token=JWT_TOKEN`
- Must re-invoke all Subscribe* methods after reconnect

---

### User Hub

**URL:** `https://rtc.thefuturesdesk.projectx.com/hubs/user`

**Subscribe methods (invoke after connect AND after reconnect):**
```
SubscribeAccounts
SubscribeOrders(accountId)
SubscribePositions(accountId)
SubscribeTrades(accountId)
```

**Unsubscribe methods:**
```
UnsubscribeAccounts
UnsubscribeOrders(accountId)
UnsubscribePositions(accountId)
UnsubscribeTrades(accountId)
```

**Events received:**

#### GatewayUserAccount
```json
{
  "id": 123,
  "name": "Main Trading Account",
  "balance": 10000.50,
  "canTrade": true,
  "isVisible": true,
  "simulated": false
}
```

#### GatewayUserOrder
```json
{
  "id": 789,
  "accountId": 123,
  "contractId": "CON.F.US.EP.U25",
  "symbolId": "F.US.EP",
  "creationTimestamp": "2024-07-21T13:45:00Z",
  "updateTimestamp": "2024-07-21T13:46:00Z",
  "status": 1,
  "type": 1,
  "side": 0,
  "size": 1,
  "limitPrice": 2100.50,
  "stopPrice": null,
  "fillVolume": 0,
  "filledPrice": null,
  "customTag": "strategy-1"
}
```

> **Fill detection:** A filled order has `status=2`, `filledPrice != null`, `fillVolume > 0`.
> Check all three fields — do not rely on status alone.

#### GatewayUserPosition
```json
{
  "id": 456,
  "accountId": 123,
  "contractId": "CON.F.US.EP.U25",
  "creationTimestamp": "2024-07-21T13:45:00Z",
  "type": 1,
  "size": 2,
  "averagePrice": 2100.25
}
```

#### GatewayUserTrade
```json
{
  "id": 101112,
  "accountId": 123,
  "contractId": "CON.F.US.EP.U25",
  "creationTimestamp": "2024-07-21T13:47:00Z",
  "price": 2100.75,
  "profitAndLoss": 50.25,
  "fees": 2.50,
  "side": 0,
  "size": 1,
  "voided": false,
  "orderId": 789
}
```

> **GatewayUserTrade vs GatewayUserOrder:**
> Use GatewayUserOrder (status=2/Filled) for fill detection.
> GatewayUserTrade fires when a trade executes but has less order context.
> For position management, GatewayUserOrder is the primary event.

---

### Market Hub

**URL:** `https://rtc.thefuturesdesk.projectx.com/hubs/market`

**Subscribe methods (invoke after connect AND after reconnect):**
```
SubscribeContractQuotes(contractId)
SubscribeContractTrades(contractId)
SubscribeContractMarketDepth(contractId)
```

**Events received:**

#### GatewayQuote
```json
{
  "symbol": "F.US.EP",
  "symbolName": "/ES",
  "lastPrice": 2100.25,
  "bestBid": 2100.00,
  "bestAsk": 2100.50,
  "change": 25.50,
  "changePercent": 0.14,
  "open": 2090.00,
  "high": 2110.00,
  "low": 2080.00,
  "volume": 12000,
  "lastUpdated": "2024-07-21T13:45:00Z",
  "timestamp": "2024-07-21T13:45:00Z"
}
```

> `symbolId` (e.g. `F.US.MNQ`) is different from `contractId` (e.g. `CON.F.US.MNQ.M26`).
> Market hub subscribes use `contractId`. Quote events return `symbol` (the symbolId).

#### GatewayTrade (market trade, not user trade)
```json
{
  "symbolId": "F.US.EP",
  "price": 2100.25,
  "timestamp": "2024-07-21T13:45:00Z",
  "type": 0,
  "volume": 2
}
```

#### GatewayDepth (DOM)
```json
{
  "timestamp": "2024-07-21T13:45:00Z",
  "type": 1,
  "price": 2100.00,
  "volume": 10,
  "currentVolume": 5
}
```

---

## COMMON PITFALLS

| # | Pitfall | Correct behaviour |
|---|---|---|
| 1 | Using wrong base URL | Must use `thefuturesdesk.projectx.com`, not `topstepx.com` or `projectx.com` |
| 2 | type=3 for stop-market | Use type=4 (Stop). Type=3 is StopLimit (needs limit price too) |
| 3 | Expecting bracket IDs in place response | Only entry orderId returned. Brackets appear in searchOpen after fill |
| 4 | Non-unique customTag | Tag must be unique per account across all orders, ever. Use timestamp+strategy hash |
| 5 | Cancel without accountId | Cancel requires BOTH accountId and orderId |
| 6 | Long=0, Short=1 | Actual: Long=1, Short=2 (PositionType enum) |
| 7 | Cancelled=Rejected | Status=3 is Cancelled, status=5 is Rejected. Handle both |
| 8 | Unrealized PnL from API | No such field. Compute from averagePrice + current market price |
| 9 | Not re-subscribing after reconnect | Must invoke Subscribe* methods again after SignalR reconnects |
| 10 | Trusting HTTP 200 = success | Always check `response.success == true` in body |

---

## CONTRACT ID FORMAT

Format: `CON.F.{exchange}.{symbol}.{expiry}`

Examples:
- `CON.F.US.MNQ.M26` — Micro NQ, June 2026
- `CON.F.US.MGC.J26` — Micro Gold, April 2026
- `CON.F.US.MES.M26` — Micro ES, June 2026

Expiry codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

> Contract IDs change at rollover. Never hardcode.
> Always resolve front-month via `/api/Contract/search` at startup
> and re-resolve on every feed reconnect.

---

*End of canonical reference. Source: https://gateway.docs.projectx.com — fetched 2026-03-25.*
*Verify against live docs before any major release.*
