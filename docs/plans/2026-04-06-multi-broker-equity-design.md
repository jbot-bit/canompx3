# Multi-Broker Account Management — Design Doc

**Date:** 2026-04-06
**Status:** APPROVED
**Phase:** 1 (ProjectX only, architecture ready for multi-broker)

## Blocking Issues Resolved

1. Sync `requests.post` in async handler → `asyncio.to_thread()`
2. `simulated` field not in REST → dropped, use `canTrade` + `isVisible`
3. DD uses starting balance not HWM → client-side HWM in `data/account_hwm.json`
4. Errors swallowed → error banner + stale indicator
5. Single try/except kills all brokers → per-broker isolation
6. New auth per request → module-level singleton with threading.Lock

## API Response Schema

```
GET /api/equity
{
  "brokers": [{
    "name": "projectx",
    "display": "TopStepX",
    "error": null,
    "accounts": [{
      "id": 20092334,
      "name": "100KTC-V2-451890-79617502",
      "balance": 103034.09,
      "hwm": 103500.00,
      "can_trade": true,
      "is_visible": true,
      "status": "tradeable"
    }]
  }],
  "fetched_at": "2026-04-06T13:00:00+10:00"
}
```

## Files Changed

- `trading_app/live/bot_dashboard.py` — rewrite `/api/equity`
- `trading_app/live/bot_dashboard.html` — account selector, error banner, HWM DD
