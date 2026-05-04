---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# TopstepX Live Bot — Preflight Checklist

## Prerequisites
- Funded TopstepX account active and visible
- This machine running (local execution only — TopStep rules)
- Dashboard: http://localhost:8080 (auto-launches with bot, or standalone: `python -m trading_app.live.bot_dashboard`)
- Terminal open in `C:\Users\joshd\canompx3`

---

## Step 1: Verify the funded account is visible

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from trading_app.live.projectx.auth import ProjectXAuth, BASE_URL
import requests
auth = ProjectXAuth()
resp = requests.post(f'{BASE_URL}/api/Account/search', json={'onlyActiveAccounts': True}, headers=auth.headers(), timeout=10)
for acct in resp.json():
    sim = 'SIM' if acct.get('simulated') else 'FUNDED'
    trade = 'CAN TRADE' if acct.get('canTrade') else 'CANNOT TRADE'
    print(f'  [{sim}] [{trade}] id={acct[\"id\"]} name={acct[\"name\"]} balance=\${acct[\"balance\"]:,.2f}')
"
```

**PASS criteria:** You see a line with `[FUNDED] [CAN TRADE]`. Note the `id` — you'll need it if auto-discovery picks the wrong account.

---

## Step 2: Run preflight checks (5-point)

```bash
python -m scripts.run_live_session --profile apex_50k_manual --preflight
```

**PASS criteria:** `Preflight: 5/5 passed`

If portfolio check shows the wrong account (sim instead of funded), specify it:
```bash
python -m scripts.run_live_session --profile apex_50k_manual --preflight --account-id <FUNDED_ACCOUNT_ID>
```

---

## Step 3: Start bot in PAPER mode (signal-only)

```bash
python -m scripts.run_live_session --profile apex_50k_manual --signal-only
```

This connects to the market data feed and generates trade signals but places **NO orders**. Watch the log for:
- `Session ready: MNQ -> CON.F.US.MNQ.M26 (SIGNAL-ONLY)`
- `Subscribed to quotes` messages
- BAR lines appearing every minute during market hours

Let it run for 5-10 minutes to confirm data is flowing.

---

## Step 4: Verify a signal cycle works (demo mode)

```bash
python -m scripts.run_live_session --profile apex_50k_manual --demo
```

This connects to the **sim account** and places real orders on the sim. Watch for:
- `Session ready: MNQ -> CON.F.US.MNQ.M26 (DEMO)`
- When a session ORB forms and breaks, you should see ENTRY events with bracket orders
- Check Telegram for notifications

**IMPORTANT:** Demo mode uses the first active account (sim). This is safe — no real money.

---

## Step 5: Switch to LIVE mode

```bash
python -m scripts.run_live_session --profile apex_50k_manual --live --account-id <FUNDED_ACCOUNT_ID>
```

You will be prompted to type `CONFIRM` before any real orders are placed.

**The `--account-id` flag is CRITICAL** — without it, the bot auto-discovers the first active account, which may be a sim account. Always specify the funded account ID explicitly.

---

## Kill Switch

### Method 1: Stop file (graceful — recommended)
```bash
echo stop > live_session.stop
```
The bot checks for this file every 2.5 seconds. When found, it:
1. Stops accepting new signals
2. Closes any open positions at market
3. Shuts down cleanly
4. Deletes the stop file

### Method 2: Ctrl+C (immediate)
Press `Ctrl+C` in the terminal. The bot runs `post_session()` cleanup on exit.

### Method 3: Kill process (emergency only)
```bash
taskkill /F /IM python.exe
```
**WARNING:** This does NOT close open positions. After using this, immediately:
1. Log into TopstepX platform manually
2. Close any open positions
3. Cancel any working orders

---

## The 4 Apex Lanes

| Lane | Session | Brisbane Time | Filter | RR | Action |
|------|---------|---------------|--------|-----|--------|
| 1 | NYSE_CLOSE | 6:00/7:00 AM | VOL_RV12_N20 | 1.0 | Auto |
| 2 | SINGAPORE_OPEN | 11:00 AM | ORB_G8 | 4.0 | Auto |
| 3 | COMEX_SETTLE | 3:30/4:30 AM | ORB_G8 | 1.0 | Auto (alarm) |
| 4 | NYSE_OPEN | 11:30 PM | X_MES_ATR60 | 1.0 | Auto |

All lanes: MNQ, E2 stop-market entry, 0.75x prop stop, 1 contract, bracket orders (SL+TP atomic).

---

## Daily Loss Limit

The risk manager enforces `max_daily_loss_r = -5.0R`. If daily P&L hits this, the bot **stops entering new trades** but continues monitoring open positions. This is in addition to TopStep's own drawdown rules.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Auth fails | API key expired | Regenerate key on TopstepX dashboard |
| 0 quotes received | Market closed (weekends, holidays) | Wait for market open |
| Feed dies repeatedly | SignalR connection issue | Check internet, restart bot |
| "No active strategies" | Profile not found in validated_setups | Run `python -m scripts.run_live_session --profile apex_50k_manual --preflight` |
| Wrong account selected | Auto-discovery picked sim | Use `--account-id <FUNDED_ID>` |
| Bracket order rejected | Tick size mismatch | Check contract resolution in preflight |

---

## End-to-End Sim Test (run anytime to verify)

```bash
python -m scripts.e2e_sim_test
```

All 7 tests must pass before going live.
