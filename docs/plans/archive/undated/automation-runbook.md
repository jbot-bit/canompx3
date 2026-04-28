---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Automation Runbook — First Automated Trade

Profile: `topstep_50k_mnq_auto` | MNQ COMEX_SETTLE | TopStep 50K | ProjectX API

## How to Start

### Signal-only (no orders — safest)
```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --signal-only
```

### Demo (paper orders on TopStep demo account)
```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --demo
```

### Live (real money — requires typing CONFIRM)
```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --live
```

### Preflight (verify everything before trading)
```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --preflight
```

## Before Every Session

1. **Data freshness:** Preflight check 3 catches this. If stale:
   ```bash
   python pipeline/daily_backfill.py  # needs fresh Databento DBN files
   ```

2. **Telegram notifications:** Set `TELEGRAM_CHAT_ID` in `.env` (one-time).

3. **No orphan positions:** Preflight check 5 verifies. If orphans exist:
   - Close manually on TopStep platform
   - Or pass `--force-orphans` (know the risk)

## How to Monitor

| Channel | What | URL/Command |
|---------|------|-------------|
| Dashboard | Live state, lanes, P&L | http://localhost:8080 (auto-launches) |
| Telegram | Entries, exits, errors | Your Telegram bot |
| Broker portal | Cross-reference fills | TopStep/ProjectX web UI |
| SPRT | Statistical drift check | `python -m trading_app.sprt_monitor` |
| Lane scorer | Strategy ranking | `python scripts/tools/score_lanes.py --current` |

## Session Schedule (Brisbane Time)

COMEX_SETTLE session times (DST-dependent):
- **Summer (Apr-Oct):** 03:30 AM Brisbane
- **Winter (Nov-Mar):** 04:30 AM Brisbane

The bot handles DST automatically via `pipeline.dst.SESSION_CATALOG`.

## How to Emergency Kill

| Method | Speed | Effect |
|--------|-------|--------|
| Dashboard KILL button | ~5s | Writes stop file, bot closes positions + shuts down |
| Ctrl+C in terminal | ~5s | Graceful SIGINT, closes positions |
| Kill process | Instant | Abrupt — check for orphan positions after |

After any kill: check broker portal for orphan positions.

## What to Check Each Morning

1. Dashboard heartbeat — was bot alive during COMEX_SETTLE session?
2. Telegram — any entry/exit/error notifications?
3. Broker P&L — matches dashboard?
4. `data/bot_state.json` — last heartbeat timestamp
5. `live_journal.db` — trades logged?

## Kill Criteria

| Condition | Action |
|-----------|--------|
| 3 consecutive losing months | Pause lane, investigate |
| Forward ExpR < 0.10 over 50+ trades | Remove lane |
| CUSUM 4-sigma sustained 2+ weeks | Remove lane |
| DD within 20% of firm limit ($400 remaining) | Downgrade to signal-only |
| Prop firm rule violation | FULL STOP |
| Dashboard/broker P&L mismatch | FULL STOP, investigate |

## Paper-to-Live Graduation

| Stage | Duration | Command | Criteria to advance |
|-------|----------|---------|---------------------|
| Signal-only | 2-3 sessions | `--signal-only` | Dashboard renders, signals match manual calc |
| Demo | 5-10 sessions | `--demo` | Orders execute, fills match, no orphans |
| Live (1 contract) | 30 trades | `--live` | >95% execution fidelity, slippage within E2 model |
| Scale (5 accounts) | 50+ trades | Multiple Express | Consistent fwd ExpR, no CUSUM alarms |

## Architecture Summary

```
ProjectX API (ticks)
  -> BarAggregator (1m bars)
    -> ExecutionEngine (ORB detection, filter check, entry/exit signals)
      -> OrderRouter (bracket orders) [signal-only: skipped]
        -> PositionTracker (reconciliation)
          -> TradeJournal (live_journal.db)
            -> PerformanceMonitor (CUSUM drift)

Dashboard (bot_state.json) <- written every bar
Notifications (Telegram) <- on entry/exit/error/kill
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Strategy | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR70_VOL |
| Family | ROBUST (7 members, PBO=0.000) |
| Sample size | 469 trades |
| Win rate | 66.7% |
| ExpR | 0.215 |
| Sharpe | 1.75 |
| 2025 forward | +25.7R (N=63) |
| Risk per trade | ~$29 (median, with SM=0.75) |
| DD budget | $935 / $2,000 (47%) |
| DLL budget | $29 / $1,000 (2.9%) |
| ORB cap | 80 pts (skip if risk > 80 pts) |
| Trades per year | ~50 |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Preflight check 3 fails | Daily features stale | Run pipeline rebuild |
| "No portfolio injected" | Wrong --profile | Use `--profile topstep_50k_mnq_auto` |
| No ticks flowing | Market closed or CME maintenance | Wait; CME break 4-5 PM CT daily |
| Filter rejects every day | atr_20_pct NULL in DB | Rebuild daily_features with fresh bars |
| "ORPHANED POSITIONS" on start | Prior crash left open position | Close on broker portal, restart |
| Notifications not firing | TELEGRAM_CHAT_ID missing | Add to .env |
