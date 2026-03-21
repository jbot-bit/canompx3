# Paper Trading Deployment — MNQ RR1.0 Raw Baseline

**Date:** 2026-03-21
**Status:** READY TO DEPLOY
**Gate:** 7 (Paper Trade with Kill Criteria)

## What

Deploy MNQ E2 RR1.0 raw baseline strategies in signal-only mode to collect 2026 forward data. This is the binding forward test for the 3 pre-registered sessions (NYSE_OPEN, COMEX_SETTLE, CME_PRECLOSE).

## Pre-Deployment Checklist (all DONE)

| Item | Status | Evidence |
|------|--------|----------|
| Pre-registration doc | DONE | `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md` |
| Kill criteria defined | DONE | Same doc: 3 thresholds + OBF sequential |
| Portfolio builds from canonical source | DONE | `build_raw_baseline_portfolio()` → 11 sessions |
| filter_type matches ALL_FILTERS | DONE | `NO_FILTER` — pass-through |
| Replay validation (Gate 6) | DONE | 2025: 2606 trades, +272.56R, 59.4% WR, max DD -15.34R |
| Cost model verified | DONE | MNQ $2.74 RT from `get_cost_spec("MNQ")` |
| Kill criteria monitor | DONE | `scripts/tools/check_kill_criteria.py` |
| 2026 holdout is SACRED | ACKNOWLEDGED | No analysis on 2026 data — paper trading IS the forward test |

## Deployment Commands

### RECOMMENDED: Daily batch forward test (no live connection needed)
```bash
# Full refresh: download from Databento -> ingest -> features -> outcomes -> kill check
python scripts/tools/forward_test.py

# Check-only (skip download, just check kill criteria on existing data)
python scripts/tools/forward_test.py --check-only

# Dry run (show what would be downloaded)
python scripts/tools/forward_test.py --dry-run
```

Run whenever you open your PC. Catches up on all missed days automatically.
Requires DATABENTO_API_KEY in .env. No live connection, no WebSocket, no broker auth.

### ALTERNATIVE: Live signal session (requires always-on connection)
```bash
python scripts/run_live_session.py --instrument MNQ --signal-only --raw-baseline
python scripts/run_live_session.py --instrument MNQ --signal-only --raw-baseline --stop-multiplier 0.75
```

Use for real-time signals when manually trading on Apex. Needs broker auth (ProjectX).
PC off = signals off. Only needed when you're actively placing manual orders.

### Kill criteria check (standalone)
```bash
# Batch mode (reads orb_outcomes from gold.db)
python scripts/tools/check_kill_criteria.py --from-outcomes

# Live mode (reads live_journal.db from broker sessions)
python scripts/tools/check_kill_criteria.py
```

### Historical replay validation (re-run if code changes)
```bash
python -m trading_app.paper_trader --instrument MNQ --raw-baseline --rr-target 1.0 --start 2025-01-01 --end 2025-12-31 --quiet
```

## Portfolio Composition

11 sessions, all MNQ E2 CB1 RR1.0 O5 NO_FILTER. NYSE_CLOSE excluded (30.8% WR).

| Session | 2025 Trades | 2025 ExpR | 2025 WR | Pre-Registered |
|---------|-------------|-----------|---------|----------------|
| COMEX_SETTLE | 236 | +0.202R | 64.0% | YES |
| CME_PRECLOSE | 235 | +0.159R | 62.2% | YES |
| EUROPE_FLOW | 256 | +0.131R | 61.3% | |
| NYSE_OPEN | 249 | +0.134R | 58.4% | YES |
| US_DATA_1000 | 240 | +0.136R | 58.8% | |
| US_DATA_830 | 247 | +0.077R | 57.1% | |
| LONDON_METALS | 255 | +0.075R | 57.6% | |
| SINGAPORE_OPEN | 257 | +0.063R | 58.8% | |
| TOKYO_OPEN | 203 | +0.072R | 57.6% | |
| CME_REOPEN | 170 | +0.058R | 59.5% | |
| BRISBANE_1025 | 258 | +0.035R | 58.5% | |

## Kill Criteria (FROZEN — from pre-registration)

1. **Per-session:** After 100 trades in a session, if ExpR < +0.03R → STOP that session
2. **Slippage:** After 100 trades, if avg slippage > 3 ticks → STOP (cost model wrong)
3. **Portfolio:** After 200 total trades, if combined ExpR < +0.05R → STOP everything
4. **Sequential:** O'Brien-Fleming boundaries checked monthly (early stopping for extreme underperformance)

## Monitoring Cadence

| When | Action |
|------|--------|
| Weekly | Run `check_kill_criteria.py`, review output |
| Monthly | Full OBF sequential review, check for regime drift |
| After 100 trades/session | Session-level kill criteria become active |
| After 200 total trades | Portfolio-level kill criteria become active |
| April 2026 (est.) | 2026 holdout test: N>=100 per pre-registered session |

## What This Does NOT Include

- **ML overlay (Layer 2):** Deferred until ML audit FAILs resolved. See `ml_methodology_audit.md`.
- **Prop firm execution:** Signal-only first. Demo/live execution after signal validation.
- **Multi-instrument:** MNQ only. MGC/MES need size filters (separate deployment).
- **Automation:** Apex prohibits automation. Tradeify/TopStep automation is Phase 2.

## Risk Acknowledgement

- Raw baseline has NO strategy selection — trades every signal on 11 sessions
- 2025 replay max DD was -15.34R — real max DD will be higher (no hindsight)
- All sessions trade regardless of regime — no vol filter, no ML gate
- The 3 pre-registered sessions are the only statistically credible ones. The other 8 provide diversification but lack formal significance at N=55 honest test count.

## Architecture

### Batch mode (RECOMMENDED)
```
forward_test.py
  -> refresh_data.py --instrument MNQ
     -> Databento API: download missing .dbn.zst files
     -> ingest_dbn.py --resume: bars into gold.db:bars_1m
     -> build_bars_5m.py: gold.db:bars_5m
     -> build_daily_features.py: gold.db:daily_features
  -> outcome_builder.py (gap dates only)
     -> gold.db:orb_outcomes (pre-computed trade results with E2 slippage)
  -> check_kill_criteria.py --from-outcomes
     -> queries orb_outcomes WHERE trading_day >= 2026-01-01
     -> per-session ExpR, portfolio ExpR, O'Brien-Fleming
     -> PASS / WAITING / KILL verdict
```

### Live mode (for manual trading with real-time signals)
```
run_live_session.py --signal-only --raw-baseline
  -> build_raw_baseline_portfolio() (portfolio.py)
  -> SessionOrchestrator (signal_only=True)
     -> DataFeed (WebSocket) -> BarAggregator -> ExecutionEngine -> signal output
     -> TradeJournal -> live_journal.db (persistent)
     -> PerformanceMonitor + CUSUM per strategy

check_kill_criteria.py (no --from-outcomes)
  -> reads live_journal.db
  -> per-session ExpR, slippage, portfolio ExpR, O'Brien-Fleming
```
