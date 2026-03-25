# Paper Trade Logging — Design Plan

**Status:** DESIGNED — awaiting user GO
**Date:** 2026-03-25

## Goal
Log all 4 Apex MNQ lane paper trades to `gold.db` for forward OOS validation by June 2026.

## 4 Apex Lanes (from prop_profiles.py)

| Lane | Strategy ID | Session | Filter | RR |
|------|-------------|---------|--------|----|
| 1 | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20_O15` | NYSE_CLOSE | VOL_RV12_N20 | 1.0 |
| 2 | `MNQ_SINGAPORE_OPEN_E2_RR4.0_CB1_ORB_G8_O15` | SINGAPORE_OPEN | ORB_G8 | 4.0 |
| 3 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` | COMEX_SETTLE | ORB_G8 | 1.0 |
| 4 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15` | NYSE_OPEN | X_MES_ATR60 | 1.0 |

## Table Schema (`paper_trades` in gold.db)

```sql
CREATE TABLE paper_trades (
    trading_day      DATE NOT NULL,
    orb_label        TEXT NOT NULL,
    entry_time       TIMESTAMPTZ,
    direction        TEXT,           -- 'long'/'short'
    entry_price      DOUBLE,
    stop_price       DOUBLE,
    target_price     DOUBLE,
    exit_price       DOUBLE,
    exit_time        TIMESTAMPTZ,
    exit_reason      TEXT,           -- 'win'/'loss'/'scratch' (from orb_outcomes.outcome)
    pnl_r            DOUBLE,
    slippage_ticks   DOUBLE DEFAULT 0,  -- placeholder for live comparison
    strategy_id      TEXT NOT NULL,
    lane_name        TEXT,           -- human-readable: 'NYSE_CLOSE_VOL', 'SING_G8', etc.
    instrument       TEXT DEFAULT 'MNQ',
    orb_minutes      INTEGER,
    rr_target        REAL,
    filter_type      TEXT,
    entry_model      TEXT,
    PRIMARY KEY (strategy_id, trading_day)
);
```

## Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| Modify | `trading_app/db_manager.py` | Add paper_trades CREATE TABLE |
| Modify | `pipeline/init_db.py` | Register table in schema init |
| Create | `trading_app/paper_trade_logger.py` | Backfill + sync from orb_outcomes × daily_features |
| Create | `scripts/tools/paper_trade_summary.py` | Daily summary CLI |

## Backfill Approach

Join `orb_outcomes` (entry/exit/pnl_r pre-computed) with `daily_features` (filter eligibility) for the 4 strategy IDs since 2026-01-01. No need to re-run paper trader — outcomes already exist.

For each lane:
1. Query orb_outcomes WHERE symbol='MNQ' AND orb_label=X AND rr_target=Y AND entry_model='E2' AND confirm_bars=1
2. Join daily_features to check filter gate (e.g., ORB_G8 = orb_size >= 8 ticks)
3. Only days with a break AND passing filter = valid paper trade
4. INSERT into paper_trades

## Daily Summary Command

```bash
python scripts/tools/paper_trade_summary.py
```

Output:
- Trades today per lane
- Cumulative R per lane since 2026-01-01
- Running trade count per lane (target: 100+ for validation)
- Stale lane alert: any lane with no trade in 5+ calendar days

## "Live" Going Forward

After each pipeline rebuild (run_pipeline + outcome_builder), run:
```bash
python trading_app/paper_trade_logger.py --sync
```
Picks up new trading days. Idempotent (DELETE+INSERT pattern). Can add to rebuild chain later.

## Blast Radius

LOW — new table, new files. Reads orb_outcomes + daily_features (no writes to existing tables). Minor schema registration in db_manager.py and init_db.py.

## Discovery Notes

- No paper trade table exists today (trades are ephemeral JournalEntry objects)
- live_trades table is in separate live_journal.db (broker fills, not paper)
- orb_outcomes has ALL break-day outcomes pre-computed for all RR × confirm_bars combos
- Lane definitions pinned in prop_profiles.py lines 279-284
- Filter application requires daily_features join (filter_type gates)
