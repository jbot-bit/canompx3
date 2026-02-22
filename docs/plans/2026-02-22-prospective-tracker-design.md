# Prior-Day Outcome Signal — Prospective Tracker Design

**Date:** 2026-02-22
**Status:** Approved

## Signal Under Test

MGC 0900 E0 CB1 RR2.0 G4+ prev=LOSS — BH FDR survivor at q=0.10 (avgR=+0.585R, t=+3.34, p=0.0016, N=49). REGIME-class hypothesis with compelling mechanism ("failed break -> fresh break" momentum). Prospective tracking to N=100 before deployment as position-size overlay.

**Source:** `research/research_prev_day_signal.py`, findings in `research/output/prev_day_signal_findings.md`

## Scope

Research tracker only. No paper_trader or execution_engine integration. Standalone table + script. Prove the signal first, wire later.

## Schema

```sql
CREATE TABLE prospective_signals (
    signal_id        VARCHAR NOT NULL,     -- 'MGC_0900_PREV_LOSS'
    trading_day      DATE NOT NULL,
    symbol           VARCHAR NOT NULL,
    session          INT NOT NULL,         -- 900
    prev_day_outcome VARCHAR NOT NULL,     -- 'LOSS'
    orb_size         DOUBLE,              -- today's ORB size (points)
    entry_model      VARCHAR NOT NULL,     -- 'E0'
    confirm_bars     INT NOT NULL,         -- 1
    rr_target        DOUBLE NOT NULL,      -- 2.0
    outcome          VARCHAR,              -- WIN/LOSS/SCRATCH
    pnl_r            DOUBLE,
    is_prospective   BOOLEAN NOT NULL,     -- FALSE=backfill, TRUE=after freeze
    freeze_date      DATE NOT NULL,        -- 2026-02-22
    created_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (signal_id, trading_day)
);
```

## Script

**File:** `scripts/tools/prospective_tracker.py`

**Behavior:**
1. Creates table if not exists
2. Backfills all historical qualifying days, marks `is_prospective=FALSE`
3. New days after freeze date marked `is_prospective=TRUE`
4. Idempotent: DELETE+INSERT for signal_id on each run
5. Prints console report with retrospective/prospective/combined stats

**Qualifying day query** (mirrors research script pattern):
```sql
WITH lag_feats AS (
    SELECT trading_day, symbol,
           orb_0900_outcome AS curr_outcome,
           LAG(orb_0900_outcome) OVER (
               PARTITION BY symbol ORDER BY trading_day
           ) AS prev_outcome,
           orb_0900_size AS orb_size_pts
    FROM daily_features
    WHERE orb_minutes = 5
      AND symbol = 'MGC'
)
SELECT l.trading_day, l.prev_outcome, l.orb_size_pts,
       o.outcome, o.pnl_r
FROM lag_feats l
JOIN orb_outcomes o
  ON l.trading_day = o.trading_day
  AND l.symbol = o.symbol
  AND o.orb_label = '0900'
  AND o.orb_minutes = 5
  AND o.entry_model = 'E0'
  AND o.rr_target = 2.0
  AND o.confirm_bars = 1
  AND o.outcome IS NOT NULL
WHERE l.prev_outcome = 'LOSS'
  AND l.orb_size_pts >= 4.0
ORDER BY l.trading_day
```

**CLI:**
```bash
python scripts/tools/prospective_tracker.py
python scripts/tools/prospective_tracker.py --freeze-date 2026-02-22
```

Freeze date defaults to `2026-02-22` (hardcoded constant). CLI override available.

## Console Report

```
=== Prospective Signal Tracker: MGC_0900_PREV_LOSS ===
Signal: MGC 0900 E0 CB1 RR2.0 G4+ | prev_day = LOSS
Freeze date: 2026-02-22

--- RETROSPECTIVE (before freeze) ---
  N=49  avgR=+0.585  WR=63.3%  t=+3.34  p=0.0016

--- PROSPECTIVE (after freeze) ---
  N=7   avgR=+0.412  WR=57.1%  t=+1.42  p=0.205

--- COMBINED ---
  N=56  avgR=+0.563  WR=62.5%  t=+3.51  p=0.0009

--- PROGRESS ---
  Prospective N:   7 / 100  [=======...............................] 7%
  Next milestone:  N=100 -> formal re-evaluation
  Final milestone: N=150 -> full validation pipeline

--- YEAR-BY-YEAR (prospective only) ---
  2026:  N=7  avgR=+0.412  WR=57.1%
```

Threshold alerts at N>=100 and N>=150.

## Milestones

| Prospective N | Action |
|---------------|--------|
| 100 | Formal re-evaluation: re-run BH FDR on prospective subset |
| 150 | Full validation pipeline, consider 1.5x position-size overlay |

## Design Decisions

1. **Backfill all history, tag as retrospective** — provides context without contaminating prospective evidence
2. **No paper_trader integration** — clean separation, prove first
3. **Manual script execution** — no health_check hook, no automation
4. **Idempotent DELETE+INSERT** — matches project convention
5. **LAG() on daily_features, not orb_outcomes** — matches research script pattern, `orb_0900_outcome` lives in daily_features
6. **Freeze date as hardcoded constant** — simple, deterministic, no persistence issues
