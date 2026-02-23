# T80 Time-Stop — Design

**Date:** 2026-02-23
**Status:** Approved
**Research:** `research/research_time_exit.py`, `research/output/time_exit_summary.md`

## Problem

The T80 time-stop is confirmed by BH FDR (116/742 groups, q=0.10) but only exists in
the execution engine (`EARLY_EXIT_MINUTES` in config.py). The outcome_builder — which
drives strategy discovery, validation, and portfolio assembly — has no time-stop logic.
Backtested metrics don't reflect the improvement.

## Design

### Additional columns on `orb_outcomes` (no extra rows)

Three new nullable columns:

| Column | Type | Description |
|--------|------|-------------|
| `ts_outcome` | VARCHAR | Outcome with time-stop applied (NULL if session has no time-stop) |
| `ts_pnl_r` | DOUBLE | P&L in R with time-stop (NULL if no time-stop) |
| `ts_exit_ts` | TIMESTAMPTZ | Exit timestamp under time-stop (NULL if no time-stop) |

### Computation logic

Added AFTER normal outcome computation (existing logic untouched):

1. Look up `EARLY_EXIT_MINUTES[orb_label]` — if None, leave `ts_*` columns NULL
2. Find the first post-entry bar at or after `entry_ts + timedelta(minutes=threshold)`
3. If normal exit happened BEFORE that bar: `ts_*` = baseline (time-stop never reached)
4. If normal exit happened AFTER (or trade is a scratch):
   - Compute MTM at time-stop bar close
   - If MTM < 0: `ts_outcome="time_stop"`, `ts_pnl_r` = MTM in R terms, `ts_exit_ts` = bar ts
   - If MTM >= 0: `ts_*` = baseline (trade keeps running, time-stop doesn't fire)

### Config source

`EARLY_EXIT_MINUTES` from `config.py` — single source of truth shared with execution engine.

Current values:
- 0900: 15 min (note: more aggressive than T80=38m — may want to revisit)
- 1000: 30 min (~T80=32m)
- 1800: 30 min (T80=36m)
- All others: None

### Downstream usage

- `pnl_r` = baseline (unchanged)
- `ts_pnl_r` = with time-stop
- `COALESCE(ts_pnl_r, pnl_r)` = time-stopped when available, baseline otherwise
- Strategy discovery/validation can opt into `ts_pnl_r`
- Portfolio assembly chooses which column per strategy

### Schema

```sql
ALTER TABLE orb_outcomes ADD COLUMN ts_outcome VARCHAR;
ALTER TABLE orb_outcomes ADD COLUMN ts_pnl_r DOUBLE;
ALTER TABLE orb_outcomes ADD COLUMN ts_exit_ts TIMESTAMPTZ;
```

Or: handled via `init_db.py` schema + `--force` rebuild.

## Files Modified

| File | Change |
|------|--------|
| `pipeline/init_db.py` | Add 3 columns to orb_outcomes schema |
| `trading_app/outcome_builder.py` | Time-stop annotation after normal outcome computation |
| `trading_app/config.py` | Export `EARLY_EXIT_MINUTES` (already exists, ensure importable) |
| `tests/test_trading_app/test_outcome_builder.py` | Tests for time-stop logic |

## Rebuild Required

After implementation: `python trading_app/outcome_builder.py --instrument MGC --force`
(repeat for MNQ, MES, M2K)

## Not In Scope

- Changing strategy discovery to USE ts_pnl_r (separate decision)
- Expanding EARLY_EXIT_MINUTES to more sessions (separate config decision)
- Changing the 0900: 15m value (separate research question)
