# MGC Backfill Integration Plan (2026-04-09) — HARDENED

## Purpose

Extend MGC clean micro data from 2.7yr (2023-09-11) to ~3.9yr (2022-06-13) using
real MGC micro bars. Extends Bailey MinBTL budget from N=4 to N=7 at strict E=1.0.

## Data verification (completed)

File: `data/raw/databento/ohlcv-1m/MGC/mgc_real_micro_backfill_2022-06-13_to_2023-09-10.ohlcv-1m.dbn.zst`
- 9.1 MB, 796,275 total rows (635,059 outrights, 161,216 spreads filtered out)
- **Real MGC contract symbols** (MGCQ2, MGCZ2, MGCM3 etc.) — NOT GC parent proxy
- Price range: $1,614–$2,200 (gold was ~$1,700–$2,000 in this period — PLAUSIBLE)
- 386 unique trading days, zero gaps > 5 calendar days
- Launch day (2022-06-13): 10 contract months, 66K volume — liquid from day one
- Boundary: backfill ends 2023-09-08 (Friday), existing starts 2023-09-11 (Monday) — weekend gap, no missing data
- Cost: $0.00 (free on Standard subscription)

## Critical design decisions

### 1. minimum_start_date must change to 2022-06-13

`pipeline/asset_configs.py` currently has `date(2023, 9, 11)` — this was our first download
date, NOT the contract launch date. MGC launched 2022-06-13. Without changing this, the
ingester silently rejects all backfill bars (forces start to minimum_start_date).

Consumers of this field:
- `ingest_dbn.py` — date filter enforcement (the gate we need to open)
- `ingest_dbn_daily.py` — same gate
- `data_era.py:micro_launch_day()` — returns this value, used by era_for_trading_day()
- `parallel_rebuild.py` — rebuild start date
- `run_pipeline.py` — logging only

Impact: `era_for_trading_day("MGC", date(2022, 9, 1))` will now return MICRO instead of
PARENT. This is CORRECT — the data IS real micro from 2022-06-13. The prior classification
was wrong because we didn't have the data.

### 2. Full rebuild required, not partial

Adding 15 months of prior bars changes the **warmup state** for ALL subsequent days:
- ATR_20 at 2023-09-11 was computed cold (no prior history). After backfill, it has
  15 months of warm history → DIFFERENT ATR value → DIFFERENT filter results.
- Overnight range at 2023-09-11 was NULL (no prior session). After backfill, it has
  data → overnight range filters now populated for the first few weeks.
- All downstream orb_outcomes for 2023-09-11+ will CHANGE because the daily_features changed.

Phase 3c used the same full-rebuild approach. Partial rebuild would create inconsistency.

### 3. Existing 9 MGC validated_setups unaffected but stale

validated_setups rows are frozen snapshots from promotion time. The rebuild changes the
underlying experimental_strategies and orb_outcomes, but doesn't auto-update validated_setups.
This is fine — the 9 are already grandfathered research-provisional (Amendment 2.4) and need
re-validation under Mode A regardless.

### 4. MGC FK situation is clean

0/9 MGC validated_setups have promoted_from set. The FK-safe DELETE we implemented for MNQ
is a no-op for MGC — no rows are protected. Full DELETE+INSERT will work.

## Stages

### Stage 1: Config fix + Ingest

**Files changed:** `pipeline/asset_configs.py`

Changes:
- Line 69 comment: update "2023-09-11 (MGC contract launch)" to "2022-06-13 (CME Micro Gold 10oz launch)"
- Line 79: `date(2023, 9, 11)` to `date(2022, 6, 13)`
- Line 79 comment: "MGC micro launch date (CME Micro Gold 10oz, first traded 2022-06-13)"

**Also:** `pipeline/data_era.py` line 128 docstring example: update `date(2023, 9, 11)` to `date(2022, 6, 13)`

**Run:** `python -m pipeline.ingest_dbn --instrument MGC --start 2022-06-13 --end 2023-09-10`

**Verify:**
- `SELECT MIN(ts_utc)::DATE FROM bars_1m WHERE symbol = 'MGC'` → 2022-06-13
- `SELECT COUNT(*) FROM bars_1m WHERE symbol = 'MGC' AND ts_utc < '2023-09-11'` → ~300-400K new bars
- Price sanity: `SELECT AVG(close) FROM bars_1m WHERE symbol = 'MGC' AND ts_utc < '2023-01-01'` → ~1750-1900

### Stage 2: Full rebuild of derived layers

**Order matters:** daily_features FIRST, then outcomes (outcomes depend on daily_features).

```
# Daily features (all 3 apertures)
python pipeline/build_daily_features.py --instrument MGC --start 2022-06-13 --end 2026-04-07 --orb-minutes 5
python pipeline/build_daily_features.py --instrument MGC --start 2022-06-13 --end 2026-04-07 --orb-minutes 15
python pipeline/build_daily_features.py --instrument MGC --start 2022-06-13 --end 2026-04-07 --orb-minutes 30

# Outcomes (all 3 apertures, --force for DELETE+INSERT)
python -m trading_app.outcome_builder --instrument MGC --force --orb-minutes 5
python -m trading_app.outcome_builder --instrument MGC --force --orb-minutes 15
python -m trading_app.outcome_builder --instrument MGC --force --orb-minutes 30
```

**Verify:**
- daily_features: `SELECT MIN(trading_day), COUNT(*) FROM daily_features WHERE symbol = 'MGC'`
  Expected: start ~2022-06-13, count ~3,500+ (was 2,229)
- orb_outcomes: `SELECT MIN(trading_day), COUNT(*) FROM orb_outcomes WHERE symbol = 'MGC'`
  Expected: start ~2022-06-13, count ~900K+ (was 617K)
- Integrity: 9 daily_features checks should pass

### Stage 3: Drift checks + validation gate

```
python -m pipeline.check_drift
python scripts/tools/audit_behavioral.py
```

Must pass before proceeding to Stage 4.

### Stage 4: Update hypothesis file + rediscovery

- Update MGC YAML header: 3.9yr horizon, N=7 budget
- Add 3 new hypothesis bundles (must be pre-registered, gold-specific)
- Run discovery: `python -m trading_app.strategy_discovery --instrument MGC --orb-minutes 5 --holdout-date 2026-01-01 --hypothesis-file docs/audit/hypotheses/2026-04-09-mgc-mode-a-rediscovery.yaml`
- Run validator on any survivors

## Blast radius

| Component | Impact |
|---|---|
| `pipeline/asset_configs.py` | One date change (line 79) |
| `pipeline/data_era.py` | One docstring example (line 128) |
| `gold.db bars_1m` | Additive INSERT (~350K new MGC bars) |
| `gold.db daily_features` | Full MGC rebuild (~1,200 new rows, existing rows re-computed) |
| `gold.db orb_outcomes` | Full MGC rebuild (~300K new rows, existing rows re-computed) |
| `gold.db experimental_strategies` | MGC rediscovery replaces existing rows |
| `gold.db validated_setups` | NOT touched (frozen snapshots, still research-provisional) |
| MNQ/MES data | ZERO impact |
| Deployed bot | ZERO impact (MGC CME_REOPEN lane reads from validated_setups, not outcomes) |

## Rollback

1. Revert asset_configs.py + data_era.py to prior values
2. `DELETE FROM bars_1m WHERE symbol = 'MGC' AND ts_utc < '2023-09-11'`
3. Re-run daily_features + outcome_builder for MGC with original date range
4. Drift checks pass → back to prior state

## Time estimate

- Stage 1: ~2 min (config edit + ingest)
- Stage 2: ~25 min (full rebuild, sequential DuckDB writes)
- Stage 3: ~2 min (drift checks)
- Stage 4: ~5 min (hypothesis update + discovery + validation)
- Total: ~35 min wall time
