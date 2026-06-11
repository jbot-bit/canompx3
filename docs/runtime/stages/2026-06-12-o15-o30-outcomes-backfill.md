# Stage: Backfill stale O15/O30 orb_outcomes → unblock Tokyo live-gate re-check

task: Backfill stale O15/O30 orb_outcomes for MNQ/MES/MGC, then re-check whether the ROBUST Tokyo head (MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_VOL_4K_O15_S075) clears the live SR gate once its recent signal is no longer stale.
mode: IMPLEMENTATION

## Scope Lock
- gold.db (orb_outcomes table — O15/O30 rows only; O5 untouched)
- (no source-code edits — canonical recompute via trading_app/outcome_builder.py)

## Blast Radius
- Writes orb_outcomes (canonical layer) via idempotent DELETE WHERE symbol=? AND orb_minutes=? then INSERT OR REPLACE. O5 rows untouched. Re-runnable.
- Reads daily_features + bars_1m (both fresh to 2026-06-10, read-only).
- Downstream INVALIDATED (regen owed AFTER, separate GO): validated_setups + edge_families fitness for O15/O30 strats; C11/C12 state envelopes (regen, don't debug — per memory baton daily_refresh_db_rebuild_invalidates_c11_c12).
- Holdout-safe: mechanical outcome generation from price data; sacred-window split applied downstream at validation, unchanged. No live/broker code touched.
- Capital path: NONE. Research-layer truth only.

## ROOT CAUSE (proven 2026-06-12)
- daily_features FRESH to 2026-06-10 (all 3 live instruments, O5/O15/O30).
- orb_outcomes STALE at O15/O30:
  | inst | O5 | O15 | O30 |
  | MNQ | 06-10 | 05-24 | 05-24 |
  | MES | 06-09 | 04-23 | 04-23 |
  | MGC | 06-09 | 04-23 | 04-23 |
- Pure outcomes-layer backfill gap. Source data complete; outcomes just not computed for 15/30-min apertures.
- IMPACT: every O15/O30 strategy scored fitness + SR-monitor against stale outcomes. The Tokyo SR-ALARM that paused the ROBUST head MAY be a stale-data artifact (its O15 "last-10 = -0.128R" ends 2026-05-22, ~3 weeks old).

## KEY EXECUTION FACT (verified)
- outcome_builder HAS checkpoint/resume (outcome_builder.py:843-861 "Skip days already computed").
- WITHOUT --force + WITH --start at the gap → computes ONLY missing tail days (~30 days), FAST.
- --dry-run WITHOUT --start scans all 2075 days (~10 min) → DO NOT use for the real backfill.
- So real backfill is: `--start <gap_start> --end <today>` NO --force NO --dry-run → fast incremental.

## EXECUTION PLAN (run from a FRESH session — context was 75% when planned)

### Step 1 — Backfill (incremental, fast). Run each; ~1-3 min each (tail only):
```
python -m trading_app.outcome_builder --instrument MNQ --orb-minutes 15 --start 2026-05-20 --end 2026-06-11
python -m trading_app.outcome_builder --instrument MNQ --orb-minutes 30 --start 2026-05-20 --end 2026-06-11
python -m trading_app.outcome_builder --instrument MES --orb-minutes 15 --start 2026-04-20 --end 2026-06-11
python -m trading_app.outcome_builder --instrument MES --orb-minutes 30 --start 2026-04-20 --end 2026-06-11
python -m trading_app.outcome_builder --instrument MGC --orb-minutes 15 --start 2026-04-20 --end 2026-06-11
python -m trading_app.outcome_builder --instrument MGC --orb-minutes 30 --start 2026-04-20 --end 2026-06-11
```
(start dates padded ~4 days before each gap so checkpoint overlap is safe; idempotent.)

### Step 2 — Verify freshness:
```
SELECT symbol, orb_minutes, MAX(trading_day) FROM orb_outcomes
WHERE orb_minutes IN (15,30) GROUP BY 1,2 ORDER BY 1,2;
```
Expect all == 2026-06-10 (or latest trading day with bars).

### Step 3 — Re-check Tokyo SR signal on NOW-FRESH O15 (the decision point):
Recompute last-10 / last-20 mean_R for MNQ_TOKYO_OPEN RR2.0 O15 + ORB_VOL_4K (orb_TOKYO_OPEN_volume>=4000), joined daily_features on (trading_day,symbol,orb_minutes).
- If last-10 turned POSITIVE with fresh data → SR-ALARM was a stale artifact → candidate to clear live gate. Proceed to downstream regen (Step 4, NEW GO).
- If still negative → cold regime is REAL → Tokyo stays paused, stand down (honest result).
- DO NOT peek at 2026 holdout for a deploy decision (HOLDOUT_SACRED_FROM=2026-01-01). SR-recheck uses recent realized R as a regime monitor, NOT as OOS confirmation of a new edge.

### Step 4 — Downstream regen (SEPARATE GO — heavy; invalidates C11/C12):
Per validation-workflow.md, only if Step 3 warrants it:
discovery (O15/O30) → validator → build_edge_families → health_check.
Then re-run lane allocation to see if the ROBUST head routes live.

## DONE CRITERIA
- O15/O30 max_day fresh across MNQ/MES/MGC (Step 2 proof).
- Tokyo SR signal recomputed on fresh data + verdict stated (Step 3).
- check_drift.py passes.
- Delete this stage file when Step 3 verdict delivered (Step 4 tracked separately if GO'd).

## STATUS: PLANNED — execution pending fresh session (planned at 75% ctx, 2026-06-12)
```
```
