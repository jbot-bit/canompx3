---
task: Fix refresh_data.py early-return bug that skips daily_features build when bars are current
mode: IMPLEMENTATION
agent: claude-code
slug: refresh-data-build-decoupling
updated: 2026-04-14
scope_lock:
  - scripts/tools/refresh_data.py
  - tests/test_tools/test_refresh_data.py
blast_radius: LOW (single script, single test file; idempotent builds via DELETE+INSERT)
acceptance:
  - refresh_data.py invokes run_build_steps even when gap_days <= 0 (for last N days)
  - tests/test_tools/test_refresh_data.py has a regression test covering this case
  - After backfill run, MES/MNQ/MGC 2026-04-13 have all 3 orb_minutes rows in daily_features
  - Drift check 58 passes (no trading days with row count != 3)
  - Resume F-1 stage afterwards
---

# Refresh Data Build Decoupling

## Root cause

`scripts/tools/refresh_data.py:247-249`:
```python
gap_days = (yesterday - last_date).days
if gap_days <= 0:
    print(f"  Already up to date (last bar: {last_date})")
    return True
```

Early return when bars already match yesterday. **Never runs `run_build_steps`**, so if bars were ingested out-of-band (e.g., live bot's BarPersister writes bars_1m at session end) but features were never built, features remain stale.

Observed symptom (2026-04-14):
- All 3 active instruments have `bars_1m` through 2026-04-13 09:59 (120 bars each)
- MES/MNQ have ZERO `daily_features` rows for 2026-04-13
- MGC has exactly 1 (orb_minutes=5) — orphan from a prior manual build

Today's 07:47 refresh ran, reported all 3 as "Already up to date" and short-circuited. Features never built.

## Approach

### Stage A: Regression test first (TDD red)
Add test covering: refresh_data invokes build steps when `gap_days <= 0` but last N days of features are stale.

### Stage B: Fix
Decouple build from download:
- If `gap_days > 0`: build from `fetch_start` to `yesterday` (existing behavior)
- If `gap_days <= 0`: still build for last ~3 days to catch out-of-band bar ingestions

`build_daily_features` is already idempotent (DELETE+INSERT pattern), so re-running on existing features is safe and cheap.

### Stage C: Backfill
Run `build_daily_features` for all 3 active instruments for 2026-04-13 across orb_minutes ∈ {5, 15, 30}. Plus `outcome_builder` for O5 2026-04-13.

### Stage D: Verify
- Drift check 58 passes
- Daily features integrity: MES=3, MNQ=3, MGC=3 rows for 2026-04-13

### Stage E: Resume F-1
Flip F-1 stage mode back to IMPLEMENTATION.

## Canonical sources reused
- `pipeline.build_daily_features.VALID_ORB_MINUTES` — ORB apertures
- `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` — instrument gate
