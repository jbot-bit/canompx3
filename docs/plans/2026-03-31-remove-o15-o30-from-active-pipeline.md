# Remove O15/O30 from Active Pipeline

**Date:** 2026-03-31
**Status:** DESIGNED — awaiting implementation

## Context

O15/O30 killed per Mar 29 audit. All 788 validated strategies are O5 only.
Pipeline still builds, tracks staleness, rebuilds, and asserts for O15/O30 — wasting
time and creating false "DECAYING" alerts.

## Design

Add `ACTIVE_ORB_MINUTES = [5]` alongside `VALID_ORB_MINUTES = [5, 15, 30]` in
`pipeline/build_daily_features.py`. `VALID` = what exists in the DB schema (immutable).
`ACTIVE` = what we actively build/track (can shrink).

### Files to Change

| File | Change |
|------|--------|
| `pipeline/build_daily_features.py` | Add `ACTIVE_ORB_MINUTES = [5]` below line 87 |
| `scripts/tools/pipeline_status.py` | Import `ACTIVE_ORB_MINUTES`, use for staleness + remove O15/O30 rebuild steps |
| `scripts/tools/refresh_data.py` | Use `ACTIVE_ORB_MINUTES` for daily_features builds |
| `scripts/tools/assert_rebuild.py` | Use `ACTIVE_ORB_MINUTES` for outcome coverage assertion |

### NOT Touched (correct as-is)

| File | Reason |
|------|--------|
| `pipeline/check_drift.py` | Row integrity expects 3 rows/date — DB truth |
| `scripts/tools/sensitivity_analysis.py` | Research tool, all apertures valid |
| `pipeline/run_pipeline.py` | CLI choices stay broad for historical use |

### Failure Modes

1. Drift check row integrity → uses `VALID_ORB_MINUTES` (unchanged). Safe.
2. Historical data → `VALID_ORB_MINUTES` unchanged, CLI unchanged. Safe.
3. Future aperture → add to both constants. Single file edit.

### Verification

1. `python scripts/tools/pipeline_status.py --status` — no O15/O30 staleness
2. `python pipeline/check_drift.py` — passes
3. `python scripts/tools/assert_rebuild.py` — no O15/O30 assertion failures
