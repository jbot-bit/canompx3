---
task: Rebuild MGC daily_features for 2026-05-08 (partial build — only O5 row, missing O15+O30)
mode: IMPLEMENTATION
slug: fix-mgc-2026-05-08-partial-daily-features
created: 2026-05-11
updated: 2026-05-11
scope_lock:
  - gold.db (daily_features table — MGC symbol, trading_day 2026-05-08 rows only)
acceptance:
  - "python pipeline/check_drift.py emits Check 63 PASS"
  - "daily_features has 3 rows for MGC where trading_day='2026-05-08' (one per orb_minutes 5/15/30)"
  - "build_daily_features integrity verifier returns OK for the rebuilt range"
---

## Blast Radius

- gold.db `daily_features` table: rebuild rows for symbol=MGC, trading_day=2026-05-08 only.
- DELETE-then-INSERT idempotent pattern per `pipeline/build_daily_features.py` canonical builder.
- No code change. No schema change. No bars_1m mutation.
- Reads: bars_1m (MGC, 2026-05-07 → 2026-05-09 UTC range), prior daily_features (HTF/ATR context).
- Writes: daily_features (3 rows expected: O5, O15, O30 for MGC on 2026-05-08).
- Downstream effect: drift check 63 (row integrity) flips PASS. No other consumer changes.

## Why

Pre-commit drift check 63 blocks. Root cause: partial daily_features build — likely interrupted nightly pipeline left only the O5 row for MGC 2026-05-08 (O15 + O30 missing). Bars_1m has 600 bars for that day, ample for all three apertures.

Unblocks the recovery commits on branch `recover/orphaned-pr48-f3-2026-05-11`.

## Procedure

1. Confirm no concurrent pipeline writer (verified — only PowerShell shell running).
2. Run `pipeline.build_daily_features` for MGC 2026-05-08 with all three orb_minutes values.
3. Verify drift check 63 passes.
4. Verify integrity check in builder reports OK.
