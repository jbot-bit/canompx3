---
mode: IMPLEMENTATION
task: Wave 5 presession filter class registration
classification: CORE_MODIFICATION
created: 2026-04-11T03:30:00+10:00
updated: 2026-04-11T03:30:00+10:00
---

# Stage: Wave 5 — Presession Filter Class Registration

## Purpose
Phase B T2-T8 battery identified 3 tradable new survivors using presession features that aren't yet wired into ALL_FILTERS:
- `atr_vel_ratio` (HIGH quintile) — MNQ TOKYO_OPEN RR1.0 (in_ExpR +0.188, p=0.0042)
- `atr_vel_ratio` (HIGH quintile) — MES US_DATA_1000 RR1.5 (in_ExpR +0.103, p=0.044)
- `garch_forecast_vol` (LOW quintile) — MNQ NYSE_OPEN RR1.5 (in_ExpR +0.240, p=0.042)

Register new filter classes in `trading_app/config.py` with proper fail-closed semantics.

## Scope Lock
- trading_app/config.py
- tests/test_trading_app/test_config.py

## Blast Radius
Config: adds 2 new filter classes (ATRVelocityFilter, GARCHForecastVolFilter) and 4 new ALL_FILTERS instances (ATR_VEL_P80, ATR_VEL_P60, GARCH_VOL_P20, GARCH_VOL_P40). Adds routing in get_filters_for_grid() for the 3 specific (instrument, session) pairs that survived T2-T8. Legacy grid behavior unchanged for non-routed sessions. Validator: no change. Discovery: new filters accessible via hypothesis injection (already shipped). Drift check #12 (filter_type sync) must still pass.

## Approach
1. **Create ATRVelocityFilter class** — reads `atr_vel_ratio` from daily_features, min_ratio threshold, fail-closed on null
2. **Create GARCHForecastVolFilter class** — reads `garch_forecast_vol` from daily_features, with direction ("high"/"low") parameter for inclusion above/below threshold, fail-closed on null
3. **Register 4 instances in ALL_FILTERS:**
   - `ATR_VEL_P80`: min_ratio=1.15 (top 20% roughly)
   - `ATR_VEL_P60`: min_ratio=1.05 (top 40% — expanding regime per existing regime bucketing)
   - `GARCH_VOL_P20`: max_pct=20, direction="low"
   - `GARCH_VOL_P40`: max_pct=40, direction="low"
4. **Route in get_filters_for_grid():**
   - ATR_VEL_P60/P80 for MNQ TOKYO_OPEN + MES US_DATA_1000
   - GARCH_VOL_P20/P40 for MNQ NYSE_OPEN only
5. **No look-ahead:** both `atr_vel_ratio` and `garch_forecast_vol` are computed from prior days only (verified in pipeline/build_daily_features.py lines 1121-1132 and 1207)

Note: pre-computed daily_features features — these don't need per-instance threshold rolling percentile computation at discovery time.

## Acceptance Criteria
1. Discovery with a hypothesis file using ATR_VEL_P80 accepts combos for MNQ TOKYO_OPEN
2. Discovery with GARCH_VOL_P20 accepts combos for MNQ NYSE_OPEN
3. Legacy mode discovery (no hypothesis file) shows no change in combo count for sessions NOT in the routing rules
4. Drift check #12 (ALL_FILTERS ↔ registered filter_type sync) passes
5. Unit test: ATRVelocityFilter.matches_row returns True when atr_vel_ratio >= threshold, False when None (fail-closed)
6. Unit test: GARCHForecastVolFilter direction="low" inverts comparison correctly
7. No existing tests regress

## Kill Criteria
- If new filters appear in non-routed sessions' grid → routing bug, revert
- If fail-closed semantics skip days where data IS present → bug, fix before commit
- If drift check #12 fails → filter registry inconsistency, fix before commit
