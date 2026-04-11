---
mode: IMPLEMENTATION
task: Wave 5 G5 — GARCH rolling-percentile filter for MNQ NYSE_OPEN LOW
classification: CORE_MODIFICATION
created: 2026-04-11T03:30:00+10:00
updated: 2026-04-12T00:00:00+10:00
---

# Stage: Wave 5 G5 — GARCH Rolling-Percentile Deployment

## Purpose

Deploy the strongest un-shipped Phase B survivor from the Wave 4 presession research: **MNQ × NYSE_OPEN × RR1.5 × garch_forecast_vol LOW quintile** (in_ExpR +0.240, WFE 1.00, p=0.042, per `scripts/research/wave4_presession_t2t8.py`). This is the single remaining tradable signal from the entire Wave 4 research arc not already deployed or blocked on deployment gates.

The research methodology used `pd.qcut(IS_2020-2023_sample, 5)` — top/bottom 20% quintile within the IS sample. Deploying as a fixed absolute threshold would fragment across instruments (MNQ Q20 gfv ≈ 0.159, MES Q20 gfv ≈ 0.111) and drift with regime changes. The correct deployment is a **rolling-percentile column** in `daily_features`, following the existing `atr_20_pct` / `overnight_range_pct` pattern.

## Supersedes

This stage file **supersedes** the original "Wave 5 presession filter class registration" plan (previous version of this file, created 03:30 2026-04-11). The original plan called for percentile-named filters (`ATR_VEL_P80`, `GARCH_VOL_P20`) but pivoted to absolute thresholds (`ATR_VEL_GE105/110/115` in commit `a877b387`). Audit on 2026-04-12 confirmed:

- **ATR_VEL** is effectively deployed — `ATR_VEL_GE105 ≈ IS_Q80 (1.0537) to within 0.7 bps`. Hypothesis file `2026-04-11-atr-vel-expansion.yaml` exists. Blockers are Criterion 8 (N_oos=13, time-driven) and Criterion 11 (TopStep Day-1 cap), not research methodology. No further research-side work needed.
- **GARCH** is the only remaining gap — no filter class, no hypothesis file, instrument-varying Q20 distribution makes a single absolute threshold statistically wrong.

## Scope Lock

- `pipeline/init_db.py`
- `pipeline/build_daily_features.py`
- `trading_app/config.py`
- `tests/test_pipeline/test_build_daily_features.py`
- `tests/test_trading_app/test_config.py`
- `tests/test_app_sync.py`  — pre-existing ATR_VEL_GE* omission + new GARCH_VOL_PCT_LT20 key
- `docs/audit/hypotheses/2026-04-12-wave5-garch-nyse-open.yaml`

## Blast Radius

- **daily_features schema** — additive column `garch_forecast_vol_pct DOUBLE`. Additive migration via existing `ALTER TABLE ADD COLUMN IF NOT EXISTS` machinery in init_db.py. No row count change. No effect on `orb_outcomes`, `validated_setups`, `live_config`, strategy lanes.
- **pipeline/build_daily_features.py** — adds rolling-percentile computation in post-pass, pattern identical to `atr_20_pct` at lines 1140-1152. Prior-only window (252 lookback, min 60). No look-ahead.
- **trading_app/config.py** — adds `GARCHForecastVolPctFilter` class (direction-parameterized low/high) + single instance `GARCH_VOL_PCT_LT20`. Drift check #12 must pass (filter_type sync).
- **Backfill compute cost** — populating the new column for existing daily_features rows runs build_daily_features on all 3 instruments × 3 apertures. ~14K rows total, ~26 min wall time per Phase 3c rebuild note. Touches only daily_features. No downstream tables.
- **Parallel session collision check (2026-04-12):** parallel sessions are touching `ROADMAP.md`, `trading_app/lane_allocator.py`, `trading_app/live_config.py`, `trading_app/prop_portfolio.py`, `trading_app/sprt_monitor.py`, `trading_app/sr_monitor.py`, `trading_app/strategy_fitness.py`. **NONE of these overlap with the scope_lock above.** Safe to proceed.
- **Drift check #9 pre-existing failure** — `pipeline/dashboard.py:30` one-way dependency violation from parallel shelf-hardening work. Not introduced by this stage; will remain failing until the parallel work lands the dashboard fix.

## Approach

1. **Stage 1 — Column + computation** (`init_db.py` + `build_daily_features.py` + test)
   - Add `garch_forecast_vol_pct DOUBLE` to the CREATE TABLE DDL (near existing garch columns, ~line 329) and to the additive migration list (~line 493)
   - Add rolling-percentile computation in the post-pass loop after the existing `overnight_range_pct` block (~line 1168). Pattern:
     ```python
     gfv_today = rows[i].get("garch_forecast_vol")
     if gfv_today is not None:
         gfv_lookback = 252
         prior_gfvs = [
             rows[j]["garch_forecast_vol"]
             for j in range(max(0, i - gfv_lookback), i)
             if rows[j].get("garch_forecast_vol") is not None
         ]
         if len(prior_gfvs) >= 60:
             sorted_prior = sorted(prior_gfvs)
             rank = bisect_left(sorted_prior, gfv_today)
             rows[i]["garch_forecast_vol_pct"] = round(rank / len(sorted_prior) * 100, 2)
     ```
   - Initialize `row["garch_forecast_vol_pct"] = None` in the row init block (~line 857)
   - Write `test_garch_vol_pct_no_lookahead` — constructs a synthetic rows list with known garch_forecast_vol values, runs the computation, asserts day i's pct uses only rows 0..i-1

2. **Stage 2 — Filter class + instance** (`trading_app/config.py` + unit tests)
   - Add `GARCHForecastVolPctFilter(StrategyFilter)` following `OwnATRPercentileFilter` at line 1143 as template
   - Direction param: `"low"` (admit rows where `garch_forecast_vol_pct <= max_pct`) or `"high"` (admit rows where `>= min_pct`)
   - Fail-closed on null (`atom_numeric` helper for None/NaN/pd.NA)
   - Register `GARCH_VOL_PCT_LT20 = GARCHForecastVolPctFilter(filter_type="GARCH_VOL_PCT_LT20", direction="low", pct_threshold=20, description="GARCH forecast vol rolling percentile <= 20 (low-vol regime)")`
   - Do NOT route in `get_filters_for_grid()` — access via hypothesis-file injection only (keeps legacy grid byte-identical, preserves criterion #3 parity)
   - Unit tests: fail-closed on null, direction semantics (low vs high), pct_threshold boundary inclusive/exclusive

3. **Stage 3 — Pre-registered hypothesis file** (`docs/audit/hypotheses/2026-04-12-wave5-garch-nyse-open.yaml`)
   - Single hypothesis: MNQ × NYSE_OPEN × E2 × RR1.5 × CB1 × S1.0 × GARCH_VOL_PCT_LT20
   - Theory citation: wave4_presession_t2t8.py + docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md (MinBTL)
   - Economic basis: low forecast vol = quieter macro release regime. NYSE_OPEN RR1.5 captures momentum continuation without overshoot into noise. WFE 1.00 (IS == OOS spread) in the research indicates perfect sign stability across 2020-2023 IS and 2024-2025 OOS.
   - MinBTL: N=1 well under the 300 clean cap; `total_expected_trials=1`
   - Holdout date: 2026-01-01 (Mode A sacred)
   - Kill criteria: raw p ≥ 0.05, WFE < 0.50, 2026 OOS ExpR < 0, era stability fail (N≥50 era with ExpR < -0.05), annualized Sharpe < 0.70
   - `git add` + `git commit` to satisfy Phase 4 cleanliness gate before discovery

4. **Stage 4 — Rebuild + discovery dry-run**
   - Run `python -m trading_app.outcome_builder --instrument MNQ --force --orb-minutes 5` is NOT needed (outcomes don't depend on the new column)
   - Run `python pipeline/build_daily_features.py --instrument MNQ --force` (and MES, MGC for consistency) to backfill `garch_forecast_vol_pct`
   - Verify row counts unchanged before/after rebuild
   - `python -m trading_app.strategy_discovery --instrument MNQ --orb-minutes 5 --dry-run --hypothesis-file docs/audit/hypotheses/2026-04-12-wave5-garch-nyse-open.yaml` — assert injection log fires with `Phase 4: injected 1 hypothesis filter type(s)`, assert combos > 0
   - Final drift check — expected state: 99 pass, 1 pre-existing dashboard.py failure, 7 advisory

## Acceptance Criteria

1. `garch_forecast_vol_pct` column exists in `daily_features` schema (both DDL and migration list)
2. `build_daily_features.py` computes the column with no-lookahead test passing
3. After MNQ rebuild, `garch_forecast_vol_pct IS NOT NULL` for >= 60% of rows where `garch_forecast_vol IS NOT NULL` (warm-up adjusted)
4. `GARCHForecastVolPctFilter` class registered in `ALL_FILTERS` with direction="low" instance
5. Filter fail-closed on null (`matches_row(row_with_null_gfv_pct, ...) == False`)
6. Drift check #12 (filter_type sync) passes
7. Hypothesis file loads via `load_hypothesis_metadata` without error, SHA computed, Phase 4 gates (git clean + Mode A + MinBTL) pass
8. Dry-run discovery with hypothesis file injects `GARCH_VOL_PCT_LT20` and produces > 0 combos for MNQ NYSE_OPEN

## Kill Criteria

- If no-lookahead test shows day i referencing rows[i] or later → BUG, revert
- If backfill changes daily_features row count → investigate before proceeding
- If drift check introduces a new failure beyond the pre-existing dashboard.py:30 → fix before commit
- If hypothesis file fails MinBTL gate (should trivially pass at N=1) → investigate loader
- If discovery dry-run produces 0 combos → either injection broken or scope mismatch → investigate

## Out of scope (explicit)

- Criterion 11 TopStep Day-1 scaling fix (Phase F deployment work, separate stage)
- ATR_VEL methodology rework — audit showed GE105 is faithful to research within 0.7 bps
- Other Wave 4 Phase B signals — all 50 non-survivors were killed in research, re-testing them is a data snooping violation
- Binary features (`overnight_took_pdh`, `took_pdh_before_1000`) — tested in Phase B, not in the 3 survivors list
- Wave 5 stage file housekeeping (renaming, reorganizing) — keep it minimal
