---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Unicorn/Trend-Day MFE Research — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build `research/research_trend_day_mfe.py` to compute TRUE session MFE from `bars_1m` (uncapped by target/stop), quantify the gap vs stored `mfe_r`, and test predictors of outsized moves.

**Architecture:** Single research script following `research_partial_profit.py` pattern. Joins `orb_outcomes` with `bars_1m` to replay price from entry to session end. Computes true MFE/MAE in R-multiples using `cost_model.py`. Statistical tests use paired comparisons with BH FDR for multi-instrument/session grid.

**Tech Stack:** DuckDB (read-only), pandas, numpy, scipy.stats, `research.lib.stats.bh_fdr`

---

## Critical Discovery (T1 Orient)

**`mfe_r` in `orb_outcomes` is CAPPED** — line 382 of `outcome_builder.py`:
```python
max_fav = max(float(np.max(pe_favorable[:first_hit_idx + 1])), 0.0)
```
This only tracks excursion up to the bar where target OR stop was hit. A RR2.0 winner shows `mfe_r ≈ 2.0`, never the true session continuation.

**`trading_day_end`** = 09:00 Brisbane next day (23:00 UTC). This is the session boundary for bar replay.

**Available predictor columns in `daily_features`:**
- `atr_20` — 20-day ATR
- `gap_open_points` — overnight gap size
- `overnight_range`, `overnight_high`, `overnight_low`
- `prev_day_range`, `prev_day_direction`, `prev_day_close`
- `garch_forecast_vol`, `garch_atr_ratio`
- `atr_vel_ratio`, `atr_vel_regime`
- `orb_{SESSION}_size` — ORB range size
- `orb_{SESSION}_compression_z` — pre-break compression (where available)
- `day_type` — day classification

---

### Task 1: Script Skeleton + DB Query

**Files:**
- Create: `research/research_trend_day_mfe.py`

**Step 1: Write the script skeleton**

Create the script with:
- Imports: duckdb, pandas, numpy, scipy.stats, project imports
- Constants: `ACTIVE_INSTRUMENTS`, `SESSIONS`, `ORB_MINUTES = [5, 15, 30]`, `RR_TARGETS = [1.5, 2.0, 2.5, 3.0]`
- `--instrument` CLI arg (default: all 4 active)
- Main query: JOIN `orb_outcomes` ON `bars_1m` to get entry_ts, entry_price, stop_price, break_dir, pnl_r, mfe_r, trading_day_end info
- Session list from `SESSION_CATALOG` keys (import, don't hardcode)

**Step 2: Write the core query**

```sql
SELECT o.trading_day, o.symbol, o.orb_label, o.orb_minutes,
       o.entry_model, o.rr_target, o.confirm_bars,
       o.entry_ts, o.entry_price, o.stop_price, o.break_dir,
       o.pnl_r, o.mfe_r AS capped_mfe_r, o.outcome,
       o.filter_type
FROM orb_outcomes o
WHERE o.symbol = ?
  AND o.entry_model IN ('E1', 'E2')
  AND o.outcome IS NOT NULL
  AND o.entry_ts IS NOT NULL
ORDER BY o.trading_day, o.orb_label
```

**Step 3: Run and verify row counts**

Run: `python research/research_trend_day_mfe.py --instrument MGC --dry-run`
Expected: Print row count per session, confirm >0 rows

**Step 4: Commit**

```bash
git add research/research_trend_day_mfe.py
git commit -m "feat: trend-day MFE research script skeleton + query"
```

---

### Task 2: TRUE Session MFE Computation

**Files:**
- Modify: `research/research_trend_day_mfe.py`

**Step 1: Write `compute_true_session_mfe()` function**

```python
def compute_true_session_mfe(
    bars_1m: pd.DataFrame,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    break_dir: str,
    trading_day_end: datetime,
    cost_spec: CostSpec,
) -> dict:
    """Compute TRUE session MFE from bars_1m, uncapped by target/stop.

    Returns:
        true_mfe_r: max favorable excursion in R from entry to session end
        true_mae_r: max adverse excursion in R from entry to session end
        session_close_r: P&L at session close in R
        bars_after_entry: number of bars from entry to session end
        time_to_mfe_min: minutes from entry to MFE bar
    """
```

Logic:
- Filter bars_1m: `entry_ts < ts_utc <= trading_day_end`
- For long: `favorable = highs - entry_price`, `adverse = entry_price - lows`
- For short: `favorable = entry_price - lows`, `adverse = highs - entry_price`
- Convert to R using `pnl_points_to_r(cost_spec, entry_price, stop_price, points)`
- Track bar index where MFE occurs → `time_to_mfe_min`
- Session close R = last bar close vs entry, in R

**Step 2: Wire into main loop**

For each outcome row:
- Derive `trading_day_end` from `compute_trading_day_utc_range(trading_day)`
- Fetch bars_1m for that day (cache per trading_day to avoid re-query)
- Call `compute_true_session_mfe()`
- Store results alongside capped values

**Step 3: Test on 10 rows**

Run with `--limit 10 --instrument MGC` flag, print per-row comparison.
Expected: `true_mfe_r >= capped_mfe_r` for every row (mathematical invariant).

**Step 4: Commit**

```bash
git add research/research_trend_day_mfe.py
git commit -m "feat: TRUE session MFE computation from bars_1m"
```

---

### Task 3: Gap Quantification (Capped vs True MFE)

**Files:**
- Modify: `research/research_trend_day_mfe.py`

**Step 1: Write `analyze_mfe_gap()` function**

For each (instrument, session, orb_minutes, rr_target) combo:
- `gap_r = true_mfe_r - capped_mfe_r`
- Summary stats: mean, median, P90, P95, P99 of gap_r
- % of trades where `true_mfe_r > 3 * rr_target` ("unicorns")
- % of trades where `true_mfe_r > 5 * rr_target` ("mega-unicorns")
- Paired t-test: `capped_mfe_r` vs `true_mfe_r` (every row is paired)

**Step 2: Output table**

Print formatted table:
```
Instrument | Session      | O | RR  | N    | Mean Gap | Median | P90  | P95  | %>3xRR | %>5xRR | p-value
MGC        | CME_REOPEN   | 5 | 2.0 | 1234 | +1.23R   | +0.45R | 3.2R | 5.1R | 12.3%  | 3.1%   | <0.001
```

**Step 3: Run full analysis**

Run: `python research/research_trend_day_mfe.py --instrument MGC`
Expected: Gap is statistically significant (p < 0.001) for most combos. Unicorn % varies by session.

**Step 4: Commit**

```bash
git add research/research_trend_day_mfe.py
git commit -m "feat: MFE gap quantification — capped vs true session MFE"
```

---

### Task 4: Predictor Analysis — Which Features Predict Unicorns?

**Files:**
- Modify: `research/research_trend_day_mfe.py`

**Step 1: Join daily_features predictors**

Query daily_features for each outcome row (JOIN on trading_day + symbol + orb_minutes).
Extract predictors:
- `atr_20` — recent volatility
- `gap_open_points` — overnight gap
- `overnight_range` — overnight trading range
- `prev_day_range` — previous day's range
- `garch_atr_ratio` — GARCH vs realized vol ratio
- `atr_vel_ratio` — ATR velocity
- `orb_{SESSION}_size` — ORB size for this session
- `orb_{SESSION}_compression_z` — pre-break compression (where available)

**Step 2: Write `test_unicorn_predictors()` function**

For each predictor:
- Binary split: unicorn (true_mfe_r > 3×RR) vs non-unicorn
- Welch's t-test on predictor values between groups
- Point-biserial correlation: predictor vs unicorn flag
- Collect p-values for BH FDR correction across all (instrument × session × predictor) tests

**Step 3: Write overnight expansion signal test**

Specific hypothesis from user spec:
- `overnight_expansion = overnight_range / atr_20` (or prev_day_range / atr_20)
- Split into tertiles: low/mid/high
- Compare unicorn rates across tertiles (chi-square or Fisher exact)
- Test: does high overnight expansion predict higher true_mfe_r?

**Step 4: BH FDR correction**

Apply BH FDR at q=0.05 across all predictor tests.
Label findings per RESEARCH_RULES.md: "confirmed edge", "promising hypothesis", "NO-GO".

**Step 5: Run and output**

Run: `python research/research_trend_day_mfe.py --instrument MGC --predictors`
Expected: Table of predictors ranked by effect size and FDR-adjusted p-value.

**Step 6: Commit**

```bash
git add research/research_trend_day_mfe.py
git commit -m "feat: unicorn predictor analysis with BH FDR"
```

---

### Task 5: Output Report + CSV Export

**Files:**
- Modify: `research/research_trend_day_mfe.py`

**Step 1: Structured output**

Add `--output-dir` flag (default: `research/output/`).
Save:
- `trend_day_mfe_gap_{instrument}.csv` — per-trade raw data (trading_day, session, true_mfe_r, capped_mfe_r, gap_r, predictors)
- `trend_day_mfe_summary_{instrument}.csv` — per-combo summary stats
- `trend_day_mfe_predictors_{instrument}.csv` — predictor test results with FDR status

**Step 2: Console summary**

Print:
1. Top-10 unicorn-producing combos (by unicorn %)
2. Statistically significant predictors (BH FDR survivors)
3. Overnight expansion signal result
4. Key finding labels per RESEARCH_RULES.md

**Step 3: Run full pipeline for all instruments**

```bash
python research/research_trend_day_mfe.py
```
(Loops all 4 active instruments by default)

**Step 4: Commit**

```bash
git add research/research_trend_day_mfe.py
git commit -m "feat: trend-day MFE report output + CSV export"
```

---

### Task 6: Validation + Memory Update

**Files:**
- Modify: `research/research_trend_day_mfe.py` (if fixes needed)

**Step 1: Validate invariant**

Confirm `true_mfe_r >= capped_mfe_r` for 100% of rows. Any violation = bug.

**Step 2: Run drift check**

```bash
python pipeline/check_drift.py
```
Expected: All checks pass. No new violations from research script.

**Step 3: Update memory**

Create or update memory file with key findings:
- Which sessions produce the most unicorns?
- Is overnight expansion a valid predictor?
- What % of trades have true MFE > 3×RR?
- Any BH FDR survivors in predictor tests?

**Step 4: Commit final**

```bash
git add research/research_trend_day_mfe.py research/output/
git commit -m "feat: trend-day MFE Phase 1 complete — unicorn discovery"
```

---

## Lookahead Guards

1. TRUE session MFE is NOT a filter — it's a post-hoc measurement for research only
2. Predictor tests use only pre-trade-entry features (atr_20, overnight_range, etc.)
3. `double_break` is LOOK-AHEAD — excluded from predictors
4. BH FDR applied across ALL tests (not cherry-picked)
5. Results labeled per RESEARCH_RULES.md thresholds

## Risk Assessment

- **Data volume:** ~6.1M outcome rows × bars_1m replay = significant compute. Mitigate: cache bars per trading_day, process one instrument at a time.
- **Memory:** bars_1m for a full day ≈ 1440 rows × 8 cols. Even 3000 trading days = trivial.
- **False positives:** BH FDR handles multiple testing. Grid is (4 instruments × 11 sessions × 3 apertures × 4 RR × 8 predictors) ≈ 4,224 tests. FDR correction is essential.
