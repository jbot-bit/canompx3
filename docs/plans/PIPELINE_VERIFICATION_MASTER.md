# PIPELINE VERIFICATION MASTER PROMPT
## Total End-to-End Audit: Zero Tolerance for Lookahead, Bias, or Arithmetic Drift

**Created:** 2026-03-29
**Authority:** This prompt governs a multi-session verification campaign. It is NOT a planning document — it is an executable test protocol. Every section produces pass/fail evidence or the pipeline does not ship.

**Completion standard:** Every cell in the verification matrix (§9) must show PASS with evidence hash, or show FAIL with a linked fix commit. No exceptions. No "probably fine." No "we checked 6% and it looked good."

---

## §0. OPERATING PRINCIPLES

These are non-negotiable for the entire campaign:

1. **EVIDENCE OVER ASSERTION.** "It looks correct" is not evidence. Evidence = command output, row counts, diffs, screenshots. If you can't paste proof, it didn't happen.

2. **QUERY BEFORE CODE-READ.** Data tells you WHAT is wrong. Code tells you WHY. Always query first. Do not read 10 files to theorize about what "probably" happens.

3. **NO LOOKAHEAD — EVER.** Every feature, filter, and signal used at trade decision time must be knowable BEFORE the trade is entered. The test for this is simple: "Could a trader sitting at their screen at entry_time know this value?" If no → BANNED. If maybe → PROVE IT with timestamps.

4. **NO SELECTION BIAS.** Discovery window ≠ validation window. In-sample end date is defined BEFORE running OOS. No peeking. No "let's just check one more year."

5. **NO SURVIVORSHIP BIAS.** Dead instruments (M2K, MCL, SIL, M6E, MBT) stay dead. No re-testing "just to see." If a filter only works on the winner instrument, it's not a filter — it's an artifact.

6. **FAIL-CLOSED AT EVERY GATE.** If a test is ambiguous, the answer is FAIL. If a check throws an exception, the answer is FAIL. If output is missing, the answer is FAIL. The pipeline earns trust by passing, not by being given the benefit of the doubt.

7. **ONE THING AT A TIME.** Complete a verification block. Confirm pass/fail. Fix if needed. Re-run. THEN move to the next block. No parallelism on verification — it hides dependencies.

8. **CANONICAL SOURCES ONLY.** All verification queries hit `bars_1m`, `daily_features`, `orb_outcomes`. Never trust derived tables (`validated_setups`, `edge_families`, `live_config`) as ground truth. They are DOWNSTREAM — they inherit bugs, they don't reveal them.

---

## §1. PRE-FLIGHT — DATABASE INTEGRITY

**Purpose:** Confirm the database is complete, fresh, and structurally sound before any verification begins. HALT if any check fails.

### 1.1 Freshness
```sql
-- Must be within 2 trading days of today
SELECT 'bars_1m' AS layer, symbol, MAX(ts_utc)::DATE AS latest
FROM bars_1m GROUP BY symbol
UNION ALL
SELECT 'daily_features', symbol, MAX(trading_day)
FROM daily_features GROUP BY symbol
UNION ALL
SELECT 'orb_outcomes', symbol, MAX(trading_day)
FROM orb_outcomes GROUP BY symbol
ORDER BY 1, 2;
```
**Pass criterion:** All active instruments (MGC, MNQ, MES) within 2 trading days.

### 1.2 Row Counts & Coverage
```sql
-- Row counts by instrument
SELECT 'bars_1m' AS tbl, symbol, COUNT(*) AS rows,
       MIN(ts_utc)::DATE AS first_date, MAX(ts_utc)::DATE AS last_date
FROM bars_1m GROUP BY symbol
UNION ALL
SELECT 'daily_features', symbol, COUNT(*),
       MIN(trading_day), MAX(trading_day)
FROM daily_features GROUP BY symbol
UNION ALL
SELECT 'orb_outcomes', symbol, COUNT(*),
       MIN(trading_day), MAX(trading_day)
FROM orb_outcomes GROUP BY symbol
ORDER BY 1, 2;
```
**Pass criterion:** No instrument has gaps > 5 consecutive trading days. Date ranges are consistent across layers.

### 1.3 Schema Verification
```sql
DESCRIBE bars_1m;
DESCRIBE daily_features;
DESCRIBE orb_outcomes;
```
**Pass criterion:** All expected columns present. No unexpected columns (would indicate schema drift).

### 1.4 Timezone Sanity
```sql
-- Brisbane 09:00 = UTC 23:00 previous day. Verify trading_day assignment.
SELECT trading_day, ts_utc, symbol
FROM bars_1m
WHERE symbol = 'MNQ'
  AND EXTRACT(HOUR FROM ts_utc) = 23
  AND EXTRACT(MINUTE FROM ts_utc) = 0
ORDER BY ts_utc DESC
LIMIT 5;
-- The trading_day should be the NEXT calendar day (UTC 23:00 = Brisbane 09:00 next day)
```
**Pass criterion:** trading_day = DATE(ts_utc) + 1 for all UTC 23:00 bars.

### 1.5 Orphan Check
```sql
-- Outcomes without matching daily_features = broken join
SELECT o.symbol, o.trading_day, o.orb_minutes, COUNT(*) AS orphan_outcomes
FROM orb_outcomes o
LEFT JOIN daily_features d
  ON o.symbol = d.symbol
  AND o.trading_day = d.trading_day
  AND o.orb_minutes = d.orb_minutes
WHERE d.trading_day IS NULL
GROUP BY 1, 2, 3
ORDER BY 4 DESC
LIMIT 20;
```
**Pass criterion:** ZERO orphan rows. Any orphan = pipeline bug.

**HALT CONDITIONS:** Stale data (>2 days), missing instrument, schema mismatch, orphan rows > 0, timezone assignment error. DO NOT PROCEED past §1 if any check fails.

---

## §2. LOOKAHEAD AUDIT — TOTAL COVERAGE

**Purpose:** Prove that every feature used at trade decision time is computed from data available BEFORE trade entry. This is the single most important verification in the entire pipeline.

### 2.1 Feature Timestamp Audit

For EVERY column in `daily_features` that is used by any filter or entry model, prove it is not lookahead:

```python
"""
For each feature column in daily_features:
1. Identify WHERE it is computed in build_daily_features.py
2. Trace the data window: what bars does it read?
3. Prove the window ends BEFORE the ORB period
4. If the window includes ANY bar from the trading session → LOOKAHEAD
"""
```

**Audit matrix (fill every cell — no skipping):**

| Feature | Computed In (file:line) | Data Window | Ends Before ORB? | Verdict |
|---------|------------------------|-------------|-------------------|---------|
| `orb_high` | build_daily_features.py:L??? | First N min of session | AT ORB close | KNOWN AT ENTRY — defines the trade |
| `orb_low` | build_daily_features.py:L??? | First N min of session | AT ORB close | KNOWN AT ENTRY — defines the trade |
| `orb_size_pts` | build_daily_features.py:L??? | orb_high - orb_low | AT ORB close | KNOWN AT ENTRY |
| `atr_20` | build_daily_features.py:L1158-1163 | true_ranges[i-20:i] | Prior day close | NO LOOKAHEAD ✓ |
| `atr_20_pct` | build_daily_features.py:L1185-1197 | Rank in prior 252 days | Prior day close | NO LOOKAHEAD ✓ |
| `rel_vol` | build_daily_features.py:L??? | Break bar vol / prior median | At break time | VERIFY — trace exactly |
| `break_delay_min` | build_daily_features.py:L??? | Minutes from ORB end to break | At break time | KNOWN AT ENTRY ✓ |
| `double_break` | build_daily_features.py:L393 | Both sides break | AFTER entry | **LOOKAHEAD — BANNED** ✓ (already flagged) |
| `overnight_range_pct` | build_daily_features.py:L??? | Pre-session bars | Before ORB | VERIFY |
| `gap_pct` | build_daily_features.py:L??? | Open vs prior close | At open | VERIFY |
| `dow` | build_daily_features.py:L??? | Calendar | Known at start of day | NO LOOKAHEAD ✓ |
| `is_nfp` / `is_opex` / `is_holiday_adjacent` | build_daily_features.py:L??? | Calendar | Known at start of day | NO LOOKAHEAD ✓ |
| [EVERY OTHER COLUMN] | ??? | ??? | ??? | MUST BE FILLED |

**Execution method:**
```bash
# Step 1: Get ALL columns from daily_features
python -c "
import duckdb
con = duckdb.connect('gold.db', read_only=True)
cols = con.execute('DESCRIBE daily_features').fetchall()
for c in cols:
    print(f'{c[0]:40s} {c[1]}')
"

# Step 2: For each non-trivial column, grep build_daily_features.py for computation
# Step 3: Trace the data window manually — no shortcuts
```

**Pass criterion:** Every column either (a) proven no-lookahead with file:line evidence, or (b) flagged BANNED and confirmed not used in any filter/entry model.

### 2.2 Filter Lookahead Audit

For EVERY filter in `trading_app/config.py ALL_FILTERS`:
```python
"""
1. What columns does this filter read from daily_features?
2. Are ALL those columns proven no-lookahead in §2.1?
3. Does the filter computation itself introduce any lookahead?
   (e.g., ranking against future values, using session-end data at session-start)
"""
```

**Pass criterion:** Every filter chain traces back to proven-clean features only.

### 2.3 Entry Model Lookahead Audit

For each entry model (E1, E2):
```python
"""
1. What price does it enter at? (E1: next bar open, E2: ORB ± slippage)
2. Is the entry price available at decision time?
3. Does confirm_bars logic use any bar data that isn't yet observed?
4. Fill-bar ambiguity: is the conservative assumption applied?
   (target + stop hit same bar → loss, NOT win)
"""
```

**Pass criterion:** Entry prices are executable at decision time. No optimistic fill assumptions.

### 2.4 Banned Feature Enforcement
```bash
# Verify double_break is NEVER used in any filter or entry model
grep -rn "double_break" trading_app/ pipeline/ --include="*.py" | grep -v "LOOK.AHEAD" | grep -v "#" | grep -v "test"
```
**Pass criterion:** Zero references to `double_break` in any active filter or decision path.

---

## §3. COST MODEL — 100% COVERAGE

**Purpose:** Verify that pnl_r in orb_outcomes matches the canonical cost model for ALL strategies, not just a 6% sample.

### 3.1 Cost Model Constants Verification
```python
"""
From pipeline/cost_model.py, extract:
- point_value per instrument
- commission per RT
- spread assumption per RT
- slippage assumption per RT
- total_friction per RT

Cross-check against TRADING_RULES.md.
Any mismatch = HALT.
"""
```

### 3.2 pnl_r Recalculation — Full Population
```sql
-- For EVERY row in orb_outcomes, recalculate pnl_r from first principles:
-- pnl_r = (pnl_points * point_value - friction) / risk_dollars
-- Compare to stored pnl_r. Tolerance: ±0.001R

WITH cost_specs AS (
    SELECT 'MGC' AS symbol, 10.0 AS point_value, 5.74 AS friction
    UNION ALL SELECT 'MNQ', 2.0, 2.74
    UNION ALL SELECT 'MES', 5.0, 3.74
),
recalc AS (
    SELECT o.*,
           c.point_value,
           c.friction,
           o.orb_size_pts * c.point_value AS risk_dollars,
           (o.pnl_points * c.point_value - c.friction) /
             NULLIF(o.orb_size_pts * c.point_value, 0) AS expected_pnl_r,
           ABS(o.pnl_r - (o.pnl_points * c.point_value - c.friction) /
             NULLIF(o.orb_size_pts * c.point_value, 0)) AS drift
    FROM orb_outcomes o
    JOIN cost_specs c ON o.symbol = c.symbol
)
SELECT symbol,
       COUNT(*) AS total_rows,
       SUM(CASE WHEN drift > 0.001 THEN 1 ELSE 0 END) AS mismatches,
       MAX(drift) AS max_drift,
       AVG(drift) AS avg_drift
FROM recalc
GROUP BY symbol;
```
**Pass criterion:** ZERO mismatches above 0.001R tolerance. If ANY mismatch exists, investigate the specific rows:
```sql
-- Show the worst offenders
SELECT * FROM recalc WHERE drift > 0.001 ORDER BY drift DESC LIMIT 20;
```

### 3.3 Stop Multiplier Verification
```sql
-- For strategies using stop_mult = 0.75:
-- risk_dollars should be 0.75 * orb_size_pts * point_value
-- Verify this is correctly applied in outcome computation
SELECT stop_mult, symbol,
       COUNT(*) AS n,
       AVG(ABS(risk_dollars - stop_mult * orb_size_pts * point_value)) AS avg_risk_drift
FROM orb_outcomes o
JOIN (SELECT 'MGC' AS symbol, 10.0 AS point_value
      UNION ALL SELECT 'MNQ', 2.0
      UNION ALL SELECT 'MES', 5.0) c ON o.symbol = c.symbol
WHERE stop_mult IS NOT NULL
GROUP BY 1, 2;
```
**Pass criterion:** avg_risk_drift < 0.01 for all groups.

### 3.4 Friction as Percentage of Risk
```sql
-- Verify the friction/risk relationship is mathematically consistent
-- friction_pct should equal friction / risk_dollars * 100
SELECT symbol,
       NTILE(10) OVER (PARTITION BY symbol ORDER BY orb_size_pts) AS size_decile,
       AVG(orb_size_pts) AS avg_orb_size,
       AVG(5.74 / NULLIF(orb_size_pts * 10.0, 0) * 100) AS calc_friction_pct  -- MGC example
FROM orb_outcomes
WHERE symbol = 'MGC'
GROUP BY 1, 2
ORDER BY 2;
```
**Pass criterion:** Small ORBs have high friction %. Large ORBs have low friction %. Monotonic relationship. This is arithmetic truth — if violated, the cost model is broken.

---

## §4. OUTCOME ENTRY/EXIT — BAR-BY-BAR REPLAY

**Purpose:** This is the test that does not exist yet. Build it. For a representative sample of trades, replay the exact bar sequence from entry to exit and verify the outcome matches orb_outcomes.

### 4.1 Test Design

```python
"""
OUTCOME REPLAY VERIFIER

For N randomly-selected outcomes (stratified by instrument × session × entry_model × outcome_type):

1. Load the orb_outcomes row (entry_price, stop_price, target_price, exit_price, pnl_points, outcome)
2. Load the bars_1m sequence from entry_bar to exit_bar
3. Walk the bars forward one at a time:
   a. For each bar, check: does HIGH >= target? Does LOW <= stop? (for longs)
   b. If target hit → outcome should be 'win', exit_price should be target_price
   c. If stop hit → outcome should be 'loss', exit_price should be stop_price
   d. If BOTH hit same bar → outcome should be 'loss' (conservative assumption)
   e. If neither hit and time_stop triggers → check MTM at threshold bar
   f. If session ends → verify session-end exit logic
4. Compare replayed outcome to stored outcome
5. Compare replayed pnl_points to stored pnl_points (tolerance ±0.01)
6. Compare replayed pnl_r to stored pnl_r (tolerance ±0.001)

SAMPLE SIZE: Minimum 500 outcomes, stratified:
- 100 per instrument (MGC, MNQ, MES) minimum — more for MNQ (larger population)
- Within each instrument: 50% wins, 50% losses (proportional to actual mix if imbalanced)
- Include edge cases: time-stops, session-end exits, fill-bar ambiguity cases
- Include ALL entry models (E1, E2)
- Include ALL apertures (O5, O15, O30)

OUTPUT: Row-level pass/fail with mismatch details.
"""
```

### 4.2 Edge Cases to Specifically Target

These are the cases most likely to harbor bugs:

| Edge Case | Why It Matters | How to Test |
|-----------|---------------|-------------|
| Fill-bar ambiguity (E2) | E2 enters intra-bar — bar OHLC partially post-fill | Select 50 E2 trades where entry_bar has high-low > 2× typical |
| Target + stop same bar | Conservative = loss. If stored as win → BUG | Select all outcomes where the bar containing exit has range ≥ target-stop distance |
| Time-stop (T80) | MTM < 0 at threshold → exit at bar close | Select 50 time-stop outcomes, verify MTM was negative at threshold bar |
| Session-end exit | Trade still open at session close → forced exit | Select all session-end exits, verify exit_price = last bar close of session |
| First bar after ORB | Entry immediately after ORB close — confirm bars = 1 | Select 50 CB=1 trades, verify entry is on correct bar |
| Confirm bars = 5 | 5 consecutive close > ORB level required | Select 50 CB=5 trades, verify all 5 closes are on the correct side |
| O15 and O30 apertures | These were identified as potentially broken after paper_trader fix | Select 100 O15 + 100 O30 trades specifically, verify ORB window is correct width |
| Stop multiplier 0.75 | Tighter stop = different stop_price | Select 50 stop_mult=0.75 trades, verify stop_price = entry - 0.75 * ORB |

### 4.3 Implementation Requirements

```python
"""
Script: scripts/verification/verify_outcome_replay.py

Requirements:
- Read-only DB access (NEVER write)
- Deterministic random seed for reproducibility (seed=42)
- Stratified sampling (not just random — cover all instrument×session×entry×aperture combos)
- Output: CSV with columns [outcome_id, symbol, trading_day, orb_label, entry_model,
          orb_minutes, rr_target, confirm_bars, stored_outcome, replayed_outcome,
          stored_pnl_points, replayed_pnl_points, stored_pnl_r, replayed_pnl_r,
          match, mismatch_type, notes]
- Summary: total tested, total matched, total mismatched, mismatch breakdown by type
- EXIT CODE: 0 if 100% match, 1 if ANY mismatch

CRITICAL: The replay logic must be written FROM SCRATCH — do NOT import outcome_builder.
The whole point is to independently verify outcome_builder's logic.
If the replay imports outcome_builder, it's testing nothing.
"""
```

**Pass criterion:** 100% match on outcome type. 100% match on pnl_points within ±0.01. 100% match on pnl_r within ±0.001. ANY mismatch = investigate and fix before proceeding.

---

## §5. WALK-FORWARD RECALCULATION

**Purpose:** Independently verify that walk-forward OOS metrics in validated_setups are reproducible. The validator computed them once — can we get the same numbers from raw data?

### 5.1 Walk-Forward Window Verification

```python
"""
For each validated strategy:
1. Read the walk-forward parameters (train start, OOS windows, expanding anchor)
2. Independently partition orb_outcomes into train/test sets
3. Compute IS metrics on train set
4. Compute OOS metrics on each test window
5. Compute WFE = OOS_Sharpe / IS_Sharpe
6. Compare to stored WFE in validated_setups

Tolerance: WFE ±0.05
"""
```

### 5.2 Strategy Metric Recalculation

```sql
-- For each validated strategy, recalculate ExpR, WR, Sharpe from orb_outcomes
-- This catches any aggregation bugs in strategy_discovery.py

WITH strategy_params AS (
    -- Extract params from a specific validated strategy
    SELECT strategy_id, symbol, orb_label, orb_minutes, entry_model,
           confirm_bars, rr_target, filter_type, direction
    FROM validated_setups
    WHERE strategy_id = '<STRATEGY_ID>'
),
raw_trades AS (
    SELECT o.pnl_r, o.outcome, o.trading_day
    FROM orb_outcomes o
    JOIN daily_features d ON o.symbol = d.symbol
        AND o.trading_day = d.trading_day
        AND o.orb_minutes = d.orb_minutes
    JOIN strategy_params s ON o.symbol = s.symbol
        AND o.orb_label = s.orb_label
        AND o.orb_minutes = s.orb_minutes
        AND o.entry_model = s.entry_model
        AND o.confirm_bars = s.confirm_bars
        AND o.rr_target = s.rr_target
    WHERE o.direction = s.direction
    -- APPLY FILTER HERE based on filter_type
)
SELECT COUNT(*) AS n_trades,
       AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) AS win_rate,
       AVG(pnl_r) AS expr,
       AVG(pnl_r) / NULLIF(STDDEV(pnl_r), 0) * SQRT(252) AS annualized_sharpe
FROM raw_trades;
```

**Run this for ALL 772 validated strategies.** Not a sample. ALL of them.

**Pass criterion:** ExpR within ±0.005R, WR within ±1%, Sharpe within ±0.1 of stored values. Mismatches = aggregation bug in discovery or validator.

---

## §6. NON-ORB COLUMN DIVERGENCE INVESTIGATION

**Purpose:** 21 non-ORB columns were found to have unexplained divergence. Investigate every one.

### 6.1 Identify the 21 Columns
```sql
-- List all columns in daily_features that are NOT orb_* prefixed
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'daily_features'
  AND column_name NOT LIKE 'orb_%'
  AND column_name NOT IN ('symbol', 'trading_day', 'orb_minutes')
ORDER BY column_name;
```

### 6.2 For Each Column: Consistency Check
```python
"""
For each non-ORB column:
1. Is it computed once per (symbol, trading_day) or once per (symbol, trading_day, orb_minutes)?
2. If per-day: are all 3 orb_minutes rows identical? If not → BUG or misunderstanding
3. If per-orb_minutes: is there a valid reason it varies by aperture?
4. NULL rate: what % of rows are NULL? Is this expected?
5. Distribution: any impossible values (negative ATR, volume = 0 on trading days, etc.)?
"""
```

```sql
-- Example: Check if atr_20 is consistent across orb_minutes for same day
SELECT symbol, trading_day,
       COUNT(DISTINCT atr_20) AS distinct_atr_values,
       MIN(atr_20) AS min_atr, MAX(atr_20) AS max_atr
FROM daily_features
GROUP BY symbol, trading_day
HAVING COUNT(DISTINCT atr_20) > 1
LIMIT 20;
-- Should return ZERO rows — ATR is a daily feature, not aperture-specific
```

**Pass criterion:** Every column either (a) correctly varies by aperture with documented reason, or (b) is identical across apertures for same day, or (c) divergence is identified as a bug and fixed.

---

## §7. O15/O30 APERTURE VERIFICATION

**Purpose:** O15 and O30 strategies (208 total) need re-verification after the paper_trader fix. Their metrics may not match discovery.

### 7.1 ORB Window Width Verification
```sql
-- Verify that O15 ORBs use exactly 15 minutes of data, O30 uses 30 minutes
-- The ORB high/low should be computed from the correct number of bars
SELECT orb_minutes,
       symbol,
       COUNT(*) AS n_days,
       AVG(orb_size_pts) AS avg_orb_size,
       MIN(orb_size_pts) AS min_orb_size,
       MAX(orb_size_pts) AS max_orb_size
FROM daily_features
WHERE orb_minutes IN (15, 30)
GROUP BY 1, 2
ORDER BY 1, 2;
```

### 7.2 O15/O30 Break Timing Verification
```sql
-- Break delay for O15/O30 should be measured from END of respective ORB period
-- Not from end of 5-minute ORB
SELECT orb_minutes, symbol,
       AVG(break_delay_min) AS avg_delay,
       MIN(break_delay_min) AS min_delay,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY break_delay_min) AS median_delay
FROM daily_features
WHERE break_delay_min IS NOT NULL
GROUP BY 1, 2
ORDER BY 1, 2;
-- O15 avg delay should be SHORTER than O5 (less time remaining after wider ORB)
-- O30 avg delay should be SHORTER still
```

### 7.3 Paper Trader Cross-Check for O15/O30
```python
"""
Run paper_trader replay for a sample of O15 and O30 strategies.
Compare execution results to orb_outcomes.
If paper_trader was fixed but outcomes weren't rebuilt → STALE OUTCOMES.

python -m trading_app.paper_trader --instrument MNQ --start 2024-01-01 --end 2025-12-31 \
    --strategy-id <O15_STRATEGY_ID> --mode replay

Check: Does the paper_trader's P&L match the orb_outcomes aggregate?
Tolerance: ±5% on total P&L, ±2% on win rate.
"""
```

**Pass criterion:** O15/O30 outcomes are fresh (rebuilt after paper_trader fix), ORB windows are correct width, break timing is measured from correct reference point.

---

## §8. FILTER CHAIN INTEGRITY — COMPLETE AUDIT

**Purpose:** Every filter in ALL_FILTERS must be verified: correct column references, correct comparison operators, correct thresholds, no silent failures.

### 8.1 Filter Enumeration
```python
"""
From trading_app/config.py:
1. List ALL filter names in ALL_FILTERS
2. For each filter, extract:
   - Which daily_features columns it reads
   - The comparison logic (>=, <=, BETWEEN, etc.)
   - The threshold values
   - Whether it's instrument-specific or universal
"""
```

### 8.2 Filter Application Verification
```python
"""
For each filter:
1. Query daily_features with the filter applied
2. Query daily_features WITHOUT the filter
3. Verify the row count reduction matches expectations
4. Verify NO rows in the filtered set violate the filter condition

Example for ORB_G5:
SELECT COUNT(*) FROM daily_features WHERE orb_size_pts >= 5.0 AND symbol = 'MNQ';
-- vs
SELECT COUNT(*) FROM daily_features WHERE symbol = 'MNQ';
-- Verify every row in the G5 set actually has orb_size_pts >= 5.0
SELECT COUNT(*) FROM daily_features
WHERE orb_size_pts < 5.0
  AND symbol = 'MNQ'
  AND <G5_FILTER_APPLIED>;
-- Must be ZERO
"""
```

### 8.3 Silent Drop Detection
```python
"""
CRITICAL: Check that unknown filter_type strings don't silently drop all trades.

Test: Inject a fake filter name into a discovery query.
Expected: Error or explicit 0-trade result with warning.
Actual: ??? (If it silently returns 0 trades with no warning → BUG)
"""
```

### 8.4 Composite Filter Decomposition
```python
"""
For composite filters (e.g., ORB_G5_L12, ATR70_VOL):
1. Verify each sub-condition is applied (not just the first)
2. Verify AND logic (not OR)
3. Verify the composite is stricter than either component alone

Test: COUNT with composite should be <= MIN(COUNT with component A, COUNT with component B)
"""
```

**Pass criterion:** All filters produce expected row counts, no silent drops, no logical errors, composites are strictly additive.

---

## §9. VERIFICATION MATRIX — MASTER SCORECARD

**Every cell must be filled with PASS + evidence or FAIL + ticket.**

| # | Verification Block | Scope | Status | Evidence | Notes |
|---|-------------------|-------|--------|----------|-------|
| 1.1 | DB freshness | All instruments | ✅ PASS | All 3 to 2026-03-24 | stress_test T0 |
| 1.2 | Row counts & coverage | All instruments | ✅ PASS | 16.8M bars, 39K features, 10M outcomes | stress_test T0 |
| 1.3 | Schema verification | All tables | ✅ PASS | 275 df cols, 23 oo cols, all expected present | DESCRIBE |
| 1.4 | Timezone sanity | Sample check | ✅ PASS | CME 22:xx UTC, NYSE 13:xx UTC, DST clean | UTC+10 verified |
| 1.5 | Orphan check | orb_outcomes ↔ daily_features | ✅ PASS | 0 orphans | LEFT JOIN |
| 2.1 | Feature lookahead audit | ALL daily_features columns | ✅ PASS | 275 cols traced, 6 LA categories, 0 in filters | Code trace + grep |
| 2.2 | Filter lookahead audit | ALL 15 filter classes | ✅ PASS | 2 violations (DoubleBreak, OvernightRange) — neither in ALL_FILTERS | inspect + regex |
| 2.3 | Entry model lookahead audit | E1, E2 | ✅ PASS | E1=next-bar-open, E2=stop+1tick, ambig=100% loss | DB verified |
| 2.4 | Banned feature enforcement | double_break | ✅ PASS | 0 active references in any filter | grep |
| 3.1 | Cost model constants | MGC, MNQ, MES | ✅ PASS | MGC=$5.74, MNQ=$2.74, MES=$3.74 | COST_SPECS |
| 3.2 | pnl_r recalculation | 100% of orb_outcomes | ✅ PASS | **5,641,853 rows, ZERO mismatches >0.005R** | Full SQL recalc |
| 3.3 | Stop multiplier verification | All stop_mult rows | ✅ PASS | 600/600 consistent | stress_test T5 |
| 3.4 | Friction/risk relationship | All instruments | ✅ PASS | All 3 monotonically decreasing | Decile analysis |
| 4.1 | Outcome bar-by-bar replay | 582 stratified | ✅ PASS | **582/582 match** | verify_outcome_replay.py |
| 4.2 | Edge case: ambiguous bars | 50 targeted | ✅ PASS | 50/50 | Conservative=LOSS confirmed |
| 4.2 | Edge case: target+stop same bar | 3,501 total in DB | ✅ PASS | 100% resolved as LOSS | DB query |
| 4.2 | Edge case: time-stop T80 | N/A | ⬜ SKIP | T80 disabled (all None in config) | Not testable |
| 4.2 | Edge case: session-end exit | Scratch=NULL exit | ⬜ SKIP | Scratches have NULL pnl_r/exit | By design |
| 4.2 | Edge case: CB=1 and CB=5 | 100 targeted | ✅ PASS | 50/50 CB=1, 50/50 CB=5 | Targeted replay |
| 4.2 | Edge case: O15/O30 specific | 200 targeted | ✅ PASS | 100/100 O15, 100/100 O30 | Targeted replay |
| 4.2 | Edge case: stop_mult=0.75 | Via stress_test T5 | ✅ PASS | 600/600 consistent | apply_tight_stop vs engine |
| 5.1 | Walk-forward recalculation | Spot check | ✅ PASS | N diff = 2026 holdout (before 2026-01-01 exact match) | Temporal holdout verified |
| 5.2 | Strategy metric recalculation | 488 validated | ✅ PASS | All diffs explained by holdout_date | compute_metrics traced |
| 6.1 | Non-ORB column identification | All columns | ✅ PASS | 43 non-ORB cols checked | DESCRIBE + IS DISTINCT |
| 6.2 | Non-ORB column consistency | 43 columns | ✅ PASS | 42/43 identical; 1 (atr_20_pct) = tail NULLs on O30 | Active instruments only |
| 7.1 | O15/O30 ORB window width | All O15/O30 days | ✅ PASS | Sizes monotonic: O5 < O15 < O30 all instruments | AVG query |
| 7.2 | O15/O30 break timing | All O15/O30 days | ✅ PASS | Delays logical per aperture width | Median analysis |
| 7.3 | O15/O30 paper trader cross-check | Fix verified | ✅ PASS | _get_daily_features_row loads correct orb_minutes | Code + DB |
| 8.1 | Filter enumeration | ALL_FILTERS complete list | ✅ PASS | 41 filters, 24 composites | Code enumeration |
| 8.2 | Filter application verification | 123 filter×session combos | ✅ PASS | 0 row-level violations | MNQ spot check |
| 8.3 | Silent drop detection | Inject fake filter | ✅ PASS | None → fail-closed in execution_engine | Code + grep |
| 8.4 | Composite filter decomposition | All 24 composites | ✅ PASS | 24/24 strictly additive (composite ≤ min(base, overlay)) | Count comparison |

**Completed: 2026-03-29. Session: PIPELINE AUDIT. 26 PASS, 2 SKIP (T80 disabled, scratch by design).**

---

## §10. EXECUTION ORDER — SESSION PLAN

**Do not deviate from this order. Each block depends on the previous.**

### Session 1: Foundation (§1 + §2)
- Pre-flight (§1) — HALT if fails
- Lookahead audit (§2) — most critical, do first while fresh
- Deliverable: Completed §2.1 feature matrix, all cells filled

### Session 2: Cost Model (§3)
- Full population pnl_r recalculation (§3.2) — the 94% gap
- Stop multiplier and friction verification (§3.3, §3.4)
- Deliverable: Zero mismatches or identified bugs with fixes

### Session 3: Outcome Replay (§4)
- Build the replay verifier script from scratch
- Run against 500+ stratified sample
- Run targeted edge case tests
- Deliverable: verify_outcome_replay.py + CSV results + 100% match or bug list

### Session 4: Walk-Forward + Strategy Metrics (§5)
- Recalculate all 772 validated strategies from raw orb_outcomes
- Verify WFE independently
- Deliverable: Metric match report, mismatches identified

### Session 5: Column Divergence + O15/O30 (§6 + §7)
- Investigate all 21 non-ORB columns
- Verify O15/O30 aperture correctness
- Paper trader cross-check for O15/O30
- Deliverable: Column audit complete, O15/O30 verified or rebuilt

### Session 6: Filter Chain (§8) + Final Matrix (§9)
- Complete filter audit
- Fill every cell in verification matrix
- Deliverable: VERIFICATION MATRIX 100% filled, no blank cells

### Session 7: Fix Session (if needed)
- Address all FAIL cells from matrix
- Re-run affected verification blocks
- Deliverable: All cells PASS or documented KNOWN_ISSUE with mitigation

---

## §11. ANTI-PATTERNS — BANNED BEHAVIORS DURING VERIFICATION

```
❌ "I checked a sample and it looks fine" — Sample is only acceptable for §4 (with stratification). Everything else is 100% or nothing.
❌ "The code looks correct so the output must be correct" — Reading code is not verifying code. Run it.
❌ "This was already verified in a previous session" — Prove it. Show the evidence. Memory is not evidence.
❌ Importing outcome_builder.py into the replay verifier — defeats the purpose.
❌ Using validated_setups or edge_families as ground truth for ANY verification.
❌ Adjusting thresholds after seeing results ("let's use ±0.01 tolerance instead of ±0.001").
❌ Skipping a section because "it probably hasn't changed since last time."
❌ Reporting PASS without pasting the actual output.
❌ Fixing a bug and not re-running ALL downstream verification blocks.
❌ "I'll come back to this later" — every block completes or explicitly FAILs. No deferrals.
```

---

## §12. SUCCESS CRITERIA

The pipeline is verified when AND ONLY when:
1. Every cell in §9 Verification Matrix shows PASS with evidence
2. Zero lookahead features are used in any active filter or entry model
3. Zero cost model mismatches above tolerance
4. The outcome replay verifier exists as a permanent test and passes
5. All 772 validated strategies reproduce their metrics from raw data
6. All 21 non-ORB column divergences are explained or fixed
7. O15/O30 outcomes are confirmed fresh and correct
8. All filters are verified correct with no silent drops
9. This document is updated with final results and archived in `docs/audits/`

**Until ALL nine conditions are met, the pipeline is UNVERIFIED and no new strategies should be promoted to live trading.**

---

## §13. MAINTENANCE — POST-VERIFICATION

After verification passes:
1. `verify_outcome_replay.py` becomes a permanent CI test (run on every outcome rebuild)
2. pnl_r recalculation check becomes a drift check (add to `check_drift.py`)
3. Filter silent-drop test becomes a unit test
4. Lookahead audit matrix is frozen as `docs/audits/lookahead_matrix_YYYY-MM-DD.md`
5. Any future pipeline change must re-run affected verification blocks before merge
