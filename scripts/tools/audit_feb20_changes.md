# Fresh Audit: Feb 20 2026 Changes

**Purpose:** Independent verification of three stacked changes before wiring C8/C3 exit rules into outcome_builder/paper_trader and updating TRADING_RULES.md. If anything here fails, STOP — don't build on a broken foundation.

**Database:** `C:/db/gold.db` (scratch copy with today's changes)
**Reference database:** `C:\Users\joshd\canompx3\gold.db` (production — may be stale, use for schema comparison only)

---

## LAYER 1: Validator Reconfig Audit

The validator was re-run with `--no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward`. This changed the strategy population from ~937 to 2,149. Verify the new population is clean.

### 1a. Population counts match expectations

```sql
-- Run against C:/db/gold.db
-- Expected: MGC=431, MNQ=1665, MES=53, total=2149
SELECT instrument, COUNT(*) AS n
FROM validated_setups
WHERE LOWER(status) = 'active'
GROUP BY instrument
ORDER BY instrument;

-- ABORT if total != 2149
SELECT COUNT(*) AS total_active
FROM validated_setups
WHERE LOWER(status) = 'active';
```

### 1b. No REGIME-grade strategies leaked through

With `--no-regime-waivers`, every validated strategy must have sample_size >= 100 (CORE tier). No 30-99 range strategies should exist.

```sql
-- Expected: 0 rows
SELECT strategy_id, sample_size, instrument
FROM validated_setups
WHERE LOWER(status) = 'active'
  AND sample_size < 100
ORDER BY sample_size;

-- ABORT if any rows returned
```

### 1c. min_years_positive_pct = 0.75 was enforced

Every strategy should have >= 75% of its active years with positive PnL. Check via the stored yearly breakdown.

```sql
-- Spot-check: pick 5 random strategies per instrument, verify yearly data
-- This requires the yearly columns — check what's available first:
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'validated_setups'
  AND column_name LIKE '%year%'
ORDER BY column_name;

-- Also check experimental_strategies for the raw yearly data:
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'experimental_strategies'
  AND column_name LIKE '%year%'
ORDER BY column_name;
```

### 1d. No walk-forward data (--no-walkforward confirmed)

strategy_trade_days should be sparse/empty for the new population.

```sql
-- Check coverage: what % of active strategies have strategy_trade_days rows?
SELECT
    COUNT(DISTINCT vs.strategy_id) AS active_strategies,
    COUNT(DISTINCT std.strategy_id) AS have_trade_days,
    ROUND(100.0 * COUNT(DISTINCT std.strategy_id) / COUNT(DISTINCT vs.strategy_id), 1) AS pct_covered
FROM validated_setups vs
LEFT JOIN (SELECT DISTINCT strategy_id FROM strategy_trade_days) std
    ON vs.strategy_id = std.strategy_id
WHERE LOWER(vs.status) = 'active';

-- Expected: pct_covered should be 15-25% (legacy walk-forward strategies only)
-- If pct_covered > 50%, something is wrong — walk-forward may have been run
```

### 1e. trade_day_hash populated in experimental_strategies for all PASSED

The edge family build depends on this. Verify 100% coverage.

```sql
-- Expected: 0
SELECT COUNT(*) AS missing_hash
FROM validated_setups vs
JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
WHERE LOWER(vs.status) = 'active'
  AND (es.trade_day_hash IS NULL OR LENGTH(es.trade_day_hash) != 32);
```

**ABORT if missing_hash > 0** — edge families used bad hashes.

### 1f. No duplicate strategy_ids

```sql
-- Expected: 0
SELECT strategy_id, COUNT(*) AS n
FROM validated_setups
WHERE LOWER(status) = 'active'
GROUP BY strategy_id
HAVING COUNT(*) > 1;
```

---

## LAYER 2: Edge Family Build Audit

554 families were built from 2,149 strategies. Verify the clustering is correct.

### 2a. Family counts match

```sql
-- Expected: MES=32, MGC=131, MNQ=391, total=554
SELECT instrument, COUNT(*) AS families
FROM edge_families
GROUP BY instrument
ORDER BY instrument;

SELECT COUNT(*) AS total_families FROM edge_families;
```

### 2b. No mega-families (hash collision check)

```sql
-- Expected: max <= 24 (MGC's largest known family)
SELECT MAX(member_count) AS max_family FROM edge_families;

-- Show top 5 largest families for manual inspection
SELECT family_hash, instrument, member_count, head_strategy_id, robustness_status
FROM edge_families
ORDER BY member_count DESC
LIMIT 5;
```

**ABORT if max_family > 50** — likely hash collision or empty-hash fallback.

### 2c. Every active strategy is tagged with a family

```sql
-- Expected: 0
SELECT COUNT(*) AS untagged
FROM validated_setups
WHERE LOWER(status) = 'active'
  AND family_hash IS NULL;
```

**ABORT if untagged > 0.**

### 2d. Family hash prefixes match instrument

Every family_hash should start with the instrument prefix (e.g., "MGC_", "MNQ_", "MES_").

```sql
-- Expected: 0 rows
SELECT family_hash, instrument
FROM edge_families
WHERE family_hash NOT LIKE instrument || '_%';
```

### 2e. No cross-instrument family assignment

```sql
-- A strategy should only belong to a family from its own instrument
-- Expected: 0 rows
SELECT vs.strategy_id, vs.instrument AS strat_inst, ef.instrument AS fam_inst
FROM validated_setups vs
JOIN edge_families ef ON vs.family_hash = ef.family_hash
WHERE vs.instrument != ef.instrument;
```

### 2f. Robustness classification is sane

```sql
SELECT robustness_status, COUNT(*) AS n,
       MIN(member_count) AS min_members,
       MAX(member_count) AS max_members
FROM edge_families
GROUP BY robustness_status
ORDER BY robustness_status;

-- Expected:
-- ROBUST: member_count >= 5
-- WHITELISTED: member_count in [3,4]
-- SINGLETON: member_count = 1
-- PURGED: everything else (member_count 1-2 that failed quality bars, or 3-4 that failed metrics)
-- ABORT if ROBUST has any member_count < 5
-- ABORT if SINGLETON has any member_count != 1
```

### 2g. Exactly one head per family

```sql
-- Expected: every family has exactly 1 head
SELECT family_hash, COUNT(*) AS head_count
FROM validated_setups
WHERE is_family_head = TRUE
GROUP BY family_hash
HAVING COUNT(*) != 1;

-- ABORT if any rows returned
```

### 2h. trade_day_count is non-zero

```sql
-- Expected: 0
SELECT COUNT(*) AS zero_tdc
FROM edge_families
WHERE trade_day_count = 0;

-- If > 0, the fallback to sample_size also failed — data integrity issue
```

### 2i. Head strategy exists in validated_setups and is active

```sql
-- Expected: 0 rows
SELECT ef.head_strategy_id, ef.instrument
FROM edge_families ef
LEFT JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
WHERE vs.strategy_id IS NULL OR LOWER(vs.status) != 'active';
```

---

## LAYER 3: Break Quality Research Verification

Research script `research/research_break_quality.py` produced findings for C3/C5/C8/C9. Verify the data and conclusions before implementing.

### 3a. Research output file exists and is non-empty

Check `research/output/break_quality_conditions.csv` exists and has expected columns.

```python
import pandas as pd
df = pd.read_csv('research/output/break_quality_conditions.csv')
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
print(df.head())
```

### 3b. C8 simulation: verify bar-30 break-even logic

C8 claims: if price holds cleanly outside ORB for 30 bars then reverses, moving stop to entry converts -1R losses to 0R. Verify the mechanism by spot-checking:

```python
# Load the CSV and check C8-specific rows
# C8 should trigger on 8-10% of total trades
# All triggered trades should show improvement (from negative to 0R)
import pandas as pd
df = pd.read_csv('research/output/break_quality_conditions.csv')

# Check C8 trigger rate per session
for session in ['0900', '1000', '1800']:
    session_df = df[df['orb_label'] == session] if 'orb_label' in df.columns else df[df['session'] == session]
    # Adapt column names to what's actually in the file
    print(f"\n--- Session {session} ---")
    print(f"Total rows: {len(session_df)}")
    print(session_df.describe())
```

### 3c. C3 skip slow breaks: verify 1000-only restriction

C3 (break_speed > 3min skip) is valid at 1000 ONLY. Verify it's NOT significant at 0900 or 1800.

```python
# Check if break_speed data exists and C3 results by session
# 1000: should show BH-sig improvement
# 0900, 1800: should NOT be significant
```

### 3d. Cross-check C8 year-by-year consistency

C8 claims "100% consistent across all available years". Verify:

```python
# If yearly breakdown exists in the CSV or can be derived from the data,
# check that C8's improvement is positive in every year
# ABORT if any year shows C8 HURTING performance
```

### 3e. Verify C9/C5 NO-GO conclusions

C9 was rejected because bar-30 close exit price is worse than holding. C5 hurts after C3.

```python
# C9: verify that bar-30 close prices are indeed deeply negative (-0.42R to -0.67R)
# C5: verify net delta is negative at 1000 when C3 is already applied
```

---

## LAYER 4: Cross-Layer Consistency

### 4a. Strategy IDs in edge_families match validated_setups

```sql
-- All head strategies should be in the active set
-- Expected: 0
SELECT ef.head_strategy_id
FROM edge_families ef
WHERE ef.head_strategy_id NOT IN (
    SELECT strategy_id FROM validated_setups WHERE LOWER(status) = 'active'
);
```

### 4b. Family member counts add up to total strategies

```sql
-- Sum of all member_counts should equal total active strategies
-- Expected: these two numbers match
SELECT SUM(member_count) AS family_member_total FROM edge_families;
SELECT COUNT(*) AS active_total FROM validated_setups WHERE LOWER(status) = 'active';
```

### 4c. No orphan families (families with 0 tagged strategies)

```sql
-- Every family should have at least 1 tagged strategy
SELECT ef.family_hash, ef.member_count
FROM edge_families ef
WHERE ef.family_hash NOT IN (
    SELECT DISTINCT family_hash FROM validated_setups WHERE family_hash IS NOT NULL
);

-- Expected: 0 rows
```

---

## Summary: Gate Results

Fill in as you run each check:

| Gate | Expected | Actual | PASS/FAIL |
|------|----------|--------|-----------|
| 1a. Population counts | MGC=431, MNQ=1665, MES=53 | | |
| 1b. No REGIME leaks | 0 rows | | |
| 1c. Yearly positive check | columns exist | | |
| 1d. Walk-forward coverage | 15-25% | | |
| 1e. Hash coverage | 0 missing | | |
| 1f. No duplicate IDs | 0 rows | | |
| 2a. Family counts | 131+391+32=554 | | |
| 2b. Max family size | ≤24 | | |
| 2c. Untagged strategies | 0 | | |
| 2d. Hash prefix match | 0 mismatches | | |
| 2e. Cross-instrument | 0 rows | | |
| 2f. Robustness sane | ROBUST min≥5, SINGLETON=1 | | |
| 2g. One head per family | 0 violations | | |
| 2h. trade_day_count > 0 | 0 zeros | | |
| 2i. Heads are active | 0 orphans | | |
| 3a. CSV exists | non-empty file | | |
| 3b. C8 trigger rate | 8-10% | | |
| 3c. C3 is 1000-only | BH-sig at 1000 only | | |
| 3d. C8 year consistency | all years positive | | |
| 3e. C9/C5 NO-GO | negative deltas confirmed | | |
| 4a. Heads in active set | 0 orphans | | |
| 4b. Member counts match | sum = 2149 | | |
| 4c. No orphan families | 0 rows | | |

**DECISION RULE:** If ALL gates pass → proceed to implement C8/C3 in outcome_builder and update TRADING_RULES.md. If ANY gate fails → STOP and investigate before touching production code.
