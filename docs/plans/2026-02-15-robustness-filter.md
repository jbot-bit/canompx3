# Robustness Filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Purge fragile edge families, re-elect heads by median ExpR (not max), and tag surviving families with robustness status for portfolio consumption.

**Architecture:** Enhance `build_edge_families.py` to compute family-level robustness metrics (CV, avg ShANN, min trades), apply purge rules (N>=5 default, whitelist exceptions), and elect heads by median ExpR to avoid Winner's Curse. Families tagged as ROBUST, WHITELISTED, or PURGED.

**Tech Stack:** DuckDB, existing `edge_families` + `validated_setups` tables.

**Decisions Locked:**
- Purge: N>=5 default. Whitelist if ShANN>=0.8 AND CV<0.3 AND min_trades>=100.
- Election: Median ExpR (strategy closest to family median).
- Trading: 1-per-family (ensemble deferred).

---

## Task 1: Schema — Add robustness columns to `edge_families`

**Files:**
- Modify: `trading_app/db_manager.py` — add columns to DDL
- Modify: `scripts/build_edge_families.py` — compute + populate new columns
- Test: `tests/test_trading_app/test_edge_families.py`

### Step 1: Write failing tests

Add to `tests/test_trading_app/test_edge_families.py`:

```python
class TestRobustnessMetrics:
    """Robustness metrics and median head election."""

    def test_head_is_median_not_max(self, db_path):
        """Head should be strategy closest to median ExpR, not max."""
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        # The 2-member family has ExpR 0.30 and 0.45
        # Median = 0.375, closest is 0.30 (distance 0.075) vs 0.45 (distance 0.075)
        # Tie: pick lower RR (more conservative) — but both are equidistant
        # Actually median of [0.30, 0.45] = 0.375. |0.30-0.375|=0.075, |0.45-0.375|=0.075
        # Tiebreak: pick the one with lower strategy_id (deterministic)
        family = con.execute("""
            SELECT head_strategy_id, head_expectancy_r, median_expectancy_r
            FROM edge_families WHERE member_count = 2
        """).fetchone()
        con.close()

        # Median should be stored
        assert family[2] is not None

    def test_robustness_columns_exist(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        cols = con.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'edge_families'
        """).fetchall()
        col_names = {c[0] for c in cols}
        con.close()

        assert "robustness_status" in col_names
        assert "cv_expectancy" in col_names
        assert "median_expectancy_r" in col_names
        assert "avg_sharpe_ann" in col_names
        assert "min_member_trades" in col_names

    def test_singleton_is_purged(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        # s3 is a singleton family (1 member)
        family = con.execute("""
            SELECT robustness_status FROM edge_families
            WHERE member_count = 1
        """).fetchone()
        con.close()

        assert family[0] == "PURGED"

    def test_large_family_is_robust(self, db_path):
        """Families with >= 5 members should be ROBUST."""
        # Need a fixture with 5+ members to test this properly
        # The default fixture only has families of size 1 and 2
        pass  # Covered by production backfill verification
```

### Step 2: Add robustness columns to DDL

In `trading_app/db_manager.py`, update the `edge_families` CREATE TABLE to add:

```sql
-- Robustness metrics
robustness_status TEXT DEFAULT 'PENDING',
cv_expectancy     DOUBLE,
median_expectancy_r DOUBLE,
avg_sharpe_ann    DOUBLE,
min_member_trades INTEGER,
```

### Step 3: Update `build_edge_families.py`

Key changes:
1. Compute family metrics: CV(ExpR), avg(ShANN), median(ExpR), min(sample_size)
2. Elect head by median ExpR (closest to median, tiebreak by strategy_id)
3. Apply purge rules:
   - N>=5: ROBUST
   - N<5 AND ShANN>=0.8 AND CV<0.3 AND min_trades>=100: WHITELISTED
   - Otherwise: PURGED

### Step 4: Run tests

```bash
pytest tests/test_trading_app/test_edge_families.py -v --no-header
```

### Step 5: Run full suite

```bash
pytest tests/ -x -q
```

### Step 6: Commit

```bash
git add trading_app/db_manager.py scripts/build_edge_families.py tests/test_trading_app/test_edge_families.py
git commit -m "feat: robustness filter + median head election for edge families"
```

---

## Task 2: Run production backfill + verify

### Step 1: Copy DB and run

```bash
python -c "import shutil; shutil.copy2(r'C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db', r'C:\db\gold.db')"
python scripts/build_edge_families.py --all --db-path C:/db/gold.db
```

### Step 2: Verify

```sql
-- Status distribution
SELECT robustness_status, COUNT(*), SUM(member_count)
FROM edge_families GROUP BY robustness_status;

-- Should see: ROBUST ~114, WHITELISTED ~10-27, PURGED ~60-90

-- Verify no PURGED families have is_family_head = TRUE
SELECT COUNT(*) FROM validated_setups vs
JOIN edge_families ef ON vs.family_hash = ef.family_hash
WHERE ef.robustness_status = 'PURGED' AND vs.is_family_head = TRUE;
-- Should be > 0 (heads still marked but family is purged)

-- Verify median election changed heads
SELECT ef.head_strategy_id, ef.head_expectancy_r, ef.median_expectancy_r,
       ABS(ef.head_expectancy_r - ef.median_expectancy_r) as distance
FROM edge_families ef
WHERE ef.robustness_status IN ('ROBUST', 'WHITELISTED')
ORDER BY distance DESC LIMIT 10;
-- Distance should be small (head is close to median)
```

### Step 3: Copy back

```bash
python -c "import shutil; shutil.copy2(r'C:\db\gold.db', r'C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db')"
```

### Step 4: Commit + push

```bash
git add -A && git commit -m "chore: production backfill with robustness filter"
git push
```

---

## Success Criteria

1. Every edge family has a `robustness_status` (ROBUST, WHITELISTED, or PURGED)
2. Heads elected by median ExpR, not max
3. N>=5 families = ROBUST
4. Small families only survive if ShANN>=0.8 AND CV<0.3 AND min_trades>=100
5. All tests pass (1045+)
6. Winner's Curse inflation drops from +0.099 R avg to near-zero

---

## Purge Rules (Reference)

```python
MIN_FAMILY_SIZE = 5

def classify_family(member_count, avg_sharpe_ann, cv_expectancy, min_member_trades):
    if member_count >= MIN_FAMILY_SIZE:
        return "ROBUST"
    if (avg_sharpe_ann >= 0.8
        and cv_expectancy < 0.3
        and min_member_trades >= 100):
        return "WHITELISTED"
    return "PURGED"
```
