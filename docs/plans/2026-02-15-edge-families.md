# Edge Families Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Group validated strategies into edge families by hashing their trade-day lists, elect cluster heads, and expose family membership for portfolio deduplication.

**Architecture:** Hash each strategy's sorted trade-day list (from `strategy_trade_days`) into a deterministic `family_hash`. Strategies with identical hashes share the same entry pattern (same edge). Store families in a new `edge_families` table and tag each `validated_setups` row with its `family_hash` + `is_family_head` flag. The cluster head = best ExpR variant within each family.

**Tech Stack:** DuckDB, Python hashlib (MD5), existing `strategy_trade_days` table as ground truth.

**Prerequisite:** `strategy_trade_days` table must be populated (Step 1, completed 2026-02-15).

---

## Task 1: Schema — Add `edge_families` table + new columns on `validated_setups`

**Files:**
- Modify: `trading_app/db_manager.py` — `init_trading_app_schema()` (add DDL) and `verify_trading_app_schema()` (add expected columns)
- Test: `tests/test_trading_app/test_db_manager.py`

### Step 1: Write failing tests

Add to `tests/test_trading_app/test_db_manager.py`:

```python
def test_creates_edge_families_table(self, db_path):
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main'
    """).fetchall()
    table_names = {t[0] for t in tables}
    con.close()

    assert "edge_families" in table_names


def test_edge_families_columns(self, db_path):
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    cols = con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'edge_families'
    """).fetchall()
    col_names = {c[0] for c in cols}
    con.close()

    expected = {
        "family_hash", "instrument", "member_count",
        "trade_day_count", "head_strategy_id",
        "head_expectancy_r", "head_sharpe_ann",
    }
    assert expected.issubset(col_names)


def test_validated_setups_family_columns(self, db_path):
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    cols = con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'validated_setups'
    """).fetchall()
    col_names = {c[0] for c in cols}
    con.close()

    assert "family_hash" in col_names
    assert "is_family_head" in col_names


def test_edge_families_pk(self, db_path):
    init_trading_app_schema(db_path=db_path)

    con = duckdb.connect(str(db_path))
    con.execute("""
        INSERT INTO validated_setups
        (strategy_id, instrument, orb_label, orb_minutes, rr_target,
         confirm_bars, entry_model, filter_type, sample_size,
         win_rate, expectancy_r, years_tested, all_years_positive,
         stress_test_passed, status)
        VALUES ('test_s1', 'MGC', '0900', 5, 2.0, 2, 'E1', 'ORB_G5',
                100, 0.55, 0.30, 3, TRUE, TRUE, 'active')
    """)
    con.execute("""
        INSERT INTO edge_families
        (family_hash, instrument, member_count, trade_day_count,
         head_strategy_id, head_expectancy_r, head_sharpe_ann)
        VALUES ('abc123', 'MGC', 3, 100, 'test_s1', 0.30, 1.2)
    """)
    con.commit()

    with pytest.raises(duckdb.ConstraintException):
        con.execute("""
            INSERT INTO edge_families
            (family_hash, instrument, member_count, trade_day_count,
             head_strategy_id, head_expectancy_r, head_sharpe_ann)
            VALUES ('abc123', 'MGC', 1, 50, 'test_s1', 0.20, 0.8)
        """)
    con.close()
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_trading_app/test_db_manager.py -v -k "edge_families or family_columns" --no-header
```

Expected: FAIL — `edge_families` table doesn't exist, `family_hash` column doesn't exist.

### Step 3: Implement schema changes in db_manager.py

**3a. Add `edge_families` table DDL** — insert after the `strategy_trade_days` CREATE block (around line 194):

```python
# Table 6: edge_families (strategy clustering by trade-day hash)
con.execute("""
    CREATE TABLE IF NOT EXISTS edge_families (
        family_hash       TEXT        PRIMARY KEY,
        instrument        TEXT        NOT NULL,
        member_count      INTEGER     NOT NULL,
        trade_day_count   INTEGER     NOT NULL,
        head_strategy_id  TEXT        NOT NULL,
        head_expectancy_r DOUBLE,
        head_sharpe_ann   DOUBLE,
        created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (head_strategy_id)
            REFERENCES validated_setups(strategy_id)
    )
""")
```

**3b. Add columns to `validated_setups` DDL** — add before the `status` column (around line 152):

```python
-- Edge family membership
family_hash       TEXT,
is_family_head    BOOLEAN     DEFAULT FALSE,
```

**3c. Add `edge_families` to force-drop list** — before `strategy_trade_days`:

```python
con.execute("DROP TABLE IF EXISTS edge_families")
```

**3d. Add to `verify_trading_app_schema()`** — add `"edge_families"` to `expected_tables` list. Add `"family_hash"` and `"is_family_head"` to the `validated_setups` expected_cols set.

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_trading_app/test_db_manager.py -v --no-header
```

Expected: ALL PASS (including existing tests — no regressions).

### Step 5: Commit

```bash
git add trading_app/db_manager.py tests/test_trading_app/test_db_manager.py
git commit -m "feat: add edge_families table + family columns on validated_setups"
```

---

## Task 2: Build edge families — hash computation + backfill script

**Files:**
- Create: `scripts/build_edge_families.py`
- Test: `tests/test_trading_app/test_edge_families.py`

### Step 1: Write failing tests

Create `tests/test_trading_app/test_edge_families.py`:

```python
"""
Tests for edge family hash computation and family building.
"""

import hashlib
import pytest
import duckdb
from datetime import date
from pathlib import Path

from trading_app.db_manager import init_trading_app_schema


@pytest.fixture
def db_path(tmp_path):
    """Create temp DB with full schema + test data."""
    path = tmp_path / "test.db"
    con = duckdb.connect(str(path))

    # Minimal daily_features for FK
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE NOT NULL,
            symbol TEXT NOT NULL,
            orb_minutes INTEGER NOT NULL,
            bar_count_1m INTEGER,
            PRIMARY KEY (symbol, trading_day, orb_minutes)
        )
    """)
    con.close()

    init_trading_app_schema(db_path=path)

    # Insert test strategies
    con = duckdb.connect(str(path))

    # 3 strategies: s1 and s2 share same trade days (same edge), s3 is different
    for sid, orb, rr, filt, expr in [
        ("MGC_0900_E1_RR2.0_CB2_ORB_G5", "0900", 2.0, "ORB_G5", 0.30),
        ("MGC_0900_E1_RR2.5_CB2_ORB_G5", "0900", 2.5, "ORB_G5", 0.45),
        ("MGC_0900_E1_RR2.0_CB2_ORB_G8", "0900", 2.0, "ORB_G8", 0.60),
    ]:
        con.execute("""
            INSERT INTO validated_setups
            (strategy_id, instrument, orb_label, orb_minutes, rr_target,
             confirm_bars, entry_model, filter_type, sample_size,
             win_rate, expectancy_r, sharpe_ann, years_tested,
             all_years_positive, stress_test_passed, status)
            VALUES (?, 'MGC', ?, 5, ?, 2, 'E1', ?, 100, 0.55, ?, 1.0,
                    3, TRUE, TRUE, 'active')
        """, [sid, orb, rr, filt, expr])

    # s1 and s2: identical trade days (same orb, EM, CB, filter — different RR)
    for sid in [
        "MGC_0900_E1_RR2.0_CB2_ORB_G5",
        "MGC_0900_E1_RR2.5_CB2_ORB_G5",
    ]:
        for d in [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]:
            con.execute(
                "INSERT INTO strategy_trade_days VALUES (?, ?)", [sid, d]
            )

    # s3: different trade days (stricter filter)
    for d in [date(2024, 1, 2), date(2024, 1, 5)]:
        con.execute(
            "INSERT INTO strategy_trade_days VALUES (?, ?)",
            ["MGC_0900_E1_RR2.0_CB2_ORB_G8", d],
        )

    con.commit()
    con.close()
    return path


def _compute_hash(days: list[date]) -> str:
    """Reference hash computation for tests."""
    day_str = ",".join(str(d) for d in sorted(days))
    return hashlib.md5(day_str.encode()).hexdigest()


class TestHashComputation:
    """Hash is deterministic and collision-resistant."""

    def test_same_days_same_hash(self):
        days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
        assert _compute_hash(days) == _compute_hash(days)

    def test_order_independent(self):
        days_a = [date(2024, 1, 5), date(2024, 1, 2), date(2024, 1, 3)]
        days_b = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)]
        assert _compute_hash(days_a) == _compute_hash(days_b)

    def test_different_days_different_hash(self):
        days_a = [date(2024, 1, 2), date(2024, 1, 3)]
        days_b = [date(2024, 1, 2), date(2024, 1, 5)]
        assert _compute_hash(days_a) != _compute_hash(days_b)

    def test_empty_days_returns_known_sentinel(self):
        assert _compute_hash([]) == hashlib.md5(b"").hexdigest()


class TestBuildEdgeFamilies:
    """Integration: build_edge_families populates tables correctly."""

    def test_groups_by_hash(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute(
            "SELECT family_hash, member_count FROM edge_families ORDER BY member_count DESC"
        ).fetchall()
        con.close()

        # s1 + s2 share a family (2 members), s3 is alone (1 member)
        assert len(families) == 2
        assert families[0][1] == 2  # 2-member family
        assert families[1][1] == 1  # 1-member family

    def test_head_is_best_expectancy(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)

        # The 2-member family: s2 has better ExpR (0.45 > 0.30)
        family = con.execute("""
            SELECT head_strategy_id, head_expectancy_r
            FROM edge_families WHERE member_count = 2
        """).fetchone()
        con.close()

        assert family[0] == "MGC_0900_E1_RR2.5_CB2_ORB_G5"
        assert family[1] == 0.45

    def test_validated_setups_tagged(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        rows = con.execute("""
            SELECT strategy_id, family_hash, is_family_head
            FROM validated_setups
            ORDER BY strategy_id
        """).fetchall()
        con.close()

        # All 3 strategies should have a family_hash
        assert all(r[1] is not None for r in rows)

        # Exactly 2 unique hashes
        hashes = {r[1] for r in rows}
        assert len(hashes) == 2

        # s1 and s2 share the same hash
        by_id = {r[0]: (r[1], r[2]) for r in rows}
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G5"][0] == by_id["MGC_0900_E1_RR2.5_CB2_ORB_G5"][0]

        # s2 is head of the shared family (best ExpR)
        assert by_id["MGC_0900_E1_RR2.5_CB2_ORB_G5"][1] is True
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G5"][1] is False

        # s3 is head of its own family (only member)
        assert by_id["MGC_0900_E1_RR2.0_CB2_ORB_G8"][1] is True

    def test_trade_day_count(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")

        con = duckdb.connect(str(db_path), read_only=True)
        families = con.execute("""
            SELECT member_count, trade_day_count
            FROM edge_families ORDER BY member_count DESC
        """).fetchall()
        con.close()

        assert families[0] == (2, 3)  # 2 members, 3 trade days
        assert families[1] == (1, 2)  # 1 member, 2 trade days

    def test_idempotent(self, db_path):
        from scripts.build_edge_families import build_edge_families

        build_edge_families(str(db_path), "MGC")
        build_edge_families(str(db_path), "MGC")  # Run again

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
        con.close()

        assert count == 2  # No duplicates
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_trading_app/test_edge_families.py -v --no-header
```

Expected: FAIL — `scripts.build_edge_families` module doesn't exist.

### Step 3: Implement `scripts/build_edge_families.py`

```python
"""
Build edge families by hashing strategy trade-day lists.

Groups validated strategies that share identical post-filter trade-day
patterns. Elects a cluster head (best ExpR) per family.

Usage:
    python scripts/build_edge_families.py --instrument MGC --db-path C:/db/gold.db
    python scripts/build_edge_families.py --all --db-path C:/db/gold.db
"""

import sys
import hashlib
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

sys.stdout.reconfigure(line_buffering=True)


def compute_family_hash(days: list) -> str:
    """Compute deterministic MD5 hash of sorted trade-day list."""
    day_str = ",".join(str(d) for d in sorted(days))
    return hashlib.md5(day_str.encode()).hexdigest()


def _migrate_columns(con):
    """Add family_hash and is_family_head columns if missing (existing DB migration)."""
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'validated_setups'"
        ).fetchall()
    }
    if "family_hash" not in cols:
        con.execute("ALTER TABLE validated_setups ADD COLUMN family_hash TEXT")
    if "is_family_head" not in cols:
        con.execute(
            "ALTER TABLE validated_setups ADD COLUMN is_family_head BOOLEAN DEFAULT FALSE"
        )
    con.commit()


def build_edge_families(db_path: str, instrument: str) -> int:
    """
    Build edge families for one instrument.

    Returns number of unique families found.
    """
    con = duckdb.connect(str(db_path))
    try:
        _migrate_columns(con)

        # Ensure edge_families table exists
        con.execute("""
            CREATE TABLE IF NOT EXISTS edge_families (
                family_hash       TEXT        PRIMARY KEY,
                instrument        TEXT        NOT NULL,
                member_count      INTEGER     NOT NULL,
                trade_day_count   INTEGER     NOT NULL,
                head_strategy_id  TEXT        NOT NULL,
                head_expectancy_r DOUBLE,
                head_sharpe_ann   DOUBLE,
                created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 1. Load validated strategies
        strategies = con.execute("""
            SELECT strategy_id, expectancy_r, sharpe_ann
            FROM validated_setups
            WHERE instrument = ? AND LOWER(status) = 'active'
            ORDER BY strategy_id
        """, [instrument]).fetchall()

        print(f"Building edge families for {len(strategies)} {instrument} strategies")

        if not strategies:
            print(f"No active strategies for {instrument}")
            return 0

        # 2. Compute hash per strategy
        hash_map = {}  # strategy_id -> family_hash
        for sid, expr, shann in strategies:
            days = con.execute("""
                SELECT trading_day FROM strategy_trade_days
                WHERE strategy_id = ?
                ORDER BY trading_day
            """, [sid]).fetchall()

            day_list = [r[0] for r in days]
            h = compute_family_hash(day_list)
            hash_map[sid] = h

        # 3. Group by hash
        families = defaultdict(list)  # hash -> [(sid, expr, shann)]
        for sid, expr, shann in strategies:
            families[hash_map[sid]].append((sid, expr, shann))

        print(f"  {len(strategies)} strategies -> {len(families)} unique families")

        # 4. Clear existing families for this instrument
        con.execute(
            "DELETE FROM edge_families WHERE instrument = ?", [instrument]
        )

        # 5. Reset family columns on validated_setups
        con.execute("""
            UPDATE validated_setups
            SET family_hash = NULL, is_family_head = FALSE
            WHERE instrument = ?
        """, [instrument])

        # 6. For each family: elect head, insert edge_families, update validated_setups
        for family_hash, members in families.items():
            # Head = best ExpR (among members)
            members_sorted = sorted(members, key=lambda m: m[1] or 0, reverse=True)
            head_sid, head_expr, head_shann = members_sorted[0]

            # Trade day count (all members share the same days, pick any)
            trade_day_count = con.execute("""
                SELECT COUNT(*) FROM strategy_trade_days
                WHERE strategy_id = ?
            """, [head_sid]).fetchone()[0]

            # Insert edge family
            con.execute("""
                INSERT INTO edge_families
                (family_hash, instrument, member_count, trade_day_count,
                 head_strategy_id, head_expectancy_r, head_sharpe_ann)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                family_hash, instrument, len(members), trade_day_count,
                head_sid, head_expr, head_shann,
            ])

            # Tag all members with family_hash
            for sid, _, _ in members:
                is_head = sid == head_sid
                con.execute("""
                    UPDATE validated_setups
                    SET family_hash = ?, is_family_head = ?
                    WHERE strategy_id = ?
                """, [family_hash, is_head, sid])

        con.commit()

        # 7. Summary
        size_dist = sorted(
            [len(m) for m in families.values()], reverse=True
        )
        print(f"  Family sizes: max={size_dist[0]}, "
              f"median={size_dist[len(size_dist)//2]}, "
              f"singletons={size_dist.count(1)}")

        return len(families)

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build edge families from strategy trade-day hashes"
    )
    parser.add_argument("--instrument", help="Instrument symbol")
    parser.add_argument(
        "--db-path", default="C:/db/gold.db", help="Database path"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run for all instruments"
    )
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Either --instrument or --all is required")

    if args.all:
        total = 0
        for inst in ["MGC", "MNQ", "MES", "MCL"]:
            total += build_edge_families(args.db_path, inst)
            print()
        print(f"Grand total: {total} unique edge families")
    else:
        build_edge_families(args.db_path, args.instrument)


if __name__ == "__main__":
    main()
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_trading_app/test_edge_families.py -v --no-header
```

Expected: ALL PASS.

### Step 5: Run full test suite for regressions

```bash
pytest tests/ -x -q
```

Expected: ALL PASS (1032+ tests).

### Step 6: Commit

```bash
git add scripts/build_edge_families.py tests/test_trading_app/test_edge_families.py
git commit -m "feat: build edge families from trade-day hashes"
```

---

## Task 3: Run backfill on production data + verify

**Files:**
- No code changes — execution + verification only

### Step 1: Copy DB to working location

```bash
cmd /c copy "C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db" "C:\db\gold.db"
```

### Step 2: Run edge family builder for all instruments

```bash
python scripts/build_edge_families.py --all --db-path C:/db/gold.db
```

Expected output (approximate):
```
Building edge families for 216 MGC strategies
  216 strategies -> ~50-80 unique families
  Family sizes: max=6, median=3, singletons=~10

Building edge families for 610 MNQ strategies
  610 strategies -> ~80-120 unique families

Building edge families for 198 MES strategies
  198 strategies -> ~40-70 unique families

No active strategies for MCL

Grand total: ~170-270 unique edge families
```

### Step 3: Verify family membership

```bash
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db', read_only=True)

# Summary
print('=== Edge Family Summary ===')
rows = con.execute('''
    SELECT instrument, COUNT(*) as families,
           SUM(member_count) as total_strategies,
           AVG(member_count) as avg_members,
           MAX(member_count) as max_members,
           SUM(CASE WHEN member_count = 1 THEN 1 ELSE 0 END) as singletons
    FROM edge_families
    GROUP BY instrument
    ORDER BY instrument
''').fetchall()
for r in rows:
    print(f'  {r[0]}: {r[1]} families, {r[2]} strategies, '
          f'avg={r[3]:.1f} members, max={r[4]}, singletons={r[5]}')

# Verify all validated strategies have family_hash
missing = con.execute('''
    SELECT COUNT(*) FROM validated_setups
    WHERE family_hash IS NULL AND LOWER(status) = 'active'
''').fetchone()[0]
print(f'\nStrategies missing family_hash: {missing} (should be 0)')

# Verify exactly 1 head per family
multi_head = con.execute('''
    SELECT family_hash, COUNT(*) FROM validated_setups
    WHERE is_family_head = TRUE
    GROUP BY family_hash
    HAVING COUNT(*) > 1
''').fetchall()
print(f'Families with multiple heads: {len(multi_head)} (should be 0)')

# Top 5 largest families
print('\n=== Top 5 Largest Families ===')
rows = con.execute('''
    SELECT ef.family_hash, ef.instrument, ef.member_count,
           ef.trade_day_count, ef.head_strategy_id, ef.head_expectancy_r
    FROM edge_families ef
    ORDER BY ef.member_count DESC
    LIMIT 5
''').fetchall()
for r in rows:
    print(f'  {r[4]}: {r[2]} members, {r[3]} days, ExpR={r[5]}')

con.close()
"
```

### Step 4: Copy DB back to master

```bash
python -c "import shutil; shutil.copy2(r'C:\db\gold.db', r'C:\Users\joshd\OneDrive\Desktop\Canompx3\gold.db'); print('OK')"
```

### Step 5: Commit backfill results (no code changes, just verification)

No commit needed — this step is data-only.

---

## Task 4: Add drift check for edge_families write isolation

**Files:**
- Modify: `pipeline/check_drift.py` — add check that pipeline/ code doesn't write to `edge_families`
- Test: drift check passes

### Step 1: Read `pipeline/check_drift.py` to find the existing trading_app write guard pattern

Look for the existing check that blocks pipeline code from writing to trading_app tables (check #8 or similar). Add `edge_families` to the protected table list.

### Step 2: Run drift check

```bash
python pipeline/check_drift.py
```

Expected: ALL PASS.

### Step 3: Commit

```bash
git add pipeline/check_drift.py
git commit -m "chore: add edge_families to drift check write guards"
```

---

## Success Criteria

1. `edge_families` table exists with PK on `family_hash`
2. Every active `validated_setups` row has a non-NULL `family_hash`
3. Exactly 1 `is_family_head = TRUE` per `family_hash`
4. Head = best `expectancy_r` within each family
5. `member_count` matches actual count of strategies per hash
6. 1,024 validated strategies compress to ~170-270 unique families
7. All existing tests pass (1032+)
8. Drift check passes (21+ checks)

---

## What This Does NOT Include (Future Work)

- **Portfolio integration**: Modifying `portfolio.py` to select one strategy per family (Step 4 in the original roadmap — separate plan after verifying family counts)
- **Discovery auto-populate**: Modifying `strategy_discovery.py` to compute hash during grid search (Step 3 — after families are verified stable)
- **Stability scoring**: `member_count / theoretical_max` normalization (requires defining theoretical_max per grid position — deferred until family distribution is understood)
- **view_strategies.py updates**: Adding family columns to strategy viewer output

---

## Critical Reference Files

| File | Path | Purpose |
|------|------|---------|
| Trade days (ground truth) | `strategy_trade_days` table | Source for hash computation |
| DB schema | `trading_app/db_manager.py` | DDL for `edge_families` + new columns |
| Backfill script | `scripts/build_edge_families.py` | Hash + group + elect head |
| Tests | `tests/test_trading_app/test_edge_families.py` | Unit + integration tests |
| Existing schema tests | `tests/test_trading_app/test_db_manager.py` | Schema verification tests |
| Portfolio (future) | `trading_app/portfolio.py` | Will consume `edge_families` for dedup |
