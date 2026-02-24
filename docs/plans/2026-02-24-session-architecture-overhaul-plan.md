# Session Architecture Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace all time-based session names with event-based names, make every session dynamic (DST-aware), and add 2 untested high-volume sessions (COMEX_SETTLE, NYSE_CLOSE).

**Architecture:** All 11 sessions become dynamic entries in SESSION_CATALOG with per-day resolver functions. Fixed sessions are eliminated. Strategy IDs migrate via a rename script applied to all DB tables. Parallel run for new sessions before cutover.

**Tech Stack:** Python, DuckDB, pipeline/dst.py resolvers, ZoneInfo timezones.

**Design doc:** `docs/plans/2026-02-24-session-architecture-overhaul.md`

---

## Phase 1 — Add New Sessions (No Breaking Changes)

### Task 1: Add COMEX_SETTLE and NYSE_CLOSE resolver functions

**Files:**
- Modify: `pipeline/dst.py:191-266` (add after existing resolvers)
- Modify: `tests/test_pipeline/test_dst.py` (add new test classes)

**Step 1: Write failing tests**

Add to `tests/test_pipeline/test_dst.py`:

```python
class TestComexSettleBrisbane:
    """COMEX gold settlement at 1:30 PM ET."""

    def test_winter_est(self):
        # 1:30 PM EST = 18:30 UTC = 04:30 Brisbane (next day)
        h, m = comex_settle_brisbane(date(2026, 1, 15))
        assert (h, m) == (4, 30)

    def test_summer_edt(self):
        # 1:30 PM EDT = 17:30 UTC = 03:30 Brisbane (next day)
        h, m = comex_settle_brisbane(date(2025, 7, 15))
        assert (h, m) == (3, 30)

    def test_spring_transition(self):
        # Mar 8 2026 is US spring-forward
        h, m = comex_settle_brisbane(date(2026, 3, 7))  # still EST
        assert (h, m) == (4, 30)
        h, m = comex_settle_brisbane(date(2026, 3, 9))  # now EDT
        assert (h, m) == (3, 30)


class TestNyseCloseBrisbane:
    """NYSE closing bell at 4:00 PM ET."""

    def test_winter_est(self):
        # 4:00 PM EST = 21:00 UTC = 07:00 Brisbane (next day)
        h, m = nyse_close_brisbane(date(2026, 1, 15))
        assert (h, m) == (7, 0)

    def test_summer_edt(self):
        # 4:00 PM EDT = 20:00 UTC = 06:00 Brisbane (next day)
        h, m = nyse_close_brisbane(date(2025, 7, 15))
        assert (h, m) == (6, 0)

    def test_spring_transition(self):
        h, m = nyse_close_brisbane(date(2026, 3, 7))  # EST
        assert (h, m) == (7, 0)
        h, m = nyse_close_brisbane(date(2026, 3, 9))  # EDT
        assert (h, m) == (6, 0)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline/test_dst.py::TestComexSettleBrisbane -v
pytest tests/test_pipeline/test_dst.py::TestNyseCloseBrisbane -v
```
Expected: FAIL with `ImportError: cannot import name 'comex_settle_brisbane'`

**Step 3: Implement resolver functions**

Add to `pipeline/dst.py` after `cme_close_brisbane()` (after line 266):

```python
def comex_settle_brisbane(trading_day: date) -> tuple[int, int]:
    """COMEX gold settlement (01:30 PM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 01:30 PM ET = 17:30 UTC = 03:30 AEST (next cal day)
      Winter (EST): 01:30 PM ET = 18:30 UTC = 04:30 AEST (next cal day)
    """
    et_settle = datetime(trading_day.year, trading_day.month, trading_day.day,
                         13, 30, 0, tzinfo=_US_EASTERN)
    bris = et_settle.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)


def nyse_close_brisbane(trading_day: date) -> tuple[int, int]:
    """NYSE closing bell (04:00 PM ET) in Brisbane local time.

    Returns (hour, minute) in Australia/Brisbane.
      Summer (EDT): 04:00 PM ET = 20:00 UTC = 06:00 AEST (next cal day)
      Winter (EST): 04:00 PM ET = 21:00 UTC = 07:00 AEST (next cal day)
    """
    et_close = datetime(trading_day.year, trading_day.month, trading_day.day,
                        16, 0, 0, tzinfo=_US_EASTERN)
    bris = et_close.astimezone(_BRISBANE)
    return (bris.hour, bris.minute)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline/test_dst.py::TestComexSettleBrisbane -v
pytest tests/test_pipeline/test_dst.py::TestNyseCloseBrisbane -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/dst.py tests/test_pipeline/test_dst.py
git commit -m "feat: add COMEX_SETTLE and NYSE_CLOSE resolver functions"
```

---

### Task 2: Register new sessions in SESSION_CATALOG

**Files:**
- Modify: `pipeline/dst.py:277-374` (SESSION_CATALOG)
- Modify: `pipeline/dst.py:72-82` (DST_AFFECTED_SESSIONS / DST_CLEAN_SESSIONS)
- Modify: `tests/test_pipeline/test_dst.py` (update catalog tests)

**Step 1: Write failing test**

Add assertion to existing `TestSessionCatalog` or `TestDynamicOrbResolvers`:

```python
def test_new_sessions_in_catalog(self):
    assert "COMEX_SETTLE" in SESSION_CATALOG
    assert "NYSE_CLOSE" in SESSION_CATALOG
    assert SESSION_CATALOG["COMEX_SETTLE"]["type"] == "dynamic"
    assert SESSION_CATALOG["NYSE_CLOSE"]["type"] == "dynamic"

def test_new_sessions_in_resolvers(self):
    assert "COMEX_SETTLE" in DYNAMIC_ORB_RESOLVERS
    assert "NYSE_CLOSE" in DYNAMIC_ORB_RESOLVERS
```

**Step 2: Run to verify fail**

```bash
pytest tests/test_pipeline/test_dst.py::TestSessionCatalog::test_new_sessions_in_catalog -v
```
Expected: FAIL with `KeyError`

**Step 3: Add entries to SESSION_CATALOG**

Add to `pipeline/dst.py` SESSION_CATALOG dict (before the fixed sessions block):

```python
    "COMEX_SETTLE": {
        "type": "dynamic",
        "resolver": comex_settle_brisbane,
        "break_group": "us",
        "event": "COMEX gold settlement 1:30 PM ET",
    },
    "NYSE_CLOSE": {
        "type": "dynamic",
        "resolver": nyse_close_brisbane,
        "break_group": "us",
        "event": "NYSE closing bell 4:00 PM ET",
    },
```

Add to DST classification (around line 72-82). Both are US DST affected:

```python
# In DST_CLEAN_SESSIONS, add:
"COMEX_SETTLE", "NYSE_CLOSE"
```

Wait — these ARE DST-affected (US Eastern). Check: `DST_CLEAN_SESSIONS` contains the dynamic sessions because they self-resolve. Verify the existing pattern: CME_OPEN, LONDON_OPEN etc. are in `DST_CLEAN_SESSIONS` because their resolvers handle DST internally. Follow the same pattern for COMEX_SETTLE and NYSE_CLOSE.

**Step 4: Run tests**

```bash
pytest tests/test_pipeline/test_dst.py -v -x
```
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```bash
git add pipeline/dst.py tests/test_pipeline/test_dst.py
git commit -m "feat: register COMEX_SETTLE and NYSE_CLOSE in SESSION_CATALOG"
```

---

### Task 3: Add new sessions to config.py

**Files:**
- Modify: `trading_app/config.py:621-636` (ORB_DURATION_MINUTES)
- Modify: `trading_app/config.py:661-681` (EARLY_EXIT_MINUTES)

**Step 1: Add ORB duration entries**

Add to `ORB_DURATION_MINUTES`:

```python
    "COMEX_SETTLE": 5,
    "NYSE_CLOSE": 5,
```

**Step 2: Add T80 entries (None for now — no T80 data yet)**

Add to `EARLY_EXIT_MINUTES`:

```python
    "COMEX_SETTLE": None,  # No T80 data yet — new session
    "NYSE_CLOSE": None,    # No T80 data yet — new session
```

**Step 3: Add instrument-session mapping**

Check `get_enabled_sessions()` or equivalent function. Add:
- COMEX_SETTLE: MGC only (gold settlement event)
- NYSE_CLOSE: MES, MNQ, M2K (equity event)

Verify existing pattern in config.py for how instrument→session mapping works. Follow exactly.

**Step 4: Add filter grid entries**

In `get_filters_for_grid()`: add COMEX_SETTLE and NYSE_CLOSE with default filter set (ORB_G4, ORB_G5, ORB_G6, NO_FILTER at minimum). No DOW/direction overlays until we have data.

**Step 5: Run tests**

```bash
pytest tests/ -x -q
```
Expected: ALL PASS

**Step 6: Commit**

```bash
git add trading_app/config.py
git commit -m "feat: add COMEX_SETTLE and NYSE_CLOSE to config"
```

---

### Task 4: Add new sessions to build_daily_features.py

**Files:**
- Modify: `pipeline/build_daily_features.py` — ORB computation for new sessions
- Modify: `pipeline/init_db.py` — schema for new ORB columns (if needed)

**Step 1: Check how dynamic sessions are integrated**

Read `build_daily_features.py` to understand:
- How `_orb_utc_window()` resolves dynamic sessions (uses DYNAMIC_ORB_RESOLVERS)
- How new columns get created (orb_{label}_high, orb_{label}_low, etc.)
- Whether new sessions auto-discover from SESSION_CATALOG or need manual registration

Since COMEX_SETTLE and NYSE_CLOSE are already in SESSION_CATALOG as "dynamic" type, and DYNAMIC_ORB_RESOLVERS is built automatically from the catalog, they should be auto-discovered by `_orb_utc_window()`.

**Step 2: Verify auto-discovery works**

Check if `get_enabled_sessions(instrument)` includes the new sessions. If it reads from SESSION_CATALOG, the new sessions are already there. If it's a hardcoded list, add them.

**Step 3: Check break_group boundaries**

Both new sessions use break_group "us". Verify `_break_detection_window()` handles them correctly — the break window should extend from ORB end to the start of the next different break_group.

COMEX_SETTLE is at 1:30 PM ET — between the "us" cluster. Verify no break window collision with CME_CLOSE (2:45 PM CT = 3:45 PM ET).

NYSE_CLOSE is at 4:00 PM ET — the LAST session of the day for equities. Its break window extends until CME maintenance break (4:00 PM CT = 5:00 PM ET for ES products). Verify the window doesn't overflow into the next trading day.

**Step 4: Add to schema if needed**

If `init_db.py` has explicit column definitions for daily_features, add:
- `orb_COMEX_SETTLE_high`, `orb_COMEX_SETTLE_low`, `orb_COMEX_SETTLE_size`, etc.
- `orb_NYSE_CLOSE_high`, `orb_NYSE_CLOSE_low`, `orb_NYSE_CLOSE_size`, etc.

If columns are dynamic (DuckDB schema-on-write), this step may not be needed.

**Step 5: Rebuild daily_features for one instrument as test**

```bash
python pipeline/build_daily_features.py --instrument MGC --start 2025-01-01 --end 2025-01-31
```

Verify new ORB columns appear:
```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)
print(con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='daily_features' AND column_name LIKE '%COMEX%'").fetchdf())
```

**Step 6: Commit**

```bash
git add pipeline/build_daily_features.py pipeline/init_db.py
git commit -m "feat: integrate COMEX_SETTLE and NYSE_CLOSE into daily_features build"
```

---

### Task 5: Rebuild daily_features for all instruments (full date range)

**Step 1: Backup gold.db**

```bash
cp gold.db gold.db.bak.phase1
```

**Step 2: Rebuild daily_features for each instrument**

```bash
python pipeline/build_daily_features.py --instrument MGC --start 2016-02-01 --end 2026-02-04
python pipeline/build_daily_features.py --instrument MES --start 2019-02-12 --end 2026-02-11
python pipeline/build_daily_features.py --instrument MNQ --start 2021-02-04 --end 2026-02-03
python pipeline/build_daily_features.py --instrument M2K --start 2021-02-22 --end 2026-02-19
```

**Step 3: Verify new ORB data exists**

```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)
# Check COMEX_SETTLE ORBs exist for MGC
print(con.execute("""
    SELECT COUNT(*), MIN(trading_day), MAX(trading_day)
    FROM daily_features
    WHERE orb_COMEX_SETTLE_size IS NOT NULL AND symbol = 'MGC'
""").fetchdf())
# Check NYSE_CLOSE ORBs exist for MES
print(con.execute("""
    SELECT COUNT(*), MIN(trading_day), MAX(trading_day)
    FROM daily_features
    WHERE orb_NYSE_CLOSE_size IS NOT NULL AND symbol = 'MES'
""").fetchdf())
```

**Step 4: No commit needed (data rebuild only)**

---

### Task 6: Build outcomes for new sessions

**Step 1: Run outcome_builder for new sessions**

Verify that outcome_builder picks up the new sessions from `get_enabled_sessions()`.

```bash
python trading_app/outcome_builder.py --instrument MGC --force --start 2016-02-01 --end 2026-02-04
```

IMPORTANT: `--force` rebuilds ALL sessions including new ones. If this takes too long, check if there's a `--session` flag to rebuild only specific sessions.

**Step 2: Verify new outcomes exist**

```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)
print(con.execute("""
    SELECT orb_label, COUNT(*) as n
    FROM orb_outcomes
    WHERE orb_label IN ('COMEX_SETTLE', 'NYSE_CLOSE')
    GROUP BY orb_label
""").fetchdf())
```

**Step 3: Repeat for other instruments**

```bash
python trading_app/outcome_builder.py --instrument MES --force --start 2019-02-12 --end 2026-02-11
python trading_app/outcome_builder.py --instrument MNQ --force --start 2021-02-04 --end 2026-02-03
python trading_app/outcome_builder.py --instrument M2K --force --start 2021-02-22 --end 2026-02-19
```

---

### Task 7: Run discovery + validation on new sessions

**Step 1: Discovery**

```bash
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_discovery.py --instrument MES
python trading_app/strategy_discovery.py --instrument MNQ
python trading_app/strategy_discovery.py --instrument M2K
```

**Step 2: Validation**

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MES --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MNQ --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument M2K --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

**Step 3: Check results for COMEX_SETTLE and NYSE_CLOSE**

```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)
print(con.execute("""
    SELECT orb_label, instrument, COUNT(*) as n_validated, AVG(expectancy_r) as avg_expr
    FROM validated_setups
    WHERE orb_label IN ('COMEX_SETTLE', 'NYSE_CLOSE')
    GROUP BY orb_label, instrument
""").fetchdf())
```

**Step 4: Document findings** — Record whether COMEX_SETTLE and NYSE_CLOSE produce validated strategies. If yes, this is a major discovery. If no, still keep the sessions for completeness.

**Step 5: Commit config/code changes only (not DB data)**

```bash
git add -A
git commit -m "feat: COMEX_SETTLE and NYSE_CLOSE discovery results"
```

---

## Phase 2 — Validate Dynamic vs Fixed (Summer Performance)

### Task 8: Write summer comparison analysis

**Files:**
- Create: `research/research_session_alignment_audit.py`

**Step 1: Write analysis script**

For each fixed↔dynamic pair, compare summer-only performance:
- `0900` vs `CME_OPEN` (CME_REOPEN)
- `1800` vs `LONDON_OPEN` (LONDON_METALS)
- `0030` vs `US_EQUITY_OPEN` (NYSE_OPEN)
- `2300` vs `US_DATA_OPEN` (US_DATA_830)
- `US_POST_EQUITY` vs same (for MGC — was equity-only, now all)

Query pattern:
```sql
-- For each pair, get summer outcomes side-by-side
SELECT
    o_fixed.orb_label as fixed_label,
    o_dynamic.orb_label as dynamic_label,
    AVG(o_fixed.pnl_r) as fixed_avg_r,
    AVG(o_dynamic.pnl_r) as dynamic_avg_r,
    COUNT(*) as n
FROM orb_outcomes o_fixed
JOIN orb_outcomes o_dynamic
  ON o_fixed.trading_day = o_dynamic.trading_day
  AND o_fixed.symbol = o_dynamic.symbol
  AND o_fixed.rr_target = o_dynamic.rr_target
  AND o_fixed.confirm_bars = o_dynamic.confirm_bars
  AND o_fixed.entry_model = o_dynamic.entry_model
WHERE o_fixed.orb_label = '0900'
  AND o_dynamic.orb_label = 'CME_OPEN'
  AND EXTRACT(MONTH FROM o_fixed.trading_day) IN (4,5,6,7,8,9,10)  -- summer months
```

Compare: win rate, ExpR, Sharpe for each pair.

**Step 2: Run analysis**

```bash
python research/research_session_alignment_audit.py
```

**Step 3: Document results**

If dynamic outperforms fixed in summer → confirms the 1-hour offset matters.
If equal → the offset doesn't affect edge (rename is still good for clarity).

**Step 4: Commit**

```bash
git add research/research_session_alignment_audit.py
git commit -m "research: summer session alignment audit — fixed vs dynamic"
```

---

## Phase 3 — Rename + Cutover

### Task 9: Create session rename migration script

**Files:**
- Create: `scripts/tools/migrate_session_names.py`

**Step 1: Define the rename map**

```python
SESSION_RENAME_MAP = {
    # Fixed → Dynamic event name
    "0900": "CME_REOPEN",
    "1000": "TOKYO_OPEN",
    "1100": "SINGAPORE_OPEN",
    "1130": "SINGAPORE_OPEN",   # Merge into SINGAPORE_OPEN
    "1800": "LONDON_METALS",
    "2300": "US_DATA_830",
    "0030": "NYSE_OPEN",
    # Dynamic → Renamed dynamic
    "CME_OPEN": "CME_REOPEN",
    "US_EQUITY_OPEN": "NYSE_OPEN",
    "US_DATA_OPEN": "US_DATA_830",
    "LONDON_OPEN": "LONDON_METALS",
    "US_POST_EQUITY": "US_DATA_1000",
    "CME_CLOSE": "CME_PRECLOSE",
}
```

**Step 2: Write migration script**

The script must update these tables:
1. `orb_outcomes.orb_label` — UPDATE SET orb_label = new WHERE orb_label = old
2. `validated_setups.orb_label` — same
3. `validated_setups.strategy_id` — string replace old session name with new
4. `experimental_strategies` — same pattern (check schema for columns)
5. `edge_families` — same pattern
6. `daily_features` — column RENAME (orb_{old}_high → orb_{new}_high, etc.)

For tables with strategy_id: use `REPLACE(strategy_id, '_0900_', '_CME_REOPEN_')`.

CRITICAL: Handle `1130` → `SINGAPORE_OPEN` merge carefully. If both `1100` and `1130` outcomes exist, and both map to `SINGAPORE_OPEN`, there could be duplicates. The `1130` data should be DROPPED (it was an alias, not a primary session).

**Step 3: Add --dry-run flag** that prints planned changes without executing.

**Step 4: Add --backup flag** that copies gold.db before migration.

**Step 5: Test on a copy of gold.db**

```bash
cp gold.db /c/db/gold_migration_test.db
DUCKDB_PATH=/c/db/gold_migration_test.db python scripts/tools/migrate_session_names.py --dry-run
DUCKDB_PATH=/c/db/gold_migration_test.db python scripts/tools/migrate_session_names.py
```

Verify:
```python
import duckdb
con = duckdb.connect('/c/db/gold_migration_test.db', read_only=True)
# No old names should remain
print(con.execute("SELECT DISTINCT orb_label FROM orb_outcomes ORDER BY orb_label").fetchdf())
print(con.execute("SELECT DISTINCT orb_label FROM validated_setups ORDER BY orb_label").fetchdf())
# Spot-check strategy IDs
print(con.execute("SELECT strategy_id FROM validated_setups LIMIT 10").fetchdf())
```

**Step 6: Commit**

```bash
git add scripts/tools/migrate_session_names.py
git commit -m "feat: session name migration script (dry-run tested)"
```

---

### Task 10: Update SESSION_CATALOG to event-based names

**Files:**
- Modify: `pipeline/dst.py:277-374`

**Step 1: Replace SESSION_CATALOG**

Remove ALL fixed session entries. Remove aliases. Replace with 11 dynamic entries:

```python
SESSION_CATALOG = {
    "CME_REOPEN": {
        "type": "dynamic",
        "resolver": cme_open_brisbane,
        "break_group": "cme",
        "event": "CME Globex electronic reopen 5:00 PM CT",
    },
    "TOKYO_OPEN": {
        "type": "dynamic",
        "resolver": tokyo_open_brisbane,
        "break_group": "asia",
        "event": "Tokyo Stock Exchange open 9:00 AM JST",
    },
    "SINGAPORE_OPEN": {
        "type": "dynamic",
        "resolver": singapore_open_brisbane,
        "break_group": "asia",
        "event": "SGX/HKEX open 9:00 AM SGT",
    },
    "LONDON_METALS": {
        "type": "dynamic",
        "resolver": london_open_brisbane,
        "break_group": "london",
        "event": "London metals AM session 8:00 AM London",
    },
    "US_DATA_830": {
        "type": "dynamic",
        "resolver": us_data_open_brisbane,
        "break_group": "us",
        "event": "US economic data release 8:30 AM ET",
    },
    "NYSE_OPEN": {
        "type": "dynamic",
        "resolver": us_equity_open_brisbane,
        "break_group": "us",
        "event": "NYSE cash open 9:30 AM ET",
    },
    "US_DATA_1000": {
        "type": "dynamic",
        "resolver": us_post_equity_brisbane,
        "break_group": "us",
        "event": "US 10:00 AM data (ISM/CC) + post-equity-open flow",
    },
    "COMEX_SETTLE": {
        "type": "dynamic",
        "resolver": comex_settle_brisbane,
        "break_group": "us",
        "event": "COMEX gold settlement 1:30 PM ET",
    },
    "CME_PRECLOSE": {
        "type": "dynamic",
        "resolver": cme_close_brisbane,
        "break_group": "us",
        "event": "CME equity futures pre-settlement 2:45 PM CT",
    },
    "NYSE_CLOSE": {
        "type": "dynamic",
        "resolver": nyse_close_brisbane,
        "break_group": "us",
        "event": "NYSE closing bell 4:00 PM ET",
    },
}
```

**Step 2: Add resolver functions for TOKYO_OPEN and SINGAPORE_OPEN**

These are "dynamic" for consistency but resolve to fixed times (no DST in Japan/Singapore):

```python
def tokyo_open_brisbane(trading_day: date) -> tuple[int, int]:
    """Tokyo Stock Exchange open (9:00 AM JST) in Brisbane local time.
    JST = UTC+9, Brisbane = UTC+10. Always 10:00 Brisbane. No DST."""
    return (10, 0)

def singapore_open_brisbane(trading_day: date) -> tuple[int, int]:
    """SGX/HKEX open (9:00 AM SGT) in Brisbane local time.
    SGT = UTC+8, Brisbane = UTC+10. Always 11:00 Brisbane. No DST."""
    return (11, 0)
```

**Step 3: Update DST_AFFECTED_SESSIONS and DST_CLEAN_SESSIONS**

Remove old fixed session references. All 11 sessions go into DST_CLEAN_SESSIONS (dynamic resolvers handle DST internally).

**Step 4: Update DOW_ALIGNED_SESSIONS and DOW_MISALIGNED_SESSIONS**

Replace old names:
```python
DOW_ALIGNED_SESSIONS = {"CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS", "US_DATA_830", ...}
DOW_MISALIGNED_SESSIONS = {"NYSE_OPEN": -1}  # was "0030": -1
```

**Step 5: Update all tests**

Update `tests/test_pipeline/test_dst.py` — change all references from old names to new names. Update expected DYNAMIC_ORB_RESOLVERS count from 6+2=8 to 11 (was 6, added 2 new, renamed rest).

**Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```

**Step 7: Commit**

```bash
git add pipeline/dst.py tests/test_pipeline/test_dst.py
git commit -m "feat: replace SESSION_CATALOG with 11 event-based dynamic sessions"
```

---

### Task 11: Update config.py with new session names

**Files:**
- Modify: `trading_app/config.py:621-636` (ORB_DURATION_MINUTES)
- Modify: `trading_app/config.py:661-681` (EARLY_EXIT_MINUTES)
- Modify: `trading_app/config.py:517-599` (get_filters_for_grid)

**Step 1: Replace ORB_DURATION_MINUTES keys**

```python
ORB_DURATION_MINUTES = {
    "CME_REOPEN": 5,
    "TOKYO_OPEN": 15,       # Was "1000", 15min aperture
    "SINGAPORE_OPEN": 5,
    "LONDON_METALS": 5,
    "US_DATA_830": 5,
    "NYSE_OPEN": 5,
    "US_DATA_1000": 5,
    "COMEX_SETTLE": 5,
    "CME_PRECLOSE": 5,
    "NYSE_CLOSE": 5,
}
```

**Step 2: Replace EARLY_EXIT_MINUTES keys**

```python
EARLY_EXIT_MINUTES = {
    "CME_REOPEN": 38,       # Was "0900", T80=38m
    "TOKYO_OPEN": 39,       # Was "1000", T80=39m
    "SINGAPORE_OPEN": 31,   # Was "1100", T80=31m
    "LONDON_METALS": 36,    # Was "1800", T80=36m
    "US_DATA_830": None,    # Was "2300"
    "NYSE_OPEN": None,      # Was "0030"
    "US_DATA_1000": None,   # Was US_POST_EQUITY
    "COMEX_SETTLE": None,   # New
    "CME_PRECLOSE": 16,     # Was CME_CLOSE, T80=16m
    "NYSE_CLOSE": None,     # New
}
```

**Step 3: Update get_filters_for_grid()**

Replace all session name references. E.g., `if session == "0900"` → `if session == "CME_REOPEN"`.

**Step 4: Update get_enabled_sessions()**

Replace session name references and instrument mappings.

**Step 5: Run tests**

```bash
pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add trading_app/config.py
git commit -m "feat: update config.py to event-based session names"
```

---

### Task 12: Update build_daily_features.py

**Files:**
- Modify: `pipeline/build_daily_features.py:66-74` (ORB_TIMES_LOCAL — remove or replace)
- Modify: `pipeline/build_daily_features.py:79-83` (SESSION_WINDOWS)
- Modify: `pipeline/build_daily_features.py:86-96` (ORB_LABELS, validation)

**Step 1: Remove ORB_TIMES_LOCAL**

This dict is only used for fixed sessions. Since all sessions are now dynamic, replace with DYNAMIC_ORB_RESOLVERS lookup (already exists via SESSION_CATALOG).

**Step 2: Update ORB_LABELS list**

Replace with the 11 new session names.

**Step 3: Update SESSION_WINDOWS**

These are stat-only approximations. Update keys to new names or remove if no longer needed.

**Step 4: Update fail-closed validation**

The check at lines 90-96 verifies all ORB_LABELS are in DST_AFFECTED_SESSIONS or DST_CLEAN_SESSIONS. Update to match new session names.

**Step 5: Run tests**

```bash
pytest tests/test_pipeline/test_build_daily_features.py -v
pytest tests/ -x -q
```

**Step 6: Commit**

```bash
git add pipeline/build_daily_features.py
git commit -m "feat: update build_daily_features to event-based session names"
```

---

### Task 13: Update outcome_builder.py and strategy_discovery.py

**Files:**
- Modify: `trading_app/outcome_builder.py`
- Modify: `trading_app/strategy_discovery.py`
- Modify: `trading_app/strategy_validator.py`

**Step 1: Update imports**

Replace any imports of old session names (ORB_LABELS, etc.).

**Step 2: Update session iteration**

Both files use `get_enabled_sessions(instrument)` — if that function is updated (Task 11), these files just need import updates.

**Step 3: Update strategy_id construction**

If strategy_id is built as `f"{instrument}_{orb_label}_..."`, verify the new names produce valid IDs (no special chars, reasonable length).

**Step 4: Grep for any remaining old session names**

```bash
grep -rn '"0900"\|"1000"\|"1100"\|"1800"\|"2300"\|"0030"\|"1130"' trading_app/ pipeline/ tests/ --include="*.py"
```

Fix ALL occurrences.

**Step 5: Run full test suite + drift check**

```bash
pytest tests/ -x -q
python pipeline/check_drift.py
```

**Step 6: Commit**

```bash
git add trading_app/ pipeline/ tests/
git commit -m "feat: update all pipeline + trading_app to event-based session names"
```

---

### Task 14: Run migration on gold.db

**Step 1: Backup**

```bash
cp gold.db gold.db.bak.pre-migration
```

**Step 2: Dry run**

```bash
python scripts/tools/migrate_session_names.py --dry-run
```

Review output. Verify every table/column rename looks correct.

**Step 3: Execute migration**

```bash
python scripts/tools/migrate_session_names.py
```

**Step 4: Verify**

```python
import duckdb
con = duckdb.connect('gold.db', read_only=True)
# No old names
old_names = con.execute("SELECT DISTINCT orb_label FROM orb_outcomes WHERE orb_label IN ('0900','1000','1100','1130','1800','2300','0030','CME_OPEN','US_EQUITY_OPEN','US_DATA_OPEN','LONDON_OPEN','US_POST_EQUITY','CME_CLOSE')").fetchdf()
assert len(old_names) == 0, f"Old names still present: {old_names}"
# All new names present
new_names = con.execute("SELECT DISTINCT orb_label FROM validated_setups ORDER BY orb_label").fetchdf()
print(new_names)
```

**Step 5: Run health check**

```bash
python pipeline/health_check.py
python pipeline/check_drift.py
```

---

### Task 15: Full rebuild with new names

**Step 1: Rebuild daily_features**

```bash
python pipeline/build_daily_features.py --instrument MGC --start 2016-02-01 --end 2026-02-04
python pipeline/build_daily_features.py --instrument MES --start 2019-02-12 --end 2026-02-11
python pipeline/build_daily_features.py --instrument MNQ --start 2021-02-04 --end 2026-02-03
python pipeline/build_daily_features.py --instrument M2K --start 2021-02-22 --end 2026-02-19
```

**Step 2: Rebuild outcomes**

```bash
python trading_app/outcome_builder.py --instrument MGC --force --start 2016-02-01 --end 2026-02-04
python trading_app/outcome_builder.py --instrument MES --force --start 2019-02-12 --end 2026-02-11
python trading_app/outcome_builder.py --instrument MNQ --force --start 2021-02-04 --end 2026-02-03
python trading_app/outcome_builder.py --instrument M2K --force --start 2021-02-22 --end 2026-02-19
```

**Step 3: Re-run discovery + validation**

```bash
python trading_app/strategy_discovery.py --instrument MGC
python trading_app/strategy_discovery.py --instrument MES
python trading_app/strategy_discovery.py --instrument MNQ
python trading_app/strategy_discovery.py --instrument M2K

python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MES --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MNQ --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument M2K --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
```

**Step 4: Rebuild edge families**

```bash
python scripts/tools/build_edge_families.py --instrument MGC
python scripts/tools/build_edge_families.py --instrument MES
python scripts/tools/build_edge_families.py --instrument MNQ
python scripts/tools/build_edge_families.py --instrument M2K
```

---

## Phase 4 — Clean Up

### Task 16: Update all documentation

**Files:**
- Modify: `TRADING_RULES.md` — replace all old session names
- Modify: `PROP_PLAYS.md` — already updated with event names from earlier
- Modify: `CLAUDE.md` — update session references
- Modify: `RESEARCH_RULES.md` — if any session names mentioned
- Modify: `docs/DST_CONTAMINATION.md` — update DST session list

**Step 1: Global search-and-replace old session names in docs**

```bash
grep -rn '"0900"\|"1000"\|"1100"\|"1800"\|"2300"\|"0030"' *.md docs/ --include="*.md"
```

Replace each occurrence with the event-based name.

**Step 2: Update PROP_PLAYS.md session references**

Replace slot names:
- `MGC_0900` → `MGC_CME_REOPEN`
- `MGC_1000` → `MGC_TOKYO_OPEN`
- `MGC_1800` → `MGC_LONDON_METALS`
- `MNQ_0030` → `MNQ_NYSE_OPEN`
- etc.

**Step 3: Commit**

```bash
git add *.md docs/
git commit -m "docs: update all documentation to event-based session names"
```

---

### Task 17: Update drift checks + final validation

**Files:**
- Modify: `pipeline/check_drift.py` — update any hardcoded session names in drift checks

**Step 1: Run drift check**

```bash
python pipeline/check_drift.py
```

If any checks reference old session names, update them.

**Step 2: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: ALL PASS

**Step 3: Run health check**

```bash
python pipeline/health_check.py
```

**Step 4: Generate updated repo map**

```bash
python scripts/tools/gen_repo_map.py
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: session architecture overhaul complete — all sessions event-based"
```

---

## Verification Checklist

- [ ] All 11 sessions have resolver functions in dst.py
- [ ] SESSION_CATALOG contains exactly 11 entries, all "dynamic" type
- [ ] No fixed session entries remain
- [ ] No old session names in any .py file (grep confirms)
- [ ] No old session names in any .md file (grep confirms)
- [ ] orb_outcomes contains all 11 session labels
- [ ] validated_setups uses event-based strategy IDs
- [ ] COMEX_SETTLE results documented (edge or no-edge)
- [ ] NYSE_CLOSE results documented (edge or no-edge)
- [ ] Summer comparison results documented (fixed vs dynamic)
- [ ] All tests pass
- [ ] All drift checks pass
- [ ] Health check passes
- [ ] PROP_PLAYS.md uses event-based names
