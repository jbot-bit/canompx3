# M6E (EUR/USD) Onboarding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ingest 6E EUR/USD futures data and run the full ORB discovery pipeline for M6E.

**Architecture:** M6E is already fully configured in code (asset_configs.py, cost_model.py, config.py, dst.py). Zero code changes needed. This plan is purely data extraction + sequential pipeline execution. Use scratch DB workflow for long-running writes.

**Tech Stack:** DuckDB, Databento DBN, Python pipeline scripts, bash

**Data:** `DB/6E_DB.zip` — 1,557 daily .dbn.zst files, 2019-02-13 to 2024-02-09 (5 years, ~300 trading days/year)

---

## Pre-flight Checks

Before starting, verify:
```bash
# Confirm source data exists
ls -lh DB/6E_DB.zip          # should be ~38MB
ls DB/M6E_DB 2>/dev/null && echo "EXISTS" || echo "NOT YET"

# Confirm no active DB connections
# (nothing else writing to gold.db)

# Confirm gold.db is canonical (not stale)
python -c "import duckdb; c=duckdb.connect('gold.db',read_only=True); print(c.execute('SELECT COUNT(*) FROM bars_1m WHERE symbol=\'MGC\'').fetchone())"
# Expect: ~3,500,000+ rows
```

---

### Task 1: Extract 6E data into M6E_DB directory

**Files:**
- Create: `DB/M6E_DB/` (directory with 1,557 .dbn.zst files)

**Step 1: Extract zip**
```bash
cd DB
python -c "
import zipfile, os
os.makedirs('M6E_DB', exist_ok=True)
with zipfile.ZipFile('6E_DB.zip') as z:
    z.extractall('M6E_DB')
print('Extracted:', len(os.listdir('M6E_DB')), 'files')
"
```
Expected output: `Extracted: 1559 files` (1557 .dbn.zst + metadata.json + condition.json)

**Step 2: Verify**
```bash
python -c "
import os
files = [f for f in os.listdir('DB/M6E_DB') if f.endswith('.dbn.zst')]
files.sort()
print('DBN files:', len(files))
print('First:', files[0])
print('Last:', files[-1])
"
```
Expected: 1557 files, first ≈ `glbx-mdp3-20190213.ohlcv-1m.dbn.zst`, last ≈ `glbx-mdp3-20240209.ohlcv-1m.dbn.zst`

---

### Task 2: Set up scratch DB and ingest M6E bars

**Why scratch:** Ingestion is a long write (~20-40 min). Use C:/db/gold.db to avoid locking project gold.db.

**Step 1: Copy to scratch**
```bash
cp gold.db C:/db/gold.db
export DUCKDB_PATH=C:/db/gold.db
echo "Scratch DB ready"
python -c "from pipeline.paths import GOLD_DB_PATH; print('DB path:', GOLD_DB_PATH)"
# Expect: [DB] Using scratch override: C:\db\gold.db
```

**Step 2: Run ingestion**
```bash
python pipeline/ingest_dbn.py --instrument M6E --start 2019-02-01 --end 2024-02-10 2>&1 | tee logs/ingest_m6e.log
```
Expected: Progress bars per year. Final output: `Ingested N bars for M6E`

Watch for errors:
- `outright_pattern` mismatch → contracts not matching `^M6E[HMUZ]\d{1,2}$` — check logs
- `UTC proof` failure → skip and continue (expected for some edge dates)

**Step 3: Verify bars_1m**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT MIN(ts_utc)::DATE, MAX(ts_utc)::DATE, COUNT(*) FROM bars_1m WHERE symbol='M6E'\").fetchone()
print('M6E bars_1m:', r)
"
```
Expected: 2019-02-xx to 2024-02-09, ~700,000+ bars

---

### Task 3: Build 5-minute bars for M6E

**Step 1: Run build_bars_5m**
```bash
python pipeline/build_bars_5m.py --instrument M6E --start 2019-02-01 --end 2024-02-10 2>&1 | tee logs/bars5m_m6e.log
```
Expected: Aggregates 1m → 5m. Fast (~2-3 min).

**Step 2: Verify**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT COUNT(*), MIN(ts_utc)::DATE, MAX(ts_utc)::DATE FROM bars_5m WHERE symbol='M6E'\").fetchone()
print('M6E bars_5m:', r)
"
```
Expected: ~140,000+ bars

---

### Task 4: Build daily features for M6E

**Step 1: Run build_daily_features**
```bash
python pipeline/build_daily_features.py --instrument M6E --start 2019-02-01 --end 2024-02-10 2>&1 | tee logs/features_m6e.log
```
Expected: ORBs computed for enabled sessions: 1000, 1100, 1800, LONDON_OPEN, US_DATA_OPEN, US_EQUITY_OPEN, 0030, US_POST_EQUITY. ~5-10 min.

**Step 2: Verify**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# Year breakdown
rows = c.execute(\"SELECT YEAR(trading_day), COUNT(DISTINCT trading_day) FROM daily_features WHERE symbol='M6E' AND orb_minutes=5 GROUP BY 1 ORDER BY 1\").fetchall()
for r in rows: print(r)
# Check key sessions have data
r2 = c.execute(\"SELECT COUNT(*) FROM daily_features WHERE symbol='M6E' AND orb_minutes=5 AND orb_london_open_high IS NOT NULL\").fetchone()
print('LONDON_OPEN ORBs:', r2[0])
"
```
Expected: ~250-293 days per year (2019-2024), LONDON_OPEN ORBs > 1000

---

### Task 5: Build pre-computed outcomes for M6E

**Step 1: Run outcome_builder**
```bash
python trading_app/outcome_builder.py --instrument M6E --start 2019-02-01 --end 2024-02-10 2>&1 | tee logs/outcomes_m6e.log
```
This is the long step (~30-60 min). Computes outcomes for all RR × CB × entry model × session combinations.

**Step 2: Verify**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT orb_label, COUNT(*) FROM orb_outcomes WHERE symbol='M6E' GROUP BY orb_label ORDER BY 2 DESC\").fetchall()
for row in r: print(row)
"
```
Expected: rows for each enabled session (LONDON_OPEN, US_DATA_OPEN, 1800, 0030, etc.)

---

### Task 6: Run strategy discovery for M6E

**Step 1: Run discovery**
```bash
python trading_app/strategy_discovery.py --instrument M6E 2>&1 | tee logs/discovery_m6e.log
```
Expected: Grid search across all filter × RR × CB × entry model combos. ~10-20 min.

**Step 2: Verify**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT orb_label, COUNT(*) FROM experimental_strategies WHERE instrument='M6E' GROUP BY orb_label ORDER BY 2 DESC\").fetchall()
for row in r: print(row)
"
```

---

### Task 7: Run strategy validation for M6E

**Step 1: Run validator**
```bash
python trading_app/strategy_validator.py --instrument M6E --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward 2>&1 | tee logs/validate_m6e.log
```

**Step 2: Verify**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT orb_label, COUNT(*) FROM validated_setups WHERE instrument='M6E' GROUP BY orb_label ORDER BY 2 DESC LIMIT 20\").fetchall()
for row in r: print(row)
"
```

---

### Task 8: Build edge families for M6E

**Step 1: Run edge families**
```bash
python scripts/tools/build_edge_families.py --instrument M6E 2>&1 | tee logs/families_m6e.log
```

**Step 2: Check results**
```bash
python -c "
import duckdb, os
os.environ['DUCKDB_PATH'] = 'C:/db/gold.db'
from pipeline.paths import GOLD_DB_PATH
c = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
r = c.execute(\"SELECT robustness_status, COUNT(*) FROM edge_families WHERE instrument='M6E' GROUP BY robustness_status\").fetchall()
for row in r: print(row)
"
```

---

### Task 9: Copy scratch back and clean up

**Step 1: Copy back**
```bash
cp C:/db/gold.db gold.db
```

**Step 2: Verify canonical DB**
```bash
python -c "
unset DUCKDB_PATH  # or: export DUCKDB_PATH=''
import duckdb
c = duckdb.connect('gold.db', read_only=True)
print('M6E validated:', c.execute(\"SELECT COUNT(*) FROM validated_setups WHERE instrument='M6E'\").fetchone()[0])
print('Total validated:', c.execute('SELECT COUNT(*) FROM validated_setups').fetchone()[0])
"
```

**Step 3: Delete scratch**
```bash
rm C:/db/gold.db
unset DUCKDB_PATH
```

**Step 4: Verify path resolution**
```bash
python -c "from pipeline.paths import GOLD_DB_PATH; print('DB:', GOLD_DB_PATH)"
# Expect: gold.db in project root (no warning)
```

---

### Task 10: Run portfolio report and record findings

**Step 1: Full portfolio report**
```bash
python scripts/reports/report_edge_portfolio.py --all
```

**Step 2: Commit**
```bash
git add -p  # review changes
git commit -m "feat: ingest M6E (EUR/USD) 2019-2024 and run full ORB discovery pipeline"
```

---

## Expected Outcomes

| Instrument | Data | Mechanism | Expectation |
|------------|------|-----------|-------------|
| M6E LONDON_OPEN | 2019-2024 | London open = peak EUR/USD volatility, clean 5m ORB | High probability of validated strategies |
| M6E US_DATA_OPEN | 2019-2024 | NFP/CPI releases drive large EUR/USD moves | Possible, but high spread during news |
| M6E 1800 | 2019-2024 | Fixed London open approx (DST-contaminated) | Blended — less clean than LONDON_OPEN |
| M6E 0030 | 2019-2024 | Fixed US equity open approx | Lower confidence than dynamic sessions |

**If no M6E strategies validate:** ORB breakout may not suit EUR/USD price action. Next step: check session-by-session avgR distribution (not just pass/fail threshold).

## Notes
- 5 years of data (2019-2024) meets CORE threshold for annual gate
- M6E pip-scaled filters already set: M6E_G4 = 4 pips, M6E_G6 = 6 pips, M6E_G8 = 8 pips
- Cost model includes realistic spread (1.25 ticks each side) and slippage
- LONDON_OPEN session uses dynamic resolver (UK DST-aware) — cleanest FX session
