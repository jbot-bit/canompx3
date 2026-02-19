# Prompt: Execute DST Pipeline Remediation

## What was changed (code already modified, ready to run)

The following files have been modified to add DST regime awareness to the pipeline:

### 1. `pipeline/dst.py` — New helper functions
- `DST_AFFECTED_SESSIONS` dict: maps session → "US" or "UK" DST type
- `DST_CLEAN_SESSIONS` set: sessions not affected
- `is_winter_for_session(trading_day, orb_label)` → True/False/None
- `classify_dst_verdict(winter_avg_r, summer_avg_r, winter_n, summer_n)` → verdict string

### 2. `trading_app/db_manager.py` — New columns
Added to both `experimental_strategies` and `validated_setups`:
```sql
dst_winter_n      INTEGER,
dst_winter_avg_r  DOUBLE,
dst_summer_n      INTEGER,
dst_summer_avg_r  DOUBLE,
dst_verdict       TEXT,
```

### 3. `trading_app/strategy_discovery.py` — DST split in discovery
- Imports DST helpers from `pipeline/dst.py`
- New `compute_dst_split_from_outcomes()` function computes winter/summer from in-memory outcomes
- Called for every strategy in grid iteration
- DST columns included in INSERT SQL

### 4. `trading_app/strategy_validator.py` — DST split in validation
- Imports DST helpers from `pipeline/dst.py`
- New `compute_dst_split()` function queries orb_outcomes and splits by DST
- Called for every strategy during validation (after walk-forward)
- DST info logged to console for affected sessions
- DST columns included in both experimental_strategies UPDATE and validated_setups INSERT

### 5. `research/research_volume_dst_analysis.py` — NEW research script
- Task 1: Volume comparison at session times by DST regime
- Task 2: New session candidate evaluation (09:30, 19:00, 10:45)

## Execution Steps

### Step 1: Schema migration (add DST columns to existing tables)

**NOTE:** `init_trading_app_schema()` now includes automatic migration for DST columns. Steps 2 & 3 will call this automatically. The manual migration below is a fallback if you want to add columns without re-running the full pipeline:


```bash
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db')

# Add DST columns to experimental_strategies
for col, typ in [('dst_winter_n', 'INTEGER'), ('dst_winter_avg_r', 'DOUBLE'),
                  ('dst_summer_n', 'INTEGER'), ('dst_summer_avg_r', 'DOUBLE'),
                  ('dst_verdict', 'TEXT')]:
    try:
        con.execute(f'ALTER TABLE experimental_strategies ADD COLUMN {col} {typ}')
        print(f'Added {col} to experimental_strategies')
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f'{col} already exists in experimental_strategies')
        else:
            raise

# Add DST columns to validated_setups
for col, typ in [('dst_winter_n', 'INTEGER'), ('dst_winter_avg_r', 'DOUBLE'),
                  ('dst_summer_n', 'INTEGER'), ('dst_summer_avg_r', 'DOUBLE'),
                  ('dst_verdict', 'TEXT')]:
    try:
        con.execute(f'ALTER TABLE validated_setups ADD COLUMN {col} {typ}')
        print(f'Added {col} to validated_setups')
    except Exception as e:
        if 'already exists' in str(e).lower():
            print(f'{col} already exists in validated_setups')
        else:
            raise

con.close()
print('Schema migration complete.')
"
```

### Step 2: Re-run discovery for all instruments

This will recompute all experimental_strategies WITH DST columns populated:

```bash
# Clear existing validation status so validator re-processes all
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db')
con.execute(\"UPDATE experimental_strategies SET validation_status = NULL, validation_notes = NULL\")
con.commit()
print('Reset validation status for all strategies')
con.close()
"

# Re-run discovery (this overwrites experimental_strategies with DST columns)
python trading_app/strategy_discovery.py --instrument MGC --db C:/db/gold.db
python trading_app/strategy_discovery.py --instrument MNQ --db C:/db/gold.db
python trading_app/strategy_discovery.py --instrument MES --db C:/db/gold.db
```

### Step 3: Re-run validation for all instruments

This will re-validate all strategies, now with DST split visible:

```bash
# Clear validated_setups so they get re-promoted with DST columns
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db')
con.execute('DELETE FROM validated_setups')
con.commit()
print('Cleared validated_setups for rebuild')
con.close()
"

python trading_app/strategy_validator.py --instrument MGC --db C:/db/gold.db
python trading_app/strategy_validator.py --instrument MNQ --db C:/db/gold.db
python trading_app/strategy_validator.py --instrument MES --db C:/db/gold.db
```

### Step 4: Verify DST columns are populated

```bash
python -c "
import duckdb
con = duckdb.connect('C:/db/gold.db', read_only=True)

# Check experimental_strategies
r = con.execute('''
    SELECT dst_verdict, COUNT(*) as cnt
    FROM experimental_strategies
    WHERE dst_verdict IS NOT NULL
    GROUP BY dst_verdict
    ORDER BY cnt DESC
''').fetchall()
print('Experimental strategies DST verdicts:')
for row in r:
    print(f'  {row[0]}: {row[1]}')

# Check validated_setups
r = con.execute('''
    SELECT dst_verdict, COUNT(*) as cnt
    FROM validated_setups
    WHERE dst_verdict IS NOT NULL
    GROUP BY dst_verdict
    ORDER BY cnt DESC
''').fetchall()
print('\\nValidated setups DST verdicts:')
for row in r:
    print(f'  {row[0]}: {row[1]}')

# Show red flags (edge dies in one regime)
r = con.execute('''
    SELECT strategy_id, dst_winter_avg_r, dst_winter_n, dst_summer_avg_r, dst_summer_n, dst_verdict
    FROM validated_setups
    WHERE dst_verdict IN ('WINTER-ONLY', 'SUMMER-ONLY')
''').fetchall()
if r:
    print('\\n⚠️ RED FLAGS (validated strategies with edge dying in one regime):')
    for row in r:
        print(f'  {row[0]}: W={row[1]:+.3f}({row[2]}) S={row[3]:+.3f}({row[4]}) → {row[5]}')
else:
    print('\\n✅ No red flags in validated strategies')

con.close()
"
```

### Step 5: Run volume analysis (parallel, optional)

```bash
python research/research_volume_dst_analysis.py --db-path C:/db/gold.db
```

### Step 6: Re-build edge families (downstream of validated_setups)

```bash
python scripts/tools/build_edge_families.py --db C:/db/gold.db
```

## Expected Results

- `experimental_strategies`: ~6,480 rows per instrument, all with DST columns populated
- `validated_setups`: same count as before (DST is INFO only, not a rejection gate)
- Every strategy at 0900/1800/0030/2300 has winter_n, summer_n, winter_avg_r, summer_avg_r, verdict
- Every strategy at 1000/1100/1130 has verdict = "CLEAN"
- Dynamic sessions have verdict = "CLEAN"

## Validation Checks

1. Combined N should equal winter_n + summer_n for affected sessions
2. Strategies at clean sessions should have dst_verdict = "CLEAN"
3. The validated strategy count should be UNCHANGED (DST is not a rejection gate)
4. DST verdicts should match the research revalidation results in `research/output/dst_strategy_revalidation.csv`
