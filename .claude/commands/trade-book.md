Show current trading book with full strategy details: $ARGUMENTS

Use when: "what do I trade", "what's live", "show strategies", "what's at [session]", "trading book", "portfolio", "show me what's validated", "what's FIT", "live strategies", "what should I trade"

## Trade Book Query

Fast direct query against gold.db. Skips MCP for speed.

### Step 1: Parse Arguments

- If $ARGUMENTS contains a session name (e.g., "CME_REOPEN", "TOKYO_OPEN"): filter by that session
- If $ARGUMENTS contains an instrument (e.g., "MGC", "MNQ"): filter by that instrument
- If $ARGUMENTS contains "all" or is empty: show everything FIT

### Step 2: Query gold.db Directly

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
df = con.execute('''
    SELECT
        v.symbol,
        v.orb_label,
        v.orb_minutes,
        v.entry_model,
        v.confirm_bars,
        v.filter_type,
        v.rr_target,
        v.direction,
        v.sample_size,
        v.win_rate,
        v.avg_r AS ExpR,
        v.sharpe,
        v.all_years_positive,
        v.years_tested,
        COALESCE(f.fitness_regime, 'UNKNOWN') as fitness
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    WHERE 1=1
    ORDER BY v.symbol, v.orb_label, v.orb_minutes, v.rr_target
''').fetchdf()
con.close()

print(df.to_string(index=False))
print(f'\\nTotal: {len(df)} strategies')
"
```

Modify the WHERE clause based on parsed arguments:
- Session filter: `AND v.orb_label = '{session}'`
- Instrument filter: `AND v.symbol = '{instrument}'`
- FIT only (default unless user asks for all): `AND COALESCE(f.fitness_regime, 'UNKNOWN') = 'FIT'`

### Step 3: Present Results

Format as a clean table grouped by session (Brisbane time order).

**For each strategy show ALL of these (MANDATORY -- never omit any):**
- Symbol, orb_label, orb_minutes (5/15/30)
- entry_model, confirm_bars
- filter_type, rr_target
- direction
- sample_size, win_rate, ExpR, Sharpe
- fitness_regime
- all_years_positive, years_tested

### Step 4: Summary

- Count by instrument
- Count by fitness regime (FIT/WATCH/DECAY/UNFIT)
- Flag any strategies with `all_years_positive = False`
- Note data freshness (latest as_of_date from strategy_fitness)

### Rules

- ALWAYS include rr_target -- user explicitly demanded this
- NEVER use MCP for this query -- too slow and may be stale
- NEVER cite strategy counts from memory -- always query fresh
- Show WATCH strategies dimmed (mention but flag as "monitor only")
- Hide UNFIT/DECAY unless user asks for them
- If query returns 0 rows, check if gold.db exists and has data
