Check portfolio fitness and regime health across all instruments: $ARGUMENTS

Use when: "fitness", "regime", "any decay", "how's the portfolio", "health of strategies", "regime check", "strategy health", "are strategies still working"

## Regime Health Check

Quick portfolio fitness snapshot. Shows regime distribution and flags transitions.

### Step 1: Query Current Fitness

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

# Summary by instrument and regime
summary = con.execute('''
    SELECT
        v.symbol,
        COALESCE(f.fitness_regime, 'NO_FITNESS') as regime,
        COUNT(*) as count
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    GROUP BY v.symbol, regime
    ORDER BY v.symbol, regime
''').fetchdf()

print('=== REGIME SUMMARY ===')
print(summary.to_string(index=False))

# Total counts
totals = con.execute('''
    SELECT
        COALESCE(f.fitness_regime, 'NO_FITNESS') as regime,
        COUNT(*) as count
    FROM validated_setups v
    LEFT JOIN (
        SELECT strategy_id, fitness_regime
        FROM strategy_fitness
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM strategy_fitness)
    ) f ON v.strategy_id = f.strategy_id
    GROUP BY regime
    ORDER BY regime
''').fetchdf()

print()
print('=== PORTFOLIO TOTALS ===')
print(totals.to_string(index=False))

# Data freshness
freshness = con.execute('''
    SELECT MAX(as_of_date) as latest, MIN(as_of_date) as earliest
    FROM strategy_fitness
''').fetchdf()
print(f'\\nFitness data: {freshness.iloc[0][\"earliest\"]} to {freshness.iloc[0][\"latest\"]}')

con.close()
"
```

### Step 2: Flag Concerns

- Any instrument with 0 FIT strategies? -> RED FLAG
- Any instrument with > 50% DECAY/UNFIT? -> YELLOW FLAG
- Data freshness > 30 days old? -> STALE WARNING
- Significant shift from previous check? -> Note the transition

### Step 3: Present

One-liner per instrument, not a wall of text:

```
=== REGIME CHECK ===
MGC:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
MNQ:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
MES:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
M2K:  X FIT, Y WATCH, Z DECAY  [HEALTHY/CONCERN/CRITICAL]
Portfolio: N total, X% FIT
Data as of: YYYY-MM-DD
====================
```

### Rules

- NEVER cite counts from memory -- always query fresh
- One query, one table, one summary. Keep it tight.
- If strategy_fitness table is empty, say so clearly -- don't show zeros as if they're real.
