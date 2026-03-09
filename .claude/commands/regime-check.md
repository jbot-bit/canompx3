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

# Summary by instrument and robustness status
summary = con.execute('''
    SELECT
        vs.instrument,
        COALESCE(ef.robustness_status, 'NO_FAMILY') as status,
        COALESCE(ef.trade_tier, 'NONE') as tier,
        COUNT(*) as count
    FROM validated_setups vs
    LEFT JOIN edge_families ef ON vs.strategy_id = ef.head_strategy_id
    WHERE LOWER(vs.status) = 'active'
    GROUP BY vs.instrument, status, tier
    ORDER BY vs.instrument, status
''').fetchdf()

print('=== REGIME SUMMARY ===')
print(summary.to_string(index=False))

# Total counts
totals = con.execute('''
    SELECT
        COALESCE(ef.robustness_status, 'NO_FAMILY') as status,
        COUNT(*) as count
    FROM validated_setups vs
    LEFT JOIN edge_families ef ON vs.strategy_id = ef.head_strategy_id
    WHERE LOWER(vs.status) = 'active'
    GROUP BY status
    ORDER BY status
''').fetchdf()

print()
print('=== PORTFOLIO TOTALS ===')
print(totals.to_string(index=False))

# Edge family summary
families = con.execute('''
    SELECT instrument, trade_tier, COUNT(*) as families,
           AVG(head_expectancy_r) as avg_expr,
           MIN(min_member_trades) as min_trades
    FROM edge_families
    GROUP BY instrument, trade_tier
    ORDER BY instrument, trade_tier
''').fetchdf()

print()
print('=== EDGE FAMILIES ===')
print(families.to_string(index=False))

# Data freshness
freshness = con.execute('''
    SELECT MAX(created_at) as latest FROM edge_families
''').fetchdf()
print(f'\nEdge families last built: {freshness.iloc[0][\"latest\"]}')

con.close()
"
```

### Step 2: Flag Concerns

- Any instrument with 0 CORE families? -> RED FLAG
- Any instrument losing families vs last check? -> YELLOW FLAG
- Edge families older than 30 days? -> STALE WARNING
- REGIME-tier families without fitness gate? -> Note

### Step 3: Present

One-liner per instrument, not a wall of text:

```
=== REGIME CHECK ===
MGC:  X CORE, Y REGIME families  [HEALTHY/CONCERN/CRITICAL]
MNQ:  X CORE, Y REGIME families  [HEALTHY/CONCERN/CRITICAL]
MES:  X CORE, Y REGIME families  [HEALTHY/CONCERN/CRITICAL]
M2K:  X CORE, Y REGIME families  [HEALTHY/CONCERN/CRITICAL]
Portfolio: N total families, X validated strategies
Data as of: YYYY-MM-DD
====================
```

### Rules

- NEVER cite counts from memory -- always query fresh
- One query, one table, one summary. Keep it tight.
- Fitness lives in edge_families (robustness_status, trade_tier) -- NOT a strategy_fitness table
- Column is `instrument` not `symbol` in validated_setups
