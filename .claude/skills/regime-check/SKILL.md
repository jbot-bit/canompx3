---
name: regime-check
description: Check portfolio fitness and regime health across all instruments
allowed-tools: Read, Grep, Glob, Bash
---
Check portfolio fitness and regime health: $ARGUMENTS

Use when: "fitness", "regime", "decay", "how's the portfolio", "strategy health"

## Step 1: Query Fitness

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

print('=== REGIME SUMMARY ===')
print(con.sql('''
    SELECT vs.instrument, COALESCE(ef.robustness_status, 'NO_FAMILY') as regime_status,
           COALESCE(ef.trade_tier, 'NONE') as tier, COUNT(*) as count
    FROM validated_setups vs
    LEFT JOIN edge_families ef ON vs.strategy_id = ef.head_strategy_id
    WHERE LOWER(vs.status) = 'active'
    GROUP BY vs.instrument, regime_status, tier ORDER BY vs.instrument, regime_status
''').fetchdf().to_string(index=False))

print('\n=== EDGE FAMILIES ===')
print(con.sql('''
    SELECT instrument, trade_tier, COUNT(*) as families,
           ROUND(AVG(head_expectancy_r), 4) as avg_expr, MIN(min_member_trades) as min_trades
    FROM edge_families GROUP BY instrument, trade_tier ORDER BY instrument, trade_tier
''').fetchdf().to_string(index=False))

last_built = con.sql('SELECT MAX(created_at) FROM edge_families').fetchone()[0]
print(f'\nEdge families last built: {last_built}')
import datetime
if last_built:
    age_days = (datetime.datetime.now(datetime.timezone.utc) - last_built.replace(tzinfo=datetime.timezone.utc)).days if hasattr(last_built, 'replace') else None
    if age_days and age_days > 30:
        print(f'  WARNING: {age_days} days old — STALE')
    elif age_days is not None:
        print(f'  ({age_days} days old — FRESH)')

print('\n=== INSTRUMENT HEALTH SUMMARY ===')
print(con.sql('''
    SELECT instrument,
           SUM(CASE WHEN trade_tier='CORE' THEN 1 ELSE 0 END) as core_families,
           SUM(CASE WHEN trade_tier='REGIME' THEN 1 ELSE 0 END) as regime_families,
           COUNT(*) as total_families,
           ROUND(AVG(head_expectancy_r), 4) as avg_expr
    FROM edge_families GROUP BY instrument ORDER BY instrument
''').fetchdf().to_string(index=False))

print('\n=== ORB SIZE TREND (MNQ O5, 6mo windows) ===')
print(con.sql('''
    SELECT CASE WHEN trading_day >= CURRENT_DATE - 180 THEN 'recent' ELSE 'prior' END as period,
           ROUND(AVG(orb_CME_PRECLOSE_size), 2) as cme_pre, ROUND(AVG(orb_NYSE_OPEN_size), 2) as nyse
    FROM daily_features WHERE symbol = 'MNQ' AND orb_minutes = 5 AND trading_day >= CURRENT_DATE - 360
    GROUP BY period ORDER BY period
''').fetchdf().to_string(index=False))
con.close()
"
```

## Step 2: Flags

- 0 CORE families for any instrument → RED
- Edge families > 30 days old → STALE
- ORB sizes trending down → edge weakening

## Step 3: Present

One-liner per instrument: `MGC: X CORE, Y REGIME [HEALTHY/CONCERN/CRITICAL]`

## Rules

- NEVER cite counts from memory — always query fresh
- Column is `instrument` not `symbol` in validated_setups
- Fitness is in `edge_families` (robustness_status, trade_tier)
