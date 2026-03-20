---
name: regime-check
description: Check portfolio fitness and regime health across all instruments
allowed-tools: Read, Grep, Glob, Bash
---
Check portfolio fitness and regime health across all instruments: $ARGUMENTS

Use when: "fitness", "regime", "any decay", "how's the portfolio", "health of strategies", "regime check", "strategy health", "are strategies still working"

## Regime Health Check

Quick portfolio fitness snapshot. Shows regime distribution, flags transitions, and checks assumptions.

### Step 0: Blueprint Context

Check `docs/STRATEGY_BLUEPRINT.md §10` — "What We Might Be Wrong About." Flag any assumption that affects this check:
- Are ORB sizes shrinking? (edge dying signal)
- Is the cost model still accurate?
- Have any sessions changed behavior (exchange schedule changes)?

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

### Step 3: Assumption Health Check

Also query for early warning signs from Blueprint §10:

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# ORB size trend — are ORBs shrinking? (if yes, G4+ filter qualifies fewer days = edge dying)
print('=== ORB SIZE TREND (MNQ O5, last 6 months vs prior 6 months) ===')
print(con.sql('''
    SELECT CASE WHEN trading_day >= CURRENT_DATE - 180 THEN 'recent_6mo' ELSE 'prior_6mo' END as period,
           ROUND(AVG(orb_CME_PRECLOSE_size), 2) as avg_cme_pre,
           ROUND(AVG(orb_NYSE_OPEN_size), 2) as avg_nyse
    FROM daily_features WHERE symbol = 'MNQ' AND orb_minutes = 5
      AND trading_day >= CURRENT_DATE - 360
    GROUP BY period ORDER BY period
''').fetchdf().to_string(index=False))
# ML model age
import os
model_path = 'models/ml/meta_label_MNQ_hybrid.joblib'
if os.path.exists(model_path):
    import joblib
    b = joblib.load(model_path)
    print(f'\nML model trained: {b.get(\"trained_at\", \"unknown\")}')
    print(f'ML model RR lock: {b.get(\"rr_target_lock\", \"unknown\")}')
con.close()
"
```

Flag if:
- ORB sizes trending down significantly → edge may be weakening
- ML model older than 30 days → consider retraining
- validated_setups count changed → pipeline rebuild may be needed

### Rules

- NEVER cite counts from memory — always query fresh
- One query, one table, one summary. Keep it tight.
- Fitness lives in edge_families (robustness_status, trade_tier) — NOT a strategy_fitness table
- Column is `instrument` not `symbol` in validated_setups
- Check Blueprint §10 assumptions — report any early warning signs
