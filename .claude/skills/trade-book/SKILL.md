---
name: trade-book
description: Show current trading book with full strategy details
allowed-tools: Read, Grep, Glob, Bash
---
Show current trading book with full strategy details: $ARGUMENTS

Use when: "what do I trade", "what's live", "show strategies", "what's at [session]", "trading book", "portfolio", "show me what's validated", "what's FIT", "live strategies", "what should I trade", "tonight", "playbook", "what sessions", "session times"

## Step 1: Generate the Trade Sheet

ALWAYS run the trade sheet generator first. It resolves session times from `pipeline/dst.py` (never guess timezone math), applies dollar gates, checks fitness, and outputs a self-contained HTML.

```bash
python scripts/tools/generate_trade_sheet.py
```

This opens in the browser automatically. The terminal output shows correct Brisbane session times.

Optional flags:
- `--date 2026-03-10` — specific trading day
- `--no-open` — don't open browser
- `--output path.html` — custom output path

## Step 2: Answer the User's Question

Use the terminal output from the generator to answer. The generator already:
- Resolves DST-correct Brisbane times via `pipeline.dst.SESSION_CATALOG` resolvers
- Filters to live portfolio specs from `trading_app.live_config.LIVE_PORTFOLIO`
- Applies ExpR gate (`LIVE_MIN_EXPECTANCY_R`) and dollar gate
- Checks fitness per strategy via `trading_app.strategy_fitness` module
- Shows only cost-positive, gate-passing trades

If the user asked about a specific session or instrument, highlight those from the output.

## Step 3: If User Wants Raw Data

For deeper queries (specific strategy IDs, historical performance, etc.), query gold.db directly:

```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# Use correct column names:
#   instrument (not symbol), orb_label (not session_name),
#   expectancy_r (not avg_r), sharpe_ann (not sharpe)
# Fitness is in edge_families.robustness_status (not strategy_fitness table)
df = con.execute('''
    SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
           vs.entry_model, vs.confirm_bars, vs.filter_type, vs.rr_target,
           vs.stop_multiplier, vs.sample_size, vs.win_rate, vs.expectancy_r,
           vs.sharpe_ann, vs.all_years_positive, vs.years_tested,
           ef.robustness_status, ef.trade_tier
    FROM validated_setups vs
    LEFT JOIN edge_families ef ON vs.strategy_id = ef.head_strategy_id
    WHERE LOWER(vs.status) = 'active'
    ORDER BY vs.orb_label, vs.instrument, vs.expectancy_r DESC
''').fetchdf()
con.close()
print(df.to_string(index=False))
"
```

## Rules

- ALWAYS run generate_trade_sheet.py FIRST — never hand-compute session times
- NEVER guess timezone offsets — the resolvers handle DST automatically
- ALWAYS include rr_target — user explicitly demanded this
- NEVER use MCP for trade book queries — too slow and may be stale
- NEVER cite strategy counts from memory — always query fresh
- NEVER reference strategy_fitness table — it does not exist. Use edge_families
- Correct column names: instrument, orb_label, expectancy_r, sharpe_ann (not symbol, session_name, avg_r, sharpe)
