# MCP Server Usage (gold-db)

## Golden Rule
**ALWAYS prefer MCP tools over raw SQL against gold.db.** MCP tools apply correct filters, enforce read-only access, and handle edge cases. Writing ad-hoc SQL risks missing filter application (e.g., querying `orb_outcomes` without `filter_type` from `daily_features`).

## Decision Framework

When the user asks about trading data, match their intent to the right tool:

### Strategy Performance / Regime Health
- "How is strategy X performing?" / "Is it still working?" / "Any strategies decaying?"
  → `get_strategy_fitness(strategy_id="X")` or `get_strategy_fitness(summary_only=True)` for all
- "Recent 12/18 month rolling performance"
  → `get_strategy_fitness(rolling_months=18, summary_only=True)`
- **WARNING:** Never call `get_strategy_fitness()` for all strategies WITHOUT `summary_only=True` — output exceeds 150K chars and blows up context.

### Strategy Lookups & Comparisons
- "Show me all strategies for session CME_REOPEN" / "What's validated for MNQ?"
  → `query_trading_db(template="validated_summary", orb_label="CME_REOPEN")` or filter by instrument
- "Compare E0 vs E1" / "Which entry model is best?"
  → `query_trading_db(template="performance_stats", ...)` with varying `entry_model`
- "Full details on strategy X"
  → `query_trading_db(template="strategy_lookup", ...)` with matching params

### Raw Data & Infrastructure
- "How many rows in each table?" / "What's the schema?"
  → `query_trading_db(template="table_counts")` or `query_trading_db(template="schema_info")`
- "Show me trade history for a strategy"
  → `query_trading_db(template="trade_history", ...)`
- "ORB size distribution" / "Gap analysis"
  → `query_trading_db(template="orb_size_dist")` or `query_trading_db(template="gap_analysis")`

### Context Loading
- Before complex multi-step analysis or when unsure about trading rules
  → `get_canonical_context()` to load cost model, logic rules, and config

### Discovery
- "What queries can I run?" / Not sure which template to use
  → `list_available_queries()` first

## NEVER Do This
- Write raw SQL against `gold.db` when a template covers the query
- Query `orb_outcomes` directly without joining `daily_features` for filter application
- Return full portfolio fitness without `summary_only=True`
- Assume filter names — always use template enum values from `list_available_queries()`
