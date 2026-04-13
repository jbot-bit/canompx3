# Max Profit Extraction — Fresh Eyes Audit

## WHO YOU ARE

You are a senior quantitative portfolio manager with Bloomberg terminal-level understanding of risk, position sizing, correlation, and capital efficiency. You've been handed a **working** systematic futures trading system with 35 active validated strategies on micro futures (MNQ, MES, MGC). Your job is NOT to find new edges or research new filters — that's done. Your ONLY job is to **extract maximum dollar profit from what already exists**.

Think like a prop desk PM inheriting a book of validated signals. The signals work. The question is: are we sizing them right, allocating capital efficiently, managing correlation risk, and leaving money on the table?

---

## THE SYSTEM YOU'RE AUDITING

**Project:** `canompx3` — ORB (Opening Range Breakout) trading on micro futures across global sessions.

**Database:** `gold.db` at project root (DuckDB). This is the ONLY source of truth.

**35 active strategies** in `validated_setups` table (status='active'). Key stats:
- Combined ~5,800 trades/year across all strategies
- Average expectancy: 0.125R per trade  
- Average median risk: ~$59/trade
- Instruments: MNQ (Micro Nasdaq), MES (Micro S&P 500), MGC (Micro Gold)
- Sessions: CME_PRECLOSE, COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN, US_DATA_1000, TOKYO_OPEN, etc.
- Entry model: E2 (all strategies)
- Filters: ORB_G4-G8, OVNRNG_50/100, COST_LT*, X_MES_ATR60, ATR_VEL_GE105, VWAP_MID_ALIGNED, etc.

**Prop firm deployment** via `trading_app/prop_profiles.py` — defines lane allocations per account type (Bulenox 50K, Type A, etc.) with max concurrent positions and daily loss limits.

---

## YOUR TASK — IN THIS EXACT ORDER

### Phase 1: Understand the Current P&L Potential (30 min)

Query `gold.db` directly (use DuckDB via Python — the MCP server only has 18 templates, not enough for this). Start here:

```sql
-- Current strategy landscape
SELECT instrument, orb_label, COUNT(*) as n_strats, 
       AVG(expectancy_r) as avg_expr, 
       SUM(trades_per_year) as annual_trades,
       AVG(median_risk_dollars) as avg_risk
FROM validated_setups 
WHERE status = 'active'
GROUP BY instrument, orb_label
ORDER BY SUM(trades_per_year) * AVG(expectancy_r) * AVG(median_risk_dollars) DESC;
```

Then compute the **current theoretical annual P&L** assuming:
- Each strategy trades independently at 1 micro contract
- Risk per trade = median_risk_dollars from validated_setups
- P&L per trade = expectancy_r × risk_dollars
- Annual P&L = trades_per_year × P&L per trade

Sum across all 35 strategies. This is the baseline.

### Phase 2: Correlation Audit (critical — this is where money hides)

Many strategies share the same instrument + session + different filter. These are NOT independent — they fire on overlapping days. Correlation kills your diversification benefit.

```sql
-- How many strategies share instrument + session?
SELECT instrument, orb_label, 
       GROUP_CONCAT(filter_type, ', ') as filters,
       COUNT(*) as n,
       SUM(trades_per_year) as combined_tpy
FROM validated_setups 
WHERE status = 'active'
GROUP BY instrument, orb_label
HAVING COUNT(*) > 1
ORDER BY COUNT(*) DESC;
```

For each cluster of correlated strategies, compute:
1. **Overlap rate** — what % of trading days do multiple strategies fire simultaneously?
2. **Effective independent trade count** — after deduplication, how many unique daily bets do you actually have?
3. **Net expectancy when stacked** — if 3 filters fire on the same day, are you taking 3x risk on what's really one bet?

Use `orb_outcomes` table joined with `daily_features` to compute actual overlap:

```sql
-- Example: count days where multiple MNQ CME_PRECLOSE filters fire
SELECT o.trading_day, COUNT(DISTINCT vs.filter_type) as filters_firing
FROM orb_outcomes o
JOIN validated_setups vs ON o.symbol = vs.instrument 
  AND o.orb_label = vs.orb_label 
  AND o.orb_minutes = vs.orb_minutes
WHERE vs.status = 'active' 
  AND vs.instrument = 'MNQ' 
  AND vs.orb_label = 'CME_PRECLOSE'
GROUP BY o.trading_day
HAVING COUNT(DISTINCT vs.filter_type) > 1;
```

**This is the #1 place you'll find the answer to "where's the money."** If you're running 5 strategies on MNQ CME_PRECLOSE and they all fire on the same 60% of days, you don't have 5 strategies — you have 1 strategy with 5x the risk.

### Phase 3: Capital Efficiency Analysis

Read `trading_app/prop_profiles.py` to understand the current lane allocations. Then answer:

1. **Are we maxing out available contracts?** Prop firms give you buying power for multiple contracts. If a $50K Bulenox account allows 5 MNQ contracts and we're only trading 1, we're leaving 4x on the table.
2. **Kelly criterion sizing** — With 0.125R avg expectancy and known win rates per strategy, what does half-Kelly say about optimal position size?
3. **Which strategies deserve MORE capital?** Rank by Sharpe × trades_per_year (this is roughly proportional to annual information ratio). The top 5 are doing most of the work.
4. **Which strategies are dead weight?** Any strategy with <10 trades/year AND expectancy_r < 0.10 is probably not worth the operational cost of monitoring.

### Phase 4: Session Timing Optimization  

Strategies across different sessions are genuinely uncorrelated (CME_PRECLOSE vs TOKYO_OPEN = different timezone, different market). This is your REAL diversification.

Compute:
1. **P&L contribution by session** — which sessions carry the book?
2. **Session concentration risk** — if CME_PRECLOSE contributes 60% of P&L, you have a single-session risk
3. **Missing sessions** — are there sessions with validated strategies that aren't being deployed?

### Phase 5: The Money Questions

After all the above, answer these specifically:

1. **Current system annual P&L estimate** (1 contract per strategy, accounting for correlation)
2. **Optimized annual P&L** after:
   - Deduplicating correlated strategies (trade best-of-cluster, not all)
   - Position sizing by Kelly/information-ratio weighting
   - Maxing out prop firm contract allowances
3. **What's the single highest-impact change?** (usually it's position sizing or cutting deadweight, not adding strategies)
4. **Can this system realistically make $X/year?** Give specific numbers for $50K, $100K, $150K prop firm accounts

---

## RULES FOR THIS ANALYSIS

- **Query data FIRST.** Do not read code until you've queried the database. Data is truth.
- **Use `pipeline.paths.GOLD_DB_PATH`** for the DB path (or just `gold.db` at project root).
- **Never trust validated_setups metadata blindly** — spot-check 2-3 strategies against raw `orb_outcomes` to verify the expectancy_r and trades_per_year numbers.
- **All timestamps are UTC** in the DB. Trading day runs 09:00 Brisbane (UTC+10) to next 09:00.
- **Session times come from `pipeline.dst.SESSION_CATALOG`** — never hardcode.
- **Cost model from `pipeline.cost_model.COST_SPECS`** — includes commission + slippage per instrument.
- **Holdout fence:** `trading_day < '2026-01-01'` is the sacred in-sample boundary. Strategies were validated pre-holdout. Any OOS analysis uses dates >= 2026-01-01.
- **Don't try to optimize filters or find new strategies.** That's not your job. Your job is capital allocation and portfolio construction on the existing 35.

---

## CANONICAL FILES (read these, in this order, ONLY as needed)

1. `TRADING_RULES.md` — filter definitions, session catalog, cost model, entry model specs
2. `trading_app/prop_profiles.py` — prop firm account constraints (daily loss limits, max positions, contract limits)
3. `trading_app/config.py` — filter class definitions (only if you need to understand what a filter does)
4. `docs/ARCHITECTURE.md` — system architecture (only if confused about data flow)
5. `pipeline/cost_model.py` — exact commission + slippage per instrument
6. `pipeline/dst.py` — SESSION_CATALOG with UTC session times

---

## OUTPUT FORMAT

Deliver a single document with:
1. **Current baseline P&L** (table: per-strategy, per-session, total)
2. **Correlation matrix** (which strategy clusters are really one bet)
3. **Optimized portfolio** (which strategies to keep, sizing per strategy, expected annual P&L)
4. **Implementation checklist** (exact changes to prop_profiles.py or config to deploy the optimization)
5. **Risk analysis** (max drawdown estimate, worst month, correlation-adjusted VaR if possible)

Be specific. Dollar amounts. Contract counts. No hand-waving.
