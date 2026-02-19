# Prompt: DST Winter/Summer Split for Strategy Validation

## Context

We discovered that four fixed sessions (0900, 1800, 0030, 2300) blend two different market contexts because they align with specific market events in one DST regime but miss by 1 hour in the other. Every validated strategy at these sessions has contaminated stats — the avgR, Sharpe, WR, and totR are blended winter+summer averages that may hide regime-dependent edges.

Sessions 1000, 1100, 1130 are CLEAN (Asia has no DST). Dynamic sessions are CLEAN (resolvers adjust per-day).

## Full context on the DST problem

Read `CLAUDE.md` section "CRITICAL: DST Contamination in Fixed Sessions" for the complete table of which sessions are affected and why.

**Key facts:**
- Brisbane = UTC+10, NO DST (Queensland)
- US DST (EDT): roughly 2nd Sunday March → 1st Sunday November. EDT = UTC-4, EST = UTC-5.
- UK DST (BST): roughly last Sunday March → last Sunday October. BST = UTC+1, GMT = UTC+0.
- In winter: 0900 Bris = CME open, 1800 Bris = London open, 0030 Bris = US equity open
- In summer: all three are 1 hour AFTER the actual event

## Task

Build `research/research_dst_strategy_revalidation.py` that re-validates ALL existing validated strategies at affected sessions with a winter/summer split.

## Database

- Path: `C:/db/gold.db` (DuckDB)
- Tables needed:
  - `validated_setups` — all validated strategies with their parameters
  - `orb_outcomes` — pre-computed trade outcomes (689K rows)
  - `daily_features` — ORB data per trading day per session
- Instruments: MGC, MNQ, MES (MCL has no validated edges)

## Method

### Step 1: Identify affected strategies
- Query `validated_setups` for all strategies where `orb_label IN ('0900', '1800', '0030', '2300')`
- Also include `experimental_strategies` at those sessions with positive avgR (they might have been rejected due to blended stats but could pass in one regime)

### Step 2: Classify each trading day's DST regime
- For sessions affected by US DST (0900, 0030, 2300): use US Eastern timezone
  - Winter (EST) = `utcoffset == -5 hours` on that date
  - Summer (EDT) = `utcoffset == -4 hours` on that date
- For session 1800: use UK London timezone
  - Winter (GMT) = `utcoffset == 0 hours` on that date
  - Summer (BST) = `utcoffset == +1 hour` on that date
- Use `zoneinfo` (stdlib) — same approach as `pipeline/dst.py`

### Step 3: Re-compute metrics per strategy, split by regime
For each strategy, split its trade days into winter and summer, then compute separately:
- `n_trades` (winter / summer / combined)
- `avg_r` (winter / summer / combined)
- `total_r` (winter / summer / combined)
- `win_rate` (winter / summer / combined)
- `sharpe` (winter / summer / combined) — use `mean(r) / std(r)` if N >= 10
- `max_dd` (winter / summer / combined)

### Step 4: Classify DST stability
For each strategy, assign a DST stability verdict:
- **STABLE**: |winter_avgR - summer_avgR| <= 0.10R AND both have N >= 15
- **WINTER-DOMINANT**: winter_avgR > summer_avgR + 0.10R AND winter N >= 15
- **SUMMER-DOMINANT**: summer_avgR > winter_avgR + 0.10R AND summer N >= 15
- **WINTER-ONLY**: winter_avgR > 0 AND summer_avgR <= 0 AND both N >= 10
- **SUMMER-ONLY**: summer_avgR > 0 AND winter_avgR <= 0 AND both N >= 10
- **LOW-N**: either regime has fewer than 10 trades

### Step 5: Output

**Console output:**
1. Summary table: count of strategies per verdict (STABLE / WINTER-DOM / SUMMER-DOM / WINTER-ONLY / SUMMER-ONLY / LOW-N)
2. Per-strategy detail table sorted by DST instability (largest |winter - summer| first):
   ```
   Strategy ID | Session | Combined avgR | Winter avgR (N) | Summer avgR (N) | Verdict
   ```
3. **RED FLAGS section:** Any strategy where edge DIES (positive → negative) in one regime
4. **RECOMMENDATION section:** For each RED FLAG strategy, suggest whether to:
   - Switch to the corresponding dynamic session year-round
   - Keep fixed but only trade in the good regime
   - Investigate further

**Save to CSV:** `research/output/dst_strategy_revalidation.csv` with all columns

### Step 6: Also check experimental (non-validated) strategies
- Query `experimental_strategies` at affected sessions where `avg_r > 0` but strategy didn't make it to `validated_setups`
- Check if splitting by DST reveals a WINTER-ONLY or SUMMER-ONLY strategy that WOULD validate in that regime (i.e., the blended average killed a good seasonal edge)
- Report these as "Hidden edges revealed by DST split"

## Important Notes

- Use DuckDB Python API (`import duckdb`)
- Use `zoneinfo` for DST detection (same as `pipeline/dst.py`)
- This is a READ-ONLY research script — no writes to gold.db
- Print progress — this may take a while if there are many strategies
- The orb_outcomes table has columns: `trading_day`, `symbol`, `orb_label`, `entry_model`, `confirm_bars`, `rr_target`, `filter_type`, `pnl_r` (and more). Join on these to get per-trade R-multiples.
- Strategy IDs follow pattern: `{SYMBOL}_{SESSION}_{ENTRY}_{RR}_{CB}_{FILTER}` e.g. `MGC_0900_E1_RR2.5_CB2_ORB_G4`

## Validation

1. Strategies at 1000/1100/1130 should NOT appear (clean sessions)
2. The combined avgR should match the existing validated_setups avgR (within rounding)
3. Winter N + Summer N should equal Combined N
4. Any strategy marked STABLE should have similar Sharpe in both regimes
