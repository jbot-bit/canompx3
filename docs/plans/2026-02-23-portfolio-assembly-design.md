# Portfolio Assembly Research Design

**Goal:** Build a combined equity curve from all 21 session slots across 4 instruments (MGC, MNQ, MES, M2K) to see honest portfolio-level performance before going live.

**Approach:** Research-first — a read-only script that queries existing data, computes combined stats, and produces a comprehensive report. No DB writes, no pipeline changes, no optimization.

---

## Scope

- **Instruments:** MGC, MNQ, MES, M2K (all 4)
- **Slots:** 21 session slots from `session_slots()` (one representative per instrument+session)
- **Sizing:** Equal R weight (1R per trade regardless of instrument/session)
- **Outcomes:** Win/loss only (exclude scratches/early exits)
- **Date ranges:** Report both full-history (each instrument contributes available data) AND common-period (all instruments present)

## Output Sections

### 1. Slot Inventory
All 21 session slots with: head strategy ID, instrument, session, ExpR, Sharpe, N, trade tier.

### 2. Combined Daily Equity Curve
Sum of all slot R-returns per calendar day. Days with no trades = 0R.

**Critical: Sharpe denominator includes ALL trading days** (weekdays in date range), not just days with trades. Otherwise Sharpe is inflated by idle days.

### 3. Portfolio Metrics
- Combined annualized Sharpe (honest, including zero-days)
- Max drawdown (R)
- Total R
- Overall win rate
- Avg trades per trading day

### 4. Per-Year Breakdown
Year-by-year: trades, WR, total R, annual Sharpe. Shows stability and whether performance is front-loaded or consistent.

### 5. Slot Correlation Matrix
Pairwise correlation between slots' daily returns. Rules:
- Only show pairs with >= 30 overlapping trade days
- Report overlap N alongside correlation coefficient
- Flag highly correlated pairs (r > 0.3) — indicates concentrated risk

### 6. Concurrent Exposure Analysis
Distribution of how many slots fire per day: 0-slot days, 1-slot, 2-slot, 3+.
- Max single-day R exposure (worst case: all losing)
- Mean slots per active day
- Capital requirement implication

### 7. Per-Instrument Contribution
How much of total R comes from each instrument. Identifies concentration risk.

### 8. Drawdown Analysis
- Maximum drawdown magnitude (R) and duration (days)
- Longest losing streak (consecutive negative days)
- Worst single day (max daily loss in R)
- Recovery time from max DD

---

## Data Flow

```
session_slots(db_path) → 21 head strategy IDs
  → _load_head_trades(con, instrument) for each instrument
  → filter to slot heads only
  → build daily ledger per slot
  → combine into portfolio daily returns (sum per calendar day)
  → compute all stats
```

Reuses functions from `report_edge_portfolio.py`:
- `session_slots()` — slot election
- `_load_head_trades()` — trade loading (adapted to filter by strategy)
- `_compute_portfolio_stats()` — daily Sharpe, max DD

### Date Range Handling

MGC: ~2021-present, MNQ: ~2024-present, MES: varies, M2K: varies.

Report TWO portfolio views:
1. **Full history:** Each instrument contributes from its earliest data. Portfolio grows as instruments come online.
2. **Common period:** Only dates where ALL included instruments have data. Honest combined view.

---

## What This Does NOT Do

- No position sizing optimization (equal weight only)
- No slot selection/dropping (all 21 included)
- No DB writes or schema changes
- No pipeline modifications
- No multi-asset per slot (future consideration)

## Success Criteria

The report answers: "If I traded all 21 slots at equal 1R risk starting tomorrow, what would the historical performance look like?" — with honest Sharpe, real drawdowns, and no cherry-picking.

## Implementation

Single script: `research/research_portfolio_assembly.py`
- CLI: `--db-path` (optional), `--exclude-regime` (optional, to see CORE-only)
- Imports from `scripts/reports/report_edge_portfolio.py`
- Output: stdout text report (same style as existing research scripts)
