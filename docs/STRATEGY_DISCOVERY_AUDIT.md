# Strategy Discovery & Search System — Comprehensive Audit

**Document Purpose:** Complete technical inventory of how strategies are discovered, searched, evaluated, and selected in the Canompx3 MGC trading pipeline. Designed for knowledge transfer and architectural brainstorming.

**Last Updated:** 2026-02-25
**Scope:** Discovery, filtering, validation, fitness assessment, portfolio selection
**Status:** Production system (1,811 tests passing, 35 drift checks passing). Note: strategy counts in this doc are from Feb 25 snapshot (1,322 pre-E0-purge). Current state: 627 validated, 206 edge families.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Discovery Phase (Grid Search)](#discovery-phase-grid-search)
3. [Filtering Architecture](#filtering-architecture)
4. [Validation Pipeline](#validation-pipeline)
5. [Fitness Assessment](#fitness-assessment)
6. [Portfolio Selection](#portfolio-selection)
7. [Metrics & Scoring](#metrics--scoring)
8. [Database Schema](#database-schema)
9. [Decision Trees](#decision-trees)
10. [What We Do Well](#what-we-do-well)
11. [Gaps & Blind Spots](#gaps--blind-spots)
12. [Anti-Patterns to Stop](#anti-patterns-to-stop)
13. [Improvement Opportunities](#improvement-opportunities)
14. [Research Archive (What Didn't Work)](#research-archive-what-didnt-work)
15. [Key Constraints & Rules](#key-constraints--rules)

---

## System Overview

### High-Level Flow

```
Raw OHLCV bars (1m)
  ↓
Daily features (ORBs, sessions, RSI)
  ↓
Pre-computed outcomes (~6.1M rows, 5/15/30m, 7 instruments)
  ↓
Grid search: session-aware combos (3 EMs x filters x RR x CB)
  ↓
Experimental strategies (39,198 candidates)
  ↓
6-phase validation pipeline
  ↓
Validated strategies (1,322 active)
  ↓
Edge family clustering (472 families)
  ↓
Portfolio construction & fitness monitoring
```

### Discovery Approach

**Exhaustive grid search** over fixed parameters:
- **10 ORB sessions** (all event-based/dynamic, per-day resolution from SESSION_CATALOG)
- **6 risk/reward targets** [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
- **5 confirm bar values** [1, 2, 3, 4, 5]
- **13+ ORB size filters** (NO_FILTER, L-filters, G-filters, VOL, band filters, direction filters)
- **3 entry models** [E0=limit on confirm (CB1 only), E1=next bar open, E3=limit retrace (CB1 only)]
- **Per-instrument session gating** (enabled_sessions from asset_configs)
- **Session-aware filter dispatch** via `get_filters_for_grid()` in config.py

**Note on entry model biases:** E0 has 3 optimistic backtest biases (fill-on-touch, fakeout exclusion, fill-bar wins). E1 is the honest baseline with no biases. See TRADING_RULES.md Entry Model Bias Audit.

### Current State

| Metric | Value | Notes |
|--------|-------|-------|
| orb_outcomes | 5,807,946 | 7 instruments × 5/15/30m ORB apertures |
| experimental_strategies | 39,198 | All instruments, all sessions |
| **Validated active** | **1,322** | MGC: 339, MES: 273, MNQ: 614, M2K: 96 |
| **By entry model** | E0: 817, E1: 442, E3: 63 | E0 CB1 only (CB2+ purged Feb 2026) |
| **Edge families** | **472** | Unique trade-day patterns |
| Dead instruments | MCL, SIL, M6E | 0 validated for ORB breakout |

---

## Discovery Phase (Grid Search)

### Entry Point: `strategy_discovery.py`

**File:** `/trading_app/strategy_discovery.py` (513 lines)
**Entry Function:** `run_discovery(db_path, instrument, start_date, end_date, orb_minutes=5, dry_run=False)`

### Execution Flow

#### Stage 1: Bulk Load Data (Minimize DB round-trips)
```python
# Single query per orb_minutes
features = _load_daily_features(con, instrument, orb_minutes, start, end)
# ~250 days × columns = fast

# Compute relative volumes once (if VolumeFilter used)
_compute_relative_volumes(con, features, instrument, sessions, ALL_FILTERS)

# Build matching day sets for all (filter, ORB) combos
filter_days = _build_filter_day_sets(features, sessions, ALL_FILTERS)

# Load outcomes grouped by (orb_label, entry_model, rr_target, confirm_bars)
outcomes_by_key = _load_outcomes_bulk(con, instrument, orb_minutes, sessions, ENTRY_MODELS)
```

**Performance:** ~1-2 minutes for full grid (5,148 combos × 250 days)

#### Stage 2: Grid Iteration (Pure Python, no DB reads)
```python
for filter_key in ALL_FILTERS:                      # 13 filters
    for orb_label in sessions:                      # 7-8 ORBs
        matching_day_set = filter_days[(filter_key, orb_label)]

        for entry_model in ENTRY_MODELS:            # E1, E3
            for rr_target in RR_TARGETS:            # 6 targets
                for confirm_bars in CONFIRM_BARS:   # 5 values
                    if entry_model == "E3" and confirm_bars > 1:
                        continue  # Skip (identical to CB1)

                    all_outcomes = outcomes_by_key.get((orb_label, em, rr, cb), [])
                    outcomes = [o for o in all_outcomes if o["trading_day"] in matching_day_set]

                    if outcomes:
                        metrics = compute_metrics(outcomes, cost_spec)
                        strategy_id = make_strategy_id(...)
                        # INSERT OR REPLACE into experimental_strategies
```

**Design Principle:** Load data once, iterate cheaply in Python. Batch writes every 500 rows.

### Key Functions

#### `compute_metrics(outcomes, cost_spec) → dict`

**Calculates:**
- `sample_size`: Count of win/loss trades (scratches/early_exits excluded)
- `win_rate`: wins / (wins + losses)
- `avg_win_r`: Mean R of winning trades
- `avg_loss_r`: Mean R of losing trades (absolute value)
- `expectancy_r`: (win_rate × avg_win_r) - ((1 - win_rate) × avg_loss_r)
- `sharpe_ratio`: mean(R) / stdev(R) [Bessel's correction: n-1]
- `max_drawdown_r`: Peak-to-trough in cumulative R-equity curve
- `trades_per_year`: N_traded / date_span_years (floored at 0.25 = 3 months)
- `sharpe_ann`: sharpe_ratio × sqrt(trades_per_year)
- `yearly_results`: JSON dict keyed by year
  - Per year: trades, wins, win_rate, total_r, avg_r

**Critical Detail:** Scratches (partial fills) are excluded from W/L stats but included in sample_size calculation internally.

#### `_build_filter_day_sets(features, orb_labels, all_filters) → dict`

**Purpose:** Pre-compute matching day sets for all (filter, ORB) combos

**Logic:**
```python
for filter_key, strategy_filter in all_filters.items():
    for orb_label in orb_labels:
        days = set()
        for row in features:
            if row.get(f"orb_{orb_label}_break_dir") is not None:
                if strategy_filter.matches_row(row, orb_label):
                    days.add(row["trading_day"])
        result[(filter_key, orb_label)] = days
```

**Why Pre-compute:** Avoids redundant filter.matches_row() calls during grid iteration (would run billions of times).

#### `_load_outcomes_bulk(con, instrument, orb_minutes, orb_labels, entry_models) → dict`

**SQL Pattern:**
```sql
SELECT trading_day, rr_target, confirm_bars, outcome, pnl_r, mae_r, mfe_r,
       entry_price, stop_price
FROM orb_outcomes
WHERE symbol = ? AND orb_minutes = ?
  AND orb_label = ? AND entry_model = ?
  AND outcome IS NOT NULL
```

**Grouped by:** (orb_label, entry_model, rr_target, confirm_bars)

**Size:** ~6.1M rows across all instruments (5/15/30m ORB apertures) (covers all RR targets × CBs for valid breaks)

### Outcome Pre-Computation (outcome_builder.py)

**Purpose:** Pre-compute orb_outcomes table (~6.1M rows) once, reuse for all strategy discovery.

**Approach:**
```
For each trading day:
    For each ORB session (0900, 1000, etc.):
        For each entry_model (E1, E3):
            For each confirm_bars (1-5):
                ├─ Detect entry: first CB bars outside ORB
                ├─ (CB resets if bar closes back inside)
                ├─ For each RR target [1.0 - 4.0]:
                │   ├─ Compute target price = entry + (entry - stop) × RR
                │   ├─ Check if target hit before stop
                │   ├─ Track early exit (N min losers)
                │   └─ Record outcome, MAE, MFE
                └─ INSERT all 6 RRs from one entry detection
```

**Optimization:** Reuse single entry detection for all 6 RR targets (reduces redundant frame slicing).

**Entry Models:**
- **E1:** Enter at next bar open after confirm bar (100% fill on breaks)
- **E3:** Limit order at ORB level, wait for retrace (96-97% fill on G5+ days)

**Early Exit:** Kill losers after N minutes (only 0900=15min, 1000=30min; others hold to target/stop)
- Rationale: 0900 losers: only 24% eventually win if held 15m. 1000 losers: only 12% if held 30m.

---

## Filtering Architecture

### Filter Types (config.py)

#### Base Class: `StrategyFilter`
```python
class StrategyFilter:
    def matches_row(self, row: dict, orb_label: str) -> bool:
        """Return True if day is eligible (passes filter conditions)."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Serialize filter state."""
```

#### No Filter (Pass-through)
```python
class NoFilter(StrategyFilter):
    def matches_row(self, row, orb_label):
        return True
    # filter_type = "NO_FILTER"
    # Result: All days eligible
    # Performance: Negative ExpR (house wins on all ORB sizes)
```

#### ORB Size Filters
```python
class OrbSizeFilter(StrategyFilter):
    def __init__(self, min_size=None, max_size=None):
        self.min_size = min_size  # Inclusive lower bound
        self.max_size = max_size  # Exclusive upper bound

    def matches_row(self, row, orb_label):
        size = row.get(f"orb_{orb_label}_size")
        if size is None:
            return False
        if self.min_size is not None and size < self.min_size:
            return False
        if self.max_size is not None and size >= self.max_size:
            return False
        return True
```

**Predefined Filters:**
- **L2, L3, L4, L6, L8:** "Less than" (all NEGATIVE ExpR)
- **G2, G3, G4, G5, G6, G8:** "Greater than" (G4+ has POSITIVE ExpR)
  - G4: size >= 4.0pt
  - G5: size >= 5.0pt (strong edge)
  - G6: size >= 6.0pt (G6+ best per-trade Sharpe)
  - G8: size >= 8.0pt (extreme conditions)

#### Volume Filter
```python
class VolumeFilter(StrategyFilter):
    def __init__(self, min_rel_vol=1.2, lookback_days=20):
        self.min_rel_vol = min_rel_vol
        self.lookback_days = lookback_days

    def matches_row(self, row, orb_label):
        rel_vol = row.get(f"rel_vol_{orb_label}")
        if rel_vol is None:
            return False  # Fail-closed: missing data rejects day
        return rel_vol >= self.min_rel_vol
```

**Relative Volume:** `break_bar_vol / median(same minute-of-day, N prior days)`

**Fail-Closed Principle:** If volume data missing → day ineligible. Prevents accidental high-vol classification.

### Filter Effectiveness

**Key Finding:** Many filters produce **identical trade sets** on the same ORB/session combination.

**Example:** MGC 0900 E1 RR2.0 CB2 with different filters may all have N=47 trades because only the top-percentile days have breaks (G6/G8 constraints become redundant).

**Deduplication Strategy:**
1. Group by (instrument, orb_label, entry_model, rr_target, confirm_bars)
2. Report only unique trade sets
3. Mark redundant filters as cosmetic (G2/G3 on many sessions)

### Session-Based Gating

**File:** `pipeline/asset_configs.py`

```python
ASSET_CONFIGS = {
    "MGC": {
        "enabled_sessions": ["0900", "1000", "1100", "1800", "2300", "CME_OPEN", "LONDON_OPEN"],
    },
    "MNQ": {
        "enabled_sessions": ["0900", "1000", "1100", "1800", "0030", "CME_OPEN", "US_EQUITY_OPEN"],
    },
    "MES": {
        "enabled_sessions": ["0900", "1000", "1100", "1800", "0030", "CME_OPEN", "US_EQUITY_OPEN", "US_DATA_OPEN"],
    },
}
```

**Purpose:** Skip sessions with no validated edge (e.g., 1100 is permanently degraded, MCL has zero edge).

**Effect:** Reduces effective grid from 5,148 to ~3,500-4,000 per instrument.

---

## Validation Pipeline

### 7-Phase Validation (strategy_validator.py + walkforward.py)

**Entry:** Experimental strategies (39,198 total across all instruments)
**Exit:** Validated strategies (promoted to validated_setups)
**Files:** `/trading_app/strategy_validator.py`, `/trading_app/walkforward.py`

### Phase 1: Sample Size Gate
```
if sample_size < min_sample:
    REJECT
if sample_size < 100:
    WARN (underfunded edge)
else:
    PASS
```

**Default:** `min_sample = 30`
**Rationale:** <30 trades = statistical noise

### Phase 2: Post-Cost Expectancy
```
exp_r_after_cost = exp_r - (friction_per_trade / median_risk_points)
if exp_r_after_cost <= 0:
    REJECT
else:
    PASS
```

**Friction:** From cost_model.py
- MGC: $10/point × $5.74 RT friction = 0.574R per round-trip
- MNQ: $2/point × $2.74 RT friction = 1.37R per RT
- MES: $5/point × $3.74 RT friction = 0.748R per RT

### Phase 3: Yearly Robustness
```
for year in years_tested:
    if year not in exclude_years:
        if avg_r_that_year <= 0:
            REJECT (year failed)

if min_years_positive_pct < threshold:
    REJECT

else:
    PASS
```

**Configurable:**
- `exclude_years` (default: empty, can exclude 2021 noise)
- `min_years_positive_pct` (default: 100%, can relax to 0.8)

### Phase 4: Stress Test
```
extra_cost_multiplier = 1.5  # +50% friction
stress_cost_per_trade_r = (friction * extra_multiplier) / median_risk_points

stress_exp_r = exp_r - stress_cost_per_trade_r
if stress_exp_r > 0:
    PASS (survives adverse conditions)
else:
    REJECT (fragile to slippage)
```

**Rationale:** Real trading has gaps, slippage, correlation jumps. Test if edge survives +50% friction.

### Phase 4b: Walk-Forward OOS Validation
```
# Anchored expanding walk-forward with 6-month non-overlapping test windows
# Minimum 12 months training before first test window

for each test window:
    test_outcomes = filtered outcomes in [window_start, window_end)
    compute test_n, test_exp_r

valid_windows = windows where test_n >= min_trades_per_window (15)

Pass rule (ALL 4, fail-closed):
  1. n_valid >= min_valid_windows (3)
  2. pct_positive >= min_pct_positive (60%)
  3. agg_oos_exp_r > 0 (trade-weighted)
  4. total_oos_trades >= min_trades * min_windows (45)
```

**Key behaviors:**
- MNQ/MES (2yr data): ~2 windows, fails min_windows=3. Use `--wf-min-windows 2` or `--no-walkforward`
- G6/G8 filters: few trades per window, most windows invalid. Correctly identifies as untestable OOS
- Results logged to `data/walkforward_results.jsonl` (JSONL, append-only, gitignored)
- Disable with `--no-walkforward` CLI flag

### Phase 5 (Optional): Sharpe Ratio Gate
```
if enable_sharpe_check:
    if sharpe_ratio < min_sharpe:
        REJECT (too volatile)
else:
    SKIP
```

### Phase 6 (Optional): Max Drawdown Gate
```
if enable_dd_check:
    if max_drawdown_r > max_drawdown:
        REJECT (unacceptable DD)
else:
    SKIP
```

### Result: Promotion Metadata
```python
# On PASS:
INSERT INTO validated_setups (
    strategy_id, promoted_from, promoted_at,
    years_tested, all_years_positive, stress_test_passed,
    status, ...
) VALUES (...)
```

---

## Fitness Assessment

### 3-Layer Fitness Model (strategy_fitness.py)

**Purpose:** Monitor validated strategies for regime fitness (live trading signal).

**Layers:**

#### Layer 1: Structural Edge (Full Period)
- From validated_setups: exp_r, sharpe_ann, max_drawdown_r
- Static: reflects entire backtest period (2024-2026 for MGC)
- **Use Case:** Portfolio allocation baseline

#### Layer 2: Rolling Regime Fitness (18-month default)
- Recompute metrics on trailing 18-month window
- If rolling_exp_r > 0 AND rolling_sample >= 15: **FIT**
- If rolling_exp_r <= 0: **DECAY**
- If rolling_sample < 10: **STALE**

#### Layer 3: Decay Monitoring (Recent Sharpe)
- 30-day and 60-day trailing Sharpe
- If recent_sharpe_30 < -0.1 (even if rolling_exp > 0): **WATCH**
- **Use Case:** Early warning of degradation

### Classification Decision Tree

```
if rolling_sample < 10:
    STALE (insufficient data)
else:
    if rolling_exp_r > 0:
        if rolling_sample >= 15:
            if recent_sharpe_30 > -0.1:
                FIT (healthy)
            else:
                WATCH (declining momentum)
        else:
            WATCH (thin data, 10-14 trades)
    else:
        DECAY (rolling ExpR negative)
```

### Output: FitnessScore Dataclass
```python
@dataclass
class FitnessScore:
    strategy_id: str
    fitness_status: str  # FIT, WATCH, DECAY, STALE
    rolling_exp_r: float
    recent_sharpe_30: float
    rolling_sample: int
    structural_edge_exp_r: float  # Full period
```

---

## Portfolio Selection

### From Validated Strategies to Portfolio (portfolio.py)

**Input:** validated_setups table (1,322 active across 4 instruments)
**Output:** 10-20 diversified strategies with position sizing

### Selection Algorithm

#### Step 1: Load Validated Strategies
```python
load_validated_strategies(
    db_path, instrument, min_expectancy_r=0.10,
    include_nested=False, include_rolling=False,
    family_heads_only=False
)
```

**Filters:**
- `status = 'active'`
- `orb_label != '1100'` (permanently degraded)
- `expectancy_r >= min_expectancy_r` (default 0.10)
- Optional: `family_heads_only=True` → only edge family heads (1 per family)

#### Step 2: Compute Correlation Matrix
```python
correlation_matrix(db_path, strategy_ids, min_overlap_days=200)
```

**Purpose:** Identify redundant strategies (high correlation = similar trade days)

**Pairs with <200 overlapping non-NaN days get NaN correlation** (insufficient data for meaningful correlation)

**Result:** Symmetric correlation matrix, diagonal = 1.0

#### Step 3: Diversify
```python
diversify_strategies(
    candidates, max_strategies=20,
    max_per_orb=5, max_per_entry_model=None,
    corr_matrix=corr, max_correlation=0.85
)
```

**Selection Priority:**
1. Highest ExpR first
2. Enforce max per ORB label (e.g., max 5 from 0900 session)
3. Enforce max per entry model (if set)
4. Reject candidates too correlated with already-selected (rho >= 0.85)
5. Stop at max_strategies

#### Step 4: Position Sizing
```python
compute_position_size(
    account_equity, risk_per_trade_pct,
    risk_points, cost_spec
)
```

**Formula:**
```
contracts = (account_equity × risk_pct / 100) / (risk_points × point_value)
```

**Example:** $25K account, 2% risk, 10pt ORB, $10/point → 5 contracts

---

## Metrics & Scoring

### Core Metrics (compute_metrics)

| Metric | Formula | Use Case | Notes |
|--------|---------|----------|-------|
| **Win Rate** | wins / (wins + losses) | % profitable trades | Excludes scratches |
| **Expectancy (ExpR)** | (WR × AvgW) - (LR × AvgL) | Expected R per trade | In risk units, not dollars |
| **Sharpe Ratio** | mean(R) / stdev(R) | Per-trade consistency | Bessel's correction (n-1) |
| **Max Drawdown** | peak - trough in cumE | Worst-case excursion | Equity curve tracking |
| **Trades/Year** | N / date_span_years | Annual trade frequency | Floored at 3 months minimum |
| **Sharpe Ann** | Sharpe × sqrt(trades_per_year) | Scaled consistency | Must match trade frequency |

### Critical Insight: Sharpe Annualization

**Wrong:** Per-trade Sharpe alone (meaningless without frequency)
**Right:** Annualized Sharpe = per_trade_Sharpe × sqrt(trades_per_year)

**Example:**
- Strategy A: Sharpe=0.5, 50 trades/year → ShAnn = 0.5 × sqrt(50) ≈ 3.5
- Strategy B: Sharpe=0.6, 10 trades/year → ShAnn = 0.6 × sqrt(10) ≈ 1.9

Strategy A is better despite lower per-trade Sharpe because it has more opportunities.

**Project Minimum Bar:** ShAnn >= 0.5 (requires ~150+ trades/year)

### Yearly Breakdown

**Computed for Each Year:**
```python
yearly[year] = {
    "trades": N,
    "wins": W,
    "win_rate": W / N,
    "total_r": sum(pnl_r),
    "avg_r": sum(pnl_r) / N,
}
```

**Validation Rule:** All years must be positive (no negative years allowed, unless `exclude_years` used)

---

## Database Schema

### Core Tables

#### bars_1m (Pipeline Origin)
```sql
PRIMARY KEY (symbol, ts_utc)
Columns: symbol, source_symbol, open, high, low, close, volume
Size: MGC 3.5M rows (2016-2026), MNQ 700K (2yr), MES 700K (2yr)
```

#### daily_features (Derived Features)
```sql
PRIMARY KEY (symbol, trading_day, orb_minutes)
Columns:
  - Session columns: 11 ORB labels × 9 columns each
    ├─ orb_0900_high, orb_0900_low, orb_0900_size
    ├─ orb_0900_break_dir (-1/0/+1), orb_0900_break_ts
    ├─ orb_0900_outcome ('win'/'loss'/'scratch'/'early_exit'/NULL)
    ├─ orb_0900_mae_r, orb_0900_mfe_r, orb_0900_double_break
    └─ ... (repeat for 1000, 1100, 1130, 1800, 2300, 0030, CME_OPEN, ...)
  - Risk: atr_20 (20-day ATR for regime detection)
  - RSI: rsi_14_at_0900
  - OHLC: daily_open, daily_high, daily_low, daily_close
  - Gap: gap_open_points (today_open - yesterday_close)
Size: MGC 8,766 days, MNQ 584 days, MES 585 days
```

#### orb_outcomes (Pre-Computed Outcomes)
```sql
PRIMARY KEY (symbol, trading_day, orb_label, orb_minutes, rr_target, confirm_bars, entry_model)
FK to daily_features (symbol, trading_day, orb_minutes)
Columns:
  - Entry: entry_ts, entry_price, stop_price, target_price
  - Exit: outcome ('win'/'loss'/'scratch'/'early_exit'/NULL)
  - exit_ts, exit_price, pnl_r
  - Excursions: mae_r, mfe_r
Size: MGC ~6.1M rows (all RR/CB combos for valid breaks)
```

#### experimental_strategies (Discovery Output)
```sql
PRIMARY KEY strategy_id
Columns:
  - Params: instrument, orb_label, orb_minutes, rr_target, confirm_bars, entry_model
  - Filter: filter_type, filter_params (JSON)
  - Metrics: sample_size, win_rate, avg_win_r, avg_loss_r, expectancy_r
  - Sharpe: sharpe_ratio, trades_per_year, sharpe_ann
  - Risk: max_drawdown_r, median_risk_points, avg_risk_points
  - Yearly: yearly_results (JSON)
  - Validation: validation_status (NULL/'pending'), validation_notes
Size: 39,198 rows total (MGC + MNQ + MES + M2K)
```

#### validated_setups (Passed Validation)
```sql
PRIMARY KEY strategy_id
FK promoted_from (experimental_strategies)
Columns:
  - All params from experimental + promotion metadata
  - Validation results: years_tested, all_years_positive, stress_test_passed
  - Status: status ('active'/'retired'), retired_at, retirement_reason
  - Family: family_hash, is_family_head (denormalized)
Size: MGC 216 active, MNQ 610, MES 198
```

#### strategy_trade_days (Ground Truth)
```sql
PRIMARY KEY (strategy_id, trading_day)
Purpose: Track all post-filter eligible trading days for each strategy
Size: MGC 332K rows (for 216 validated strategies)
```

#### edge_families (Strategy Clustering)
```sql
PRIMARY KEY family_hash
Columns:
  - Clustering: instrument, member_count, trade_day_count
  - Head election: head_strategy_id, head_expectancy_r, head_sharpe_ann
  - Metrics: cv_expectancy, median_expectancy_r, avg_sharpe_ann, min_member_trades
  - Classification: robustness_status, trade_tier
Size: MGC 48 families, MNQ 119, MES 48
```

### Query Patterns

**Find Strategies by Filter:**
```sql
SELECT * FROM validated_setups
WHERE instrument = 'MGC' AND orb_label = '0900' AND filter_type = 'ORB_G5'
ORDER BY expectancy_r DESC
```

**Find Strategies by ORB/Session:**
```sql
SELECT COUNT(*), AVG(expectancy_r), MAX(sharpe_ann)
FROM validated_setups
WHERE instrument = 'MGC' AND orb_label IN ('0900', '1000')
GROUP BY orb_label
```

**Find Family Heads:**
```sql
SELECT vs.* FROM validated_setups vs
JOIN edge_families ef ON vs.strategy_id = ef.head_strategy_id
WHERE vs.instrument = 'MGC' AND ef.robustness_status != 'PURGED'
```

---

## Decision Trees

### Discovery Grid Decision Tree

```
for each (filter, orb_label):
    ├─ Compute matching_day_set via filter.matches_row()
    └─ for each entry_model in [E1, E3]:
        └─ for each rr_target in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            └─ for each confirm_bars in [1, 2, 3, 4, 5]:
                ├─ Skip if (entry_model == E3 AND confirm_bars > 1)
                ├─ Load pre-computed outcomes for (orb, em, rr, cb)
                ├─ Filter outcomes to matching_day_set
                ├─ Compute metrics: ExpR, Sharpe, WR, etc.
                ├─ If sample_size > 0:
                │   └─ INSERT INTO experimental_strategies
                └─ Batch flush every 500 rows
```

### Validation Decision Tree

```
Is sample_size >= min_sample?
├─ NO → REJECT
└─ YES → Phase 2

Is exp_r_after_cost > 0?
├─ NO → REJECT
└─ YES → Phase 3

Do all included years have avg_r > 0?
├─ NO → REJECT (fails yearly robustness)
└─ YES → Phase 4

Does stress_exp_r (1.5x friction) > 0?
├─ NO → REJECT (fragile)
└─ YES → Phase 5/6 (optional checks)

[PASSED] → Promote to validated_setups
```

### Fitness Classification Tree

```
rolling_sample >= 10?
├─ NO → STALE
└─ YES → Continue

rolling_exp_r > 0?
├─ NO → DECAY
└─ YES → Continue

recent_sharpe_30 > -0.1?
├─ NO → WATCH (declining)
└─ YES → FIT
```

### Portfolio Selection Tree

```
For each candidate (ordered by ExpR DESC):
    ├─ Already selected >= max_strategies?
    │   └─ YES → Stop
    │
    ├─ orb_counts[candidate.orb] >= max_per_orb?
    │   └─ YES → Skip
    │
    ├─ em_counts[candidate.em] >= max_per_entry_model?
    │   └─ YES → Skip
    │
    └─ Is candidate too correlated (rho >= 0.85) with any selected?
        ├─ YES → Skip
        └─ NO → SELECT & update counts
```

---

## What We Do Well

### 1. Exhaustive Grid Search
- **Strength:** 5,148 combos ensures no obvious combo is missed
- **Coverage:** All ORB sizes, RR targets, entry models, filters evaluated
- **Repeatability:** Same combo always produces same metrics (deterministic)

### 2. Pre-Computed Outcomes
- **Efficiency:** ~6.1M outcomes pre-computed once, reused for all discovery
- **Accuracy:** Outcomes computed at exact bar timestamps (no approximation)
- **Flexibility:** Can re-run discovery with different filters on same outcomes

### 3. Multi-Phase Validation
- **Rigor:** 7 gates catch fragile strategies (low sample, negative years, slippage-sensitive)
- **Stress Testing:** +50% friction gate filters unrealistic winners
- **Yearly Robustness:** Rejects "lucky year" strategies (must be positive all years)
- **Walk-Forward OOS:** Phase 4b anchored walk-forward rejects strategies with concentrated performance

### 4. Filter Architecture
- **Modularity:** New filters can be added by subclassing StrategyFilter
- **Reusability:** Same filters used in discovery, setup_detector, portfolio analytics
- **Composability:** Multiple filters tested in same grid iteration

### 5. Correlation-Aware Portfolio
- **Redundancy Detection:** Identifies and deduplicates strategies with same trade days
- **Diversification:** Enforces limits per ORB/session to spread exposure
- **Overlap Guard:** Only counts days with sufficient overlap (>200 days) for correlation

### 6. Edge Family Clustering
- **Deduplication:** 1,322 validated strategies collapse into 472 unique families
- **Head Election:** Median ExpR head avoids Winner's Curse bias
- **Robustness Classification:** ROBUST/WHITELISTED/PURGED per Duke Protocol #3c

### 7. Comprehensive Testing
- **1,072 tests** passing (all critical paths covered)
- **35 drift checks** passing (architectural guardrails active)
- **Pre-commit hooks** enforce validation on every commit

### 8. Well-Documented Parameters
- **Cost model:** Explicit friction + point values per instrument
- **Entry models:** E1/E3 clearly defined
- **ORB durations:** Per-session, documented in config
- **Filter logic:** Each filter has clear include/exclude bounds

---

## Gaps & Blind Spots

### 1. **No A/B Testing Framework for Discovery Parameters**
- **Gap:** Grid search parameters (RR targets, CB values, ORB durations) are fixed
- **Evidence:** 1000 session uses 30-min early exit, but no test if 15/45-min better
- **Opportunity:** Parameterize exit windows, test sensitivity
- **Cost:** Would require nested grid search (exponential complexity)

### 2. **Limited Signal Diversity**
- **Current Signals:** Only ORB breakouts + size filters + entry models
- **Missing:**
  - Volume breakouts (VolumeFilter exists but untested at scale)
  - Volatility regime switching
  - Session cascade (prior session range influence)
  - Market microstructure (VWAP, order imbalance)
  - Machine learning features
- **Evidence:** Research archive shows zero edge from ADX, RSI, double-break, gap-fade, prior-day levels

### 3. **No Real-Time Parameter Adaptation**
- **Gap:** All strategies have fixed parameters (0900 always 5-min ORB, etc.)
- **Question:** Should CME_OPEN ORB be 3m vs 5m vs 10m on volatile days?
- **Blocker:** Would require live feature engineering + decision tree

### 4. **Weak Nested ORB Research**
- **Current State:** Tables exist but only partially populated (MCL, older instruments)
- **Gap:** 1000 session tested at 15m ORB, but comparison vs 5m/30m incomplete
- **Opportunity:** Systematic 15m/30m ORB grid for all instruments + entry bars
- **Cost:** Would double outcome pre-computation size

### 5. **No Cross-Instrument Signal Transfer**
- **Gap:** Each instrument (MGC, MNQ, MES) has independent discovery
- **Question:** MGC 0900 edge → does MNQ 0900 have same edge?
- **Blocker:** Different contract structures, cost models, tick sizes complicate transfer
- **Opportunity:** Correlation study across instruments, shared regime detection

### 6. **Limited Ensemble Approaches**
- **Current:** Single-strategy position sizing + diversification
- **Missing:**
  - Weighted ensemble (ExpR-weighted + Sharpe-weighted variations)
  - Dynamic rebalancing (weekly/monthly strategy rotation)
  - Conditional selection (use REGIME-class only on high-volatility days)
  - Time-of-day gates (disable 0900 strategy on FOMC days)

### 7. **No Adaptive Confirm Bars**
- **Current:** CB is fixed (1-5, usually 1-3 best)
- **Gap:** Does CB optimal differ on low-liquidity vs high-liquidity days?
- **Opportunity:** Liquidity-based confirm bar selection

### 8. **Portfolio Optimization Oversimplified**
- **Current:** Max-per-ORB + max-correlation filters only
- **Missing:**
  - Mean-variance optimization (Markowitz)
  - Risk parity (equal risk contribution)
  - Drawdown minimization (portfolio DD, not per-strategy DD)
  - Sharpe maximization across selected subset

### 9. **No Intra-Day Regime Detection**
- **Gap:** Same strategy runs identically on 20pt vol days vs 2pt vol days
- **Question:** Should G4 filter apply only on normal vol, G8 on high vol?
- **Opportunity:** Dynamic filter selection based on overnight range / gap / prior session

### 10. **Validation Assumes IID Outcomes**
- **Issue:** Yearly robustness check treats each year independently
- **Gap:** No check for sequential correlation (is 2024 success because of 2023 persistence?)
- **Opportunity:** Serial autocorrelation test, regime persistence metrics

---

## Anti-Patterns to Stop

### 1. **"Higher Sharpe Always Better" Myth**
- **Anti-Pattern:** Selecting strategies purely by sharpe_ann
- **Problem:** Sharpe inflates with lower frequency (10 trade/year strategy can have high ShAnn)
- **What to Do Instead:** Require minimum trades_per_year (150+) AND ShAnn >= 0.5
- **Evidence:** Some G6/G8 strategies have ShAnn=0.8 with N=12 trades/year (statistically invalid)

### 2. **Ignoring Filter Redundancy**
- **Anti-Pattern:** Treating G2, G3, G4, etc. as independent strategies
- **Problem:** Many produce identical trade sets (redundant coverage)
- **What to Do Instead:** Deduplicate by (instrument, orb_label, entry_model, rr_target, confirm_bars) before promotion
- **Evidence:** MGC 0900 E1 RR2.0 CB2: G2 and G3 often match G4 trade days, only G4+ meaningfully filters

### 3. **Trusting Full-Period Statistics**
- **Anti-Pattern:** Promoting strategy based on 2021-2026 backtest
- **Problem:** Market regime changed (2021 tiny ORBs, 2024-26 larger ORBs), 2016-2020 data is stale
- **What to Do Instead:** Always recompute on rolling windows, enforce minimum years_tested >= 2
- **Evidence:** Validated strategies flagged DECAY with rolling fitness
- **Checkpoint:** strategy_fitness.py must run before live trading

### 4. **Neglecting Cost Model Sensitivity**
- **Anti-Pattern:** Using base cost for validation, different cost in live trading
- **Problem:** $5.74 friction vs $12.00 actual friction → ExpR flips negative
- **What to Do Instead:** Always stress-test at 1.5x friction in Phase 4
- **Evidence:** Stress gate catches ~30% of discovered strategies

### 5. **One Strategy Per ORB Session**
- **Anti-Pattern:** Portfolio with single 0900 strategy
- **Problem:** Overnight news → missed edge opportunity
- **What to Do Instead:** Use edge families + multiple heads per session (if uncorrelated)
- **Checkpoint:** Portfolio should have 2+ strategies per ORB session

### 6. **Ignoring MAE/MFE**
- **Anti-Pattern:** Only tracking final PnL_R
- **Problem:** High-MAE strategies are fragile to slippage/gaps
- **What to Do Instead:** Correlate MAE with slippage % (if mae_r > 2R on avg, flag risky)
- **Evidence:** MGC 1800 E1 strategies have high MAE (1.5-2R) despite positive expectancy

### 7. **Static Grid Parameters**
- **Anti-Pattern:** Same ORB size filters across bull/bear/chop markets
- **Problem:** G5 filter optimal in 10-30pt ORB regime, but G8 needed for 2-5pt regime
- **What to Do Instead:** Compute regime classifier (vol-based), apply conditional filters
- **Blocker:** Requires live regime detection infrastructure

### 8. **Counting Scratches as Zero-Return Days**
- **Anti-Pattern:** Including scratches (partial fills) in daily return series
- **Problem:** Understates variance (scratches are often smallest-risk entries)
- **What to Do Instead:** Exclude scratches from Sharpe calc, track separately as "N_entry_signals"
- **Current Practice:** Scratches excluded from W/L, but included in sample_size (confusing)

### 9. **Assuming Linear Correlation**
- **Anti-Pattern:** Using Pearson correlation for strategy pairs
- **Problem:** Strategies can be uncorrelated linearly but co-move in tail events
- **What to Do Instead:** Also compute tail-risk correlation, drawdown correlation
- **Opportunity:** Use rolling correlation windows (not full-period)

### 10. **Promoting Low-Frequency Strategies**
- **Anti-Pattern:** Validating 15-trade-per-year strategy if N=100+ total
- **Problem:** Extremely lumpy equity curve (long droughts between trades)
- **What to Do Instead:** Require trades_per_year >= 30 for CORE strategies
- **Evidence:** Current REGIME class (30-99 samples) often translates to <15 trades/year

---

## Improvement Opportunities

### Immediate (1-2 weeks, low risk)

#### 1. **Add Nested ORB Discovery for All Instruments**
- **Action:** Build nested_outcomes for MNQ, MES at 15m/30m ORB windows
- **Benefit:** Identify if longer ORB windows have better risk/reward
- **Effort:** Extend outcome_builder.py, re-run nested discovery
- **Expected Impact:** 10-20 new validated strategies per instrument

#### 2. **Implement Adaptive Early Exit Windows**
- **Action:** Grid search over early_exit_minutes: [10, 15, 20, 30, 45]
- **Benefit:** Find optimal hold time per session
- **Effort:** Parameterize outcome_builder.py, re-compute outcomes
- **Expected Impact:** 10-20% improvement in per-trade Sharpe for 1000 session

#### 3. **Add Regime-Based Filter Selection**
- **Action:** Classify trading days by ATR percentile, apply G4 on normal/G8 on high-vol
- **Benefit:** Reduce filter redundancy, capture vol-regime edge
- **Effort:** Add new column to daily_features, extend filter logic
- **Expected Impact:** 5-10 new validated strategies

#### 4. **Deduplicate Strategy Portfolio**
- **Action:** Use edge families for portfolio construction (1 head per family)
- **Benefit:** Eliminate redundant exposure, cleaner position sizing
- **Effort:** Modify portfolio.py to use family_heads_only flag
- **Expected Impact:** Same returns with fewer positions

#### 5. **Add MAE/MFE Analysis Dashboard**
- **Action:** Report distribution of MAE/MFE per strategy
- **Benefit:** Identify fragile strategies (high MAE = slippage-sensitive)
- **Effort:** Query orb_outcomes, compute percentiles
- **Expected Impact:** Flag ~20-30% of strategies as risky

### Short-term (1 month, moderate risk)

#### 6. **Implement Cross-Instrument Signal Correlation**
- **Action:** Test if MGC 0900 edge transfers to MNQ/MES 0900
- **Benefit:** Understand regime commonality across instruments
- **Effort:** Correlate validated_setups across instruments
- **Expected Impact:** Confidence in multi-instrument strategies

#### 7. **Build Rolling Window Stability Scorer**
- **Action:** Compute strategy stability (ShAnn on 6m/12m windows)
- **Benefit:** Surface degrading strategies before validation flag
- **Effort:** Add rolling queries to strategy_fitness.py
- **Expected Impact:** Proactive de-allocation of DECAY strategies

#### 8. **Add Session Cascade Feature**
- **Action:** Test if morning session quality predicts afternoon session quality
- **Benefit:** Conditional strategy selection (e.g., skip 1800 if 0900 was flat)
- **Effort:** Add cascade columns to daily_features, test in discovery
- **Expected Impact:** Likely minimal (research archive shows NO-GO, but worth re-checking)

#### 9. **Implement Drawdown-Minimization Portfolio**
- **Action:** Select strategies not by Sharpe, but by max_drawdown minimization
- **Benefit:** Compare Sharpe-max vs DD-min allocation
- **Effort:** Add new diversify_strategies variant using portfolio optimization
- **Expected Impact:** Lower DD at cost of lower returns

#### 10. **Add Entry Signal Count Tracking**
- **Action:** Report total "entry conditions met" (trades + scratches) per strategy
- **Benefit:** Distinguish high-frequency low-fill-rate from low-frequency high-fill-rate
- **Effort:** Query orb_outcomes, sum non-NULL entries
- **Expected Impact:** Better assessment of true trading frequency

### Medium-term (2-3 months, higher risk)

#### 11. **Implement Machine Learning Feature Discovery**
- **Action:** Add calculated features (RSI, MACD, Bollinger Bands, volume patterns)
- **Benefit:** Test if signals beyond ORB size improve edge
- **Effort:** Extend outcome_builder.py, compute features at break time
- **Expected Impact:** Potentially +30-50 new strategies (or zero, based on research)

#### 12. **Build Intra-Session Regime Detector**
- **Action:** Classify each trading day into regime (choppy/trending/volatile)
- **Benefit:** Conditional strategy selection mid-session
- **Effort:** Add regime state machine, gate live trading on regime
- **Expected Impact:** Reduce DD by skipping bad-regime sessions

#### 13. **Add Multi-Day Trend Alignment**
- **Action:** Test if multi-day close trends improve breakout follow-through
- **Benefit:** Identify trend-aligned opportunities
- **Effort:** Add prior-day close trend columns, test in discovery
- **Expected Impact:** Likely minimal (research shows NO-GO)

#### 14. **Implement Ensemble Strategy Voting**
- **Action:** Select 3-5 strategies per ORB, vote on direction before entry
- **Benefit:** Reduce false signals, improve confidence
- **Effort:** Modify execution_engine.py to accept multi-strategy voting
- **Expected Impact:** Lower win-rate but higher average-win (smaller DD)

#### 15. **Build Live Fitness Monitoring Dashboard**
- **Action:** Real-time fitness update, flash alerts on DECAY
- **Benefit:** Visual early-warning system for strategy degradation
- **Effort:** Add WebUI, integrate with paper_trader.py
- **Expected Impact:** Faster de-allocation decisions

### Long-term (6+ months, architectural)

#### 16. **Port Discovery to GPU-Accelerated Compute**
- **Action:** Rewrite outcome computation using PyTorch/CuPy
- **Benefit:** 10-100x speedup for nested ORB grids
- **Effort:** Complete rewrite of outcome_builder.py
- **Expected Impact:** Can explore 50,000+ combos instead of 5,148

#### 17. **Add Reinforcement Learning for Entry Timing**
- **Action:** Train RL agent to optimize confirm_bars dynamically
- **Benefit:** Adaptive entry model (not fixed E1/E3)
- **Effort:** New RL pipeline (reward shaping, training infrastructure)
- **Expected Impact:** Potentially +20-30% Sharpe improvement (high uncertainty)

#### 18. **Implement Live Market Microstructure Analysis**
- **Action:** Integrate order book data (if available) for fill prediction
- **Benefit:** Adaptive entry price (E3 limit better calibrated)
- **Effort:** API integration with broker/market data feed
- **Expected Impact:** Reduce slippage 10-20%

---

## Research Archive (What Didn't Work)

### NO-GO Findings

| Research | Finding | Why Failed | P-Value |
|----------|---------|-----------|---------|
| **ADX Trend Filter** | Trend-filtering pre-open trades | Adding ADX trend gate reduced edge size more than added selectivity | 0.73 |
| **Double-Break Reversal** | Fade second break after reversal | Out-of-sample: N=12, win_rate=45%, ExpR=-0.08 | N/A |
| **First Half-Hour Momentum** | Trade only first 30m | Weakest ORB advantage; edge degraded sharply | 0.82 |
| **Overnight Gap Fade** | Fade gap > 2pt | Insufficient signal strength; gap rarely predictive | 0.91 |
| **Prior Day High/Low Levels** | Test levels as support/resistance | No correlation between prior day levels and current breakout follow-through | 0.67 |
| **RSI Mean-Reversion** | RSI > 70 = reversal setup | Backtest showed edge, but money incinerator in OOS testing | Loss: $2.3K |
| **Session Cascade** | Use prior session range to predict current | Next-session quality independent of prior session quality | 0.88 |
| **Session Fade** | Asia→London range fadeout | Consistently negative expectancy across all combos | -0.15 ExpR avg |
| **VWAP Pullback** | Pullback to VWAP = entry | Worse than baseline; VWAP adds noise not signal | 0.58 |
| **Volume Confirmation** | High volume at break = stronger | Volume-to-follow uncorrelated; VolumeFilter created only ineligible days | N/A |

**Key Insight:** Only confirmed edge is **momentum breakouts (ORB > 4pt)** with size filters. Everything else either adds no signal or reduces edge through false-positive filtering.

---

## Key Constraints & Rules

### FIX5: Trade Day Invariant
- **Rule:** A trade day requires BOTH: (1) break occurred AND (2) filter makes day eligible
- **Implementation:** orb_outcomes contains ALL break-days regardless of filter
- **Portfolio Overlay Rule:** Only write pnl_r on eligible days (series.loc[td] == 0.0)
- **Violation Impact:** Double-counts opportunities, inflates frequency-based metrics

### Filter Deduplication
- **Rule:** Many filter variants produce identical trade sets
- **Check:** Always group by (instrument, orb_label, em, rr, cb) and count unique trades
- **Anti-Pattern:** Reporting G2, G3, G4, etc. as independent strategies (they're not)
- **Enforcement:** Deduplicate in validation phase

### Classification Hierarchy
```
CORE (>= 100 trades) > REGIME (30-99) > INVALID (< 30)
```

### Sharpe Annualization Formula (Non-Negotiable)
```
ShAnn = sharpe_ratio * sqrt(trades_per_year)
Minimum bar: ShAnn >= 0.5 (requires ~150 trades/year minimum)
```

### Yearly Robustness (No Carve-Outs)
- **Rule:** All years must have avg_r > 0
- **Exception:** Can exclude specific years (e.g., --exclude-years 2021)
- **Enforcement:** Phase 3 of validation

### Stress Test at 1.5x Friction
- **Rule:** Phase 4 validation must pass with +50% costs
- **Purpose:** Catch slippage-sensitive strategies
- **Formula:** stress_exp = base_exp - (1.5 × friction / risk_points)

### ORB Size Edge Zone
```
< 4pt:   House wins (NO EDGE)
4-10pt:  Core edge zone (CORE strategies)
> 10pt:  Thin opportunity set (REGIME only)
```

### Session Gating by Instrument
```
MGC: 0900, 1000, 1100, 1800, 2300, CME_OPEN, LONDON_OPEN (skip 0030)
MNQ: 0900, 1000, 1100, 1800, 0030, CME_OPEN, US_EQUITY_OPEN (skip 1130)
MES: 0900, 1000, 1100, 1800, 0030, CME_OPEN, US_EQUITY_OPEN, US_DATA_OPEN
MCL: (ZERO EDGE — all sessions skipped)
```

---

## Summary: What to Handoff to Next AI

### Critical Knowledge
1. **Grid search** produces 5,148 strategy combos (E1 + E3 entry models)
2. **Outcomes pre-computed** (~6.1M rows) and reused for all discovery
3. **Validation is rigorous:** 6-phase pipeline filters to 1,322 viable active strategies across 4 instruments
4. **Filtering creates redundancy:** Many combos produce identical trade sets
5. **Sharpe annualization** requires frequency matching (must have trades_per_year >= 30 minimum)
6. **Only confirmed edge:** Momentum breakouts (ORB > 4pt) + entry model selection
7. **Everything else NO-GO:** ADX, RSI, gaps, VWAP, session cascade all tested, zero edge

### Architectural Strengths to Preserve
- Pre-computed outcomes (reusable, efficient)
- Multi-phase validation (catches fragile strategies)
- Filter modularity (new filters easy to add)
- Edge family deduplication (1 head per family)
- Correlation-aware portfolio (identifies redundancy)

### Low-Hanging Fruit to Explore
- Nested ORB discovery (15m/30m windows)
- Adaptive early exit windows (per-session tuning)
- Regime-based filter selection (vol-aware)
- Cross-instrument signal transfer (MGC→MNQ→MES)

### Don't Waste Time On
- Session cascade research (already proven NO-GO)
- RSI/ADX trend filters (already tested, negative)
- Volume confirmation (uncorrelated signal)
- Gap-fade strategies (insufficient predictive power)
- Higher-order entry models (E1/E3 are already exhaustive)

---

## Files to Review First

**For Strategy Discovery:**
- `trading_app/strategy_discovery.py` (main grid search)
- `trading_app/outcome_builder.py` (outcome pre-computation)
- `trading_app/config.py` (filter definitions)

**For Validation:**
- `trading_app/strategy_validator.py` (6-phase pipeline)
- `trading_app/strategy_fitness.py` (regime fitness monitoring)

**For Selection:**
- `trading_app/portfolio.py` (diversification + sizing)
- `scripts/report_edge_portfolio.py` (portfolio reporting)

**For Configuration:**
- `pipeline/asset_configs.py` (per-instrument settings)
- `pipeline/cost_model.py` (friction/point values)
- `TRADING_RULES.md` (documented findings)

---

## Contact & Context

**Created:** 2026-02-15
**For:** Knowledge transfer & architectural brainstorming
**Scope:** Complete strategy discovery & search system
**Status:** Production-ready (1,811 tests, 35 drift checks passing)
