# M2.5 Audit Improvements — Implementation Plan
**Date:** 2026-03-04
**Source:** M2.5 grounded triage (verified at ~65% accuracy, false claims stripped)
**Method:** 2-pass implementation (Discovery complete, this is the plan)

---

## Scope

Three phases, ordered by ROI and complexity:

| Phase | What | Complexity | Files Changed | Est. Tasks |
|-------|------|-----------|---------------|------------|
| **1** | Wire vol-adjusted position sizing | LOW | 3 production + 2 test | 6 |
| **2** | T80 extension to 6 remaining sessions | VERY LOW | 1 config + 1 research | 4 |
| **3** | Partial profit taking research | HIGH | 1 production + 1 research | 5 |

---

## Phase 1: Wire Vol-Adjusted Position Sizing

### Problem
`compute_position_size_vol_scaled()` and `compute_vol_scalar()` exist in `portfolio.py` (lines 193-240) with full test coverage (5 tests). But they are **never called** from ExecutionEngine or paper_trader. All trades execute with `contracts=1`.

### What Exists
- `compute_vol_scalar(atr_20, median_atr_20)` → scalar clamped [0.5, 1.5]
- `compute_position_size_vol_scaled(account_equity, risk_per_trade_pct, risk_points, cost_spec, vol_scalar)` → int contracts
- `_daily_features_row` passed to ExecutionEngine on day start — **already contains `atr_20`**
- `risk_points = abs(entry_price - stop_price)` computed in `_try_entry()` before risk check

### What's Missing
1. **`median_atr_20`**: Rolling median of ATR_20 over N prior days. Not in `daily_features`. Must be computed from DB.
2. **`account_equity` and `risk_per_trade_pct`**: Not passed to ExecutionEngine. Must be added to constructor or day-start.
3. **Sizing call in `_try_entry()`**: The actual call to `compute_position_size_vol_scaled()` before the RiskManager check.

### Implementation Tasks

#### Task 1.1: Add median_atr_20 to paper_trader day loop
**File:** `trading_app/paper_trader.py`
**What:** In `replay_historical()`, after `_get_daily_features_row()`, query DB for rolling median ATR_20 (prior 252 trading days = ~1 year). Add `median_atr_20` to the `daily_features_row` dict passed to engine.
```python
# After line 265: df_row = _get_daily_features_row(con, instrument, td)
median_atr_20 = _get_median_atr_20(con, instrument, td, lookback_days=252)
if df_row is not None:
    df_row["median_atr_20"] = median_atr_20
```
**New helper:**
```python
def _get_median_atr_20(con, instrument: str, trading_day: date, lookback_days: int = 252) -> float:
    """Rolling median ATR_20 over prior N trading days."""
    result = con.execute("""
        SELECT MEDIAN(atr_20) FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
          AND trading_day < ? AND trading_day >= ? - INTERVAL ? DAY
    """, [instrument, trading_day, trading_day, lookback_days * 2]).fetchone()
    return result[0] if result[0] is not None else 0.0
```
**Risk:** LOW — read-only query, no schema change.

#### Task 1.2: Add account_equity and risk_per_trade_pct to ExecutionEngine
**File:** `trading_app/execution_engine.py`
**What:** Add two optional params to `__init__()`:
```python
def __init__(self, portfolio, cost_spec, ...,
             account_equity: float = 0.0,
             risk_per_trade_pct: float = 2.0):
```
When `account_equity > 0`, position sizing is active. When `account_equity == 0` (default), behavior is unchanged (`contracts=1`).

**Backward compatibility:** Default `account_equity=0.0` means ALL existing code paths are unchanged. No test breakage.

#### Task 1.3: Wire sizing into _try_entry()
**File:** `trading_app/execution_engine.py`
**What:** In each entry path (E2 line ~580, E1 line ~746, E3 line ~851), AFTER computing `risk_points` and BEFORE the RiskManager check, add:
```python
# Position sizing (when account_equity > 0)
if self.account_equity > 0 and risk_points > 0:
    from trading_app.portfolio import compute_position_size_vol_scaled, compute_vol_scalar

    atr_20 = (self._daily_features_row or {}).get("atr_20", 0.0) or 0.0
    median_atr_20 = (self._daily_features_row or {}).get("median_atr_20", 0.0) or 0.0
    vol_scalar = compute_vol_scalar(atr_20, median_atr_20)

    trade.contracts = compute_position_size_vol_scaled(
        self.account_equity, self.risk_per_trade_pct,
        risk_points, cost,  # cost already resolved (session-adjusted)
        vol_scalar,
    )
    if trade.contracts == 0:
        # Risk too large for account — reject entry
        trade.state = TradeState.EXITED
        self.completed_trades.append(trade)
        events.append(TradeEvent(..., reason="sizing_rejected: risk_exceeds_budget"))
        return events
```
**CRITICAL:** This goes BEFORE the RiskManager check. The `suggested_contract_factor` from RiskManager then multiplies the computed contracts (existing line 608 already does this).

**Pattern:** DRY — extract to a private method `_compute_contracts()` called from all 3 entry paths.

#### Task 1.4: Pass account_equity from paper_trader
**File:** `trading_app/paper_trader.py`
**What:** Add `--account-equity` CLI arg (default 0 = disabled). Pass to ExecutionEngine constructor:
```python
engine = ExecutionEngine(portfolio, cost_spec, ...,
                         account_equity=args.account_equity,
                         risk_per_trade_pct=args.risk_per_trade_pct)
```
**Default behavior:** `account_equity=0` → sizing disabled, `contracts=1`, identical to today.

#### Task 1.5: Tests for vol-adjusted sizing integration
**File:** `tests/test_trading_app/test_execution_engine.py`
**Tests:**
1. `test_sizing_disabled_when_no_equity`: `account_equity=0` → `contracts=1` (regression guard)
2. `test_sizing_computes_contracts_from_equity`: `account_equity=25000, risk_pct=2, risk_points=10, point_value=10` → `contracts=5`
3. `test_sizing_applies_vol_scalar`: High ATR → fewer contracts, low ATR → more
4. `test_sizing_rejects_when_risk_too_large`: Risk exceeds budget → entry rejected
5. `test_sizing_before_risk_manager`: Verify contracts computed before RiskManager.can_enter()

#### Task 1.6: Drift check for position sizing
**File:** `pipeline/check_drift.py`
**What:** Add check: `compute_position_size_vol_scaled` must exist in `portfolio.py` and be called from `execution_engine.py` when `account_equity > 0`.

### Phase 1 Verification
```bash
python -m pytest tests/test_trading_app/test_execution_engine.py -x -q
python -m pytest tests/test_trading_app/test_portfolio.py -x -q
python pipeline/check_drift.py
# Regression: replay with default account_equity=0 must match baseline
python trading_app/paper_trader.py --instrument MGC --start 2024-01-01 --end 2024-12-31
```

---

## Phase 2: T80 Extension to 6 Remaining Sessions

### Problem
6 sessions have `EARLY_EXIT_MINUTES = None`: US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, NYSE_CLOSE, BRISBANE_1025.

T80 early exit is **already wired** into both:
- `execution_engine.py` (lines 938-951) — fires at threshold, exits if MTM < 0
- `outcome_builder.py` — `_annotate_time_stop()` bakes T80 into outcomes

Only missing piece: **the actual T80 values** from winner_speed analysis.

### Implementation Tasks

#### Task 2.1: Run winner_speed analysis on all sessions
**Script:** `research/research_winner_speed.py`
**Command:**
```bash
python research/research_winner_speed.py
```
**Output:** `research/output/winner_speed_summary.csv` with T50/T80/T90 per session.
**Action:** Read output, record T80 values for the 6 sessions with None.

#### Task 2.2: Update EARLY_EXIT_MINUTES in config.py
**File:** `trading_app/config.py` (lines 686-711)
**What:** Replace `None` with computed T80 values for sessions where:
1. Winner speed pattern exists (80% of winners hit target by minute X)
2. N >= 30 winners (statistical significance)
3. Post-T80 trades are net negative (time exit = free improvement)

**Gate:** If a session has <30 winners or no clear T80 pattern, leave as None.

#### Task 2.3: Rebuild outcomes for affected instruments
**Command:**
```bash
# For each instrument that has strategies on newly-T80'd sessions:
bash scripts/tools/run_rebuild_with_sync.sh MGC
bash scripts/tools/run_rebuild_with_sync.sh MNQ
bash scripts/tools/run_rebuild_with_sync.sh MES
bash scripts/tools/run_rebuild_with_sync.sh M2K
```
**Note:** Full rebuild chain includes outcome_builder → strategy_discovery → strategy_validator → build_edge_families → health_check → sync_pinecone.

#### Task 2.4: Validate T80 improvement
**What:** Compare paper_trader results before/after T80 extension:
- Total R change
- Number of early_exit trades
- Session-by-session breakdown
- Ensure no regression on sessions that already had T80

### Phase 2 Verification
```bash
python pipeline/check_drift.py
python -m pytest tests/ -x -q
python pipeline/health_check.py
```

---

## Phase 3: Partial Profit Taking Research (RESEARCH ONLY)

### Problem
No scale-out logic exists. MAE/MFE data shows 45% of losses went 0.5R favorable first, 27% went 1.0R favorable. Carver warns this "reduces geometric growth" — need data to decide.

### Why Research First
- Structural change to outcome_builder (binary exit → bar-by-bar simulation)
- Carver's geometric growth warning is real
- Transaction costs double (two exits per trade)
- Decision gate: only implement if improvement is statistically significant

### Implementation Tasks

#### Task 3.1: Design scale-out simulation
**File:** New `research/research_partial_profit.py`
**Logic:**
```
For each outcome in orb_outcomes:
    1. Replay bars_1m from fill to exit
    2. Track bar-by-bar price path
    3. At first bar where MFE >= 1.0R:
        - Record "partial exit" at 1.0R (50% of position)
        - Move stop to 0.5R above entry
        - Continue tracking remaining 50%
    4. Compute composite pnl_r:
        pnl_r = 0.5 * 1.0R + 0.5 * remaining_exit_r
    5. Compare to baseline pnl_r (no partial)
```

#### Task 3.2: Implement and run simulation
**Instruments:** All 4 active (MGC, MNQ, MES, M2K)
**RR targets:** All validated (1.0, 1.5, 2.0, 2.5, 3.0)
**Output:** Per-session comparison table:
- Baseline total R vs partial total R
- Win rate change
- Average trade R change
- Geometric growth comparison (Carver's concern)

#### Task 3.3: Statistical validation
**Tests:**
- Paired t-test on daily P&L (baseline vs partial)
- BH FDR correction across session × instrument × RR grid
- Year-by-year stability (not just aggregate)
- Transaction cost impact (double exits)

#### Task 3.4: Decision gate
**GO criteria (ALL must hold):**
1. Paired t-test p < 0.05 after BH FDR
2. Improvement in 4+ of 5 years tested
3. Geometric growth impact < 10% reduction
4. Net R improvement after doubled transaction costs

**NO-GO:** If any criterion fails, document as research finding and close.

#### Task 3.5: Production implementation (conditional)
**Only if GO:**
- Modify `outcome_builder.py` to support `scale_out_at_r` parameter
- Add `partial_pnl_r` column to `orb_outcomes` schema
- Rebuild outcomes chain
- Update paper_trader to handle partial exits

### Phase 3 Verification
```bash
# Research validation
python research/research_partial_profit.py --instrument MGC
# Statistical checks are built into the script
```

---

## Dependency Graph

```
Phase 1 (Vol Sizing)      Phase 2 (T80 Extension)
    │                          │
    ├─ Task 1.1               ├─ Task 2.1 (run analysis)
    ├─ Task 1.2               ├─ Task 2.2 (update config)
    ├─ Task 1.3 (depends 1.2) ├─ Task 2.3 (rebuild, depends 2.2)
    ├─ Task 1.4 (depends 1.2) └─ Task 2.4 (validate, depends 2.3)
    ├─ Task 1.5 (depends 1.3)
    └─ Task 1.6
                               Phase 3 (Partial Profit Research)
    Phases 1 & 2 are              │
    INDEPENDENT — can             ├─ Task 3.1 (design)
    run in parallel               ├─ Task 3.2 (implement + run)
                                  ├─ Task 3.3 (stats, depends 3.2)
                                  ├─ Task 3.4 (decision, depends 3.3)
                                  └─ Task 3.5 (conditional, depends 3.4)
```

Phases 1 and 2 have zero overlap — different files, different concerns. Can be done in parallel or either order.

Phase 3 is independent but should come after 1+2 so the baseline for comparison includes vol-sizing and T80 improvements.

---

## What We're NOT Doing (and Why)

| Item | Reason |
|------|--------|
| RR by regime | Chan warns curve-fitting. BH FDR already handles RR grid search in strategy_discovery. |
| CB by regime | Parameter explosion (CB × ORB_size × ATR × session). Per-cell N too small. |
| Correlation-aware concurrency | HIGH complexity (execution engine rewrite). Current max_concurrent=3 is conservative. |
| Conditional breakeven | Murray calls breakeven stops "most common mistake." No prior evidence it works. |
| Multi-instrument portfolio | Architectural constraint (Portfolio.instrument is singular). Separate portfolios already work independently. |
| Daily loss tiering | Current -5R hard halt is more conservative than tiered (-2R reduce, -3R halt). Not broken. |
| Kelly criterion | Chan: "optimal but unstable" for small samples. Fixed 2% is safer. |

---

## Verification Checklist (Post-All-Phases)

- [ ] `python -m pytest tests/ -x -q` — all tests pass
- [ ] `python pipeline/check_drift.py` — all drift checks pass
- [ ] `python pipeline/health_check.py` — health check green
- [ ] Paper trader replay with `account_equity=0` matches pre-change baseline (regression guard)
- [ ] Paper trader replay with `account_equity=25000` shows sensible contract counts
- [ ] T80 values backed by winner_speed research output
- [ ] Memory files updated with findings
- [ ] Pinecone synced
