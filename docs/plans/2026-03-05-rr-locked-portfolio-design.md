# RR-Locked Portfolio Architecture

**Date:** 2026-03-05
**Status:** Plan (v2 — post-audit)
**Scope:** Lock one fixed RR target per family, propagate through all downstream systems

---

## Problem Statement

`_load_best_regime_variant()` in `live_config.py` picks RR via `ORDER BY expectancy_r DESC LIMIT 1`. This creates four concrete failures:

1. **Non-stationary track record** — RR can change every DB rebuild, so historical performance is mixed-RR and meaningless for Kelly sizing (Van Tharp).
2. **ML population mismatch** — `meta_label.py` captures `training_rr` from the first row of session data (line 322). In single_config mode this is deterministic (all rows share the same RR), but the source is implicit rather than explicit. If live resolves a different RR, the ML model predicts on an out-of-population trade.
3. **No RR in spec** — `LiveStrategySpec` has NO `rr_target` field. RR is resolved at runtime, invisible to the trader.
4. **Mixed-RR Sharpe is meaningless** — Bailey (2014): non-stationary returns invalidate Sharpe and all derived performance metrics.

## Statistical Evidence (Jobson-Korkie Tests, Mar 5 2026)

| Finding | Test | Result |
|---------|------|--------|
| Per-trade Sharpe CANNOT distinguish RR levels | Jobson-Korkie (rho=0.7) | 193/198 families p>0.05 (97%) |
| MaxDD IS different across RR levels | Kruskal-Wallis | p<0.000001 |
| All pairwise RR comparisons significant for DD | Mann-Whitney U | All p<0.000001 |
| RR1.0 median MaxDD=13.8R vs RR4.0=25.8R | Descriptive | 2x difference |

**Note:** The 198-family count above used the OLD 4-column grouping `(instrument, orb_label, filter_type, entry_model)`. The correct 6-column grouping (adding `orb_minutes`, `confirm_bars`) produces ~490 families. The statistical conclusions hold — JK test results are per-family and don't change with grouping granularity.

**Implication:** Since Sharpe can't tell RR levels apart, but DD can, the optimal selection criterion is: among statistically-equal Sharpes (JK p>0.05 vs best), pick lowest MaxDD. This is the **SharpeDD criterion**.

## SharpeDD Selection Result

*(To be re-computed after fixing grouping to include `orb_minutes` + `confirm_bars`)*

Preliminary distribution from 4-column grouping (directionally correct):

| Locked RR | % Families | Method |
|-----------|-----------|--------|
| RR1.0 | ~71% | SharpeDD / MAX_SHARPE / ONLY_RR |
| RR1.5 | ~13% | SharpeDD / MAX_SHARPE |
| RR2.0 | ~11% | SharpeDD / MAX_SHARPE |
| RR2.5+ | ~5% | SharpeDD / MAX_SHARPE |

## Two-Layer Architecture

- **Layer 1 (System, permanent):** Lock RR per family via SharpeDD — stored in `family_rr_locks` DB table. This never changes based on account size.
- **Layer 2 (Deployment, runtime):** Account DD ceiling determines position sizing and which families are tradeable. $2K, $6K, $20K accounts all use the SAME locked RRs, different sizing.

---

## Audit Findings (Mar 5 2026)

Three parallel audits ran against v1 of this plan: M2.5 improvements/bugs audit, deep code-vs-plan verification, and data/algorithm edge case audit. The following issues were found and incorporated into v2:

### Critical Fixes Applied

1. **Primary key too narrow** — v1 used `(instrument, orb_label, filter_type, entry_model)`. Data audit found 126/250 groups were contaminated (mixing 5m/15m/30m apertures and CB1-CB5). **Fixed:** PK is now `(instrument, orb_label, filter_type, entry_model, orb_minutes, confirm_bars)`.

2. **`rolling_portfolio.py` missing from plan** — CORE path in `build_live_portfolio()` tries `load_rolling_validated_strategies()` FIRST (line 360-376). That function uses `ORDER BY rv.expectancy_r DESC LIMIT 1` independently of `_load_best_regime_variant()`. **Fixed:** Added to Phase 2.

3. **`paper_trader.py` uses `portfolio.py:build_portfolio()`, not `build_live_portfolio()`** — Phase 6 comparison had no mechanism to use locked RR. **Fixed:** `portfolio.py` query added to Phase 2, paper trader clarified in Phase 6.

### High-Priority Fixes Applied

4. **"Family hash includes rr_target" was FALSE** — Edge families' `family_hash` is `{instrument}_{orb_min}m_{md5_of_trade_days}`. RR is NOT in the hash. **Fixed:** Removed incorrect claim.

5. **RR mismatch → ERROR contradicts fail-open philosophy** — Both M2.5 and code audit flagged this. **Fixed:** Aggressive mismatch → error+skip (unsafe direction). Conservative mismatch → warning+proceed (safe direction).

6. **`gen_playbook.py` reads CSV, not DB** — Plan assumed direct DB access. **Fixed:** Added CSV generation filtering step.

7. **`generate_trade_sheet.py` has independent query function** — `_load_best_by_expr()` is a separate copy of variant selection logic. **Fixed:** Clarified as separate function needing its own JOIN.

---

## Implementation Plan

### Phase 1: DB Schema + Selection Script
**Goal:** Create `family_rr_locks` table and populate it.

**Task 1.1: Add `family_rr_locks` table to `init_db.py`**
- File: `pipeline/init_db.py`
- Add new schema constant `FAMILY_RR_LOCKS_SCHEMA` after line ~108 (after `PROSPECTIVE_SIGNALS_SCHEMA`)
- Columns: `instrument TEXT, orb_label TEXT, filter_type TEXT, entry_model TEXT, orb_minutes INTEGER, confirm_bars INTEGER, locked_rr REAL, method TEXT, sharpe_at_rr REAL, maxdd_at_rr REAL, n_at_rr INTEGER, expr_at_rr REAL, tpy_at_rr REAL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- **Primary key: `(instrument, orb_label, filter_type, entry_model, orb_minutes, confirm_bars)`**
- Add CREATE TABLE in `init_db()` after line ~414 (after prospective_signals)
- Add `"family_rr_locks"` to the DROP TABLE list at line ~275 in force mode

**Task 1.2: Create `scripts/tools/select_family_rr.py`**
- New file implementing SharpeDD criterion
- Logic: For each family **(instrument, orb_label, filter_type, entry_model, orb_minutes, confirm_bars)**:
  1. Query all RR levels from `validated_setups WHERE status='active' AND entry_model IN ('E1','E2')`
  2. If only 1 RR → method=ONLY_RR, lock that RR
  3. If 2+ RR → run Jobson-Korkie test (rho=0.7) against best Sharpe
  4. Among candidates with JK p>0.05: pick lowest MaxDD → method=SHARPE_DD
  5. If best Sharpe is also lowest DD among candidates → method=MAX_SHARPE
- Write results to `family_rr_locks` table (DELETE + INSERT pattern)
- Print summary: locked RR distribution, method distribution, per-instrument breakdown
- **Note:** `max_drawdown_r` is stored as POSITIVE values in validated_setups (min=2.3, max=94.9). No sign flip needed.
- Core algorithm structure in `scripts/tools/rr_selection_analysis.py` but must be fixed for 6-column grouping

**Task 1.3: Test `family_rr_locks` integrity**
- New file: `tests/test_rr_selection.py`
- Test: Every active family (by 6-column key) in `validated_setups` has exactly one row in `family_rr_locks`
- Test: `locked_rr` is always one of the family's available RR levels
- Test: SharpeDD selection logic produces correct results for known test cases (single RR, multi RR where best Sharpe has lowest DD, multi RR where lower DD is picked)
- Test: `method` field is correct (ONLY_RR, MAX_SHARPE, SHARPE_DD)

### Phase 2: Enforce Locked RR in All Variant Resolution Paths
**Goal:** Every code path that picks a variant uses locked RR.

**Task 2.1: Add `rr_target` to `LiveStrategySpec`**
- File: `trading_app/live_config.py`
- Add `rr_target: Optional[float] = None` field to `LiveStrategySpec` dataclass (after `regime_gate`, line ~46)
- LIVE_PORTFOLIO stays instrument-agnostic — `rr_target=None` in spec, resolved at build time from `family_rr_locks`

**Task 2.2: Modify `_load_best_regime_variant()` to enforce locked RR**
- File: `trading_app/live_config.py`, lines 176-216
- Current: `ORDER BY vs.expectancy_r DESC NULLS LAST LIMIT 1`
- New: JOIN `family_rr_locks frl ON vs.instrument = frl.instrument AND vs.orb_label = frl.orb_label AND vs.filter_type = frl.filter_type AND vs.entry_model = frl.entry_model AND vs.orb_minutes = frl.orb_minutes AND vs.confirm_bars = frl.confirm_bars` and add `WHERE vs.rr_target = frl.locked_rr`
- **Note:** After JOIN+WHERE, there may still be multiple rows (different apertures/CB for the same family spec). Keep `ORDER BY vs.expectancy_r DESC LIMIT 1` as tiebreaker among locked-RR rows.
- If no row found after JOIN → fail-closed (log error, skip family)

**Task 2.3: Modify `load_rolling_validated_strategies()` in `rolling_portfolio.py`**
- File: `trading_app/rolling_portfolio.py`, line ~432
- **THIS WAS MISSING FROM v1.** CORE path tries rolling eval FIRST. If rolling eval returns a non-locked RR, the entire lock is bypassed.
- Current: `ORDER BY rv.expectancy_r DESC LIMIT 1`
- New: JOIN `family_rr_locks` and add `WHERE rv.rr_target = frl.locked_rr`
- Same pattern as Task 2.2

**Task 2.4: Modify `build_portfolio()` in `portfolio.py`**
- File: `trading_app/portfolio.py`, line ~290
- **THIS WAS MISSING FROM v1.** Paper trader uses this function, not `build_live_portfolio()`.
- Current: `ORDER BY vs.expectancy_r DESC`
- New: JOIN `family_rr_locks` and `WHERE vs.rr_target = frl.locked_rr`
- Also check `load_validated_strategies()` at line ~318 for same pattern

**Task 2.5: Update `build_live_portfolio()` to propagate locked RR**
- File: `trading_app/live_config.py`, lines 332-538
- CORE path (line ~390-420): After resolving variant, `rr_target` comes from query result which now enforces locked RR via JOIN
- REGIME path (line ~440-470): Same — verify rr_target from locked RR
- Dollar gate (line ~500-520): Uses `rr_target * point_value * cost`. Verify locked RR flows through.

**Task 2.6: Test locked RR enforcement**
- File: `tests/test_trading_app/test_live_config.py`
- Test: `LiveStrategySpec` has `rr_target` field
- Test: `_load_best_regime_variant()` returns locked RR, not max ExpR
- Test: If `family_rr_locks` has no row for a family, variant resolution fails (fail-closed)
- Test: Dollar gate uses locked RR
- Test: Rolling portfolio variant uses locked RR

### Phase 3: ML Fixes
**Goal:** Ensure ML trains and predicts on the locked RR population.

**Task 3.1: Source `training_rr` explicitly in `meta_label.py`**
- File: `trading_app/ml/meta_label.py`, line 322
- Current: `training_rr = float(meta_all["rr_target"].iloc[session_indices[0]])` — deterministic in single_config mode (all rows share same RR), but implicit
- New: Source from `family_rr_locks` query (cached once at session start, not per-batch) or from the config parameter already passed in
- Store in bundle at line 518 and joblib at line 612 (already done, just need correct source)

**Task 3.2: Fix RR guard in `predict_live.py`**
- File: `trading_app/ml/predict_live.py`, lines 263-282
- **Aggressive** (trade_rr > training_rr): `logger.error()` + return `MLPrediction(p_win=0.5, take=False)` — model trained on easier target, prediction unreliable for harder target
- **Conservative** (trade_rr < training_rr): `logger.warning()` + proceed with prediction — model P(win) is understated (safe direction), acceptable
- **None** (legacy models): skip guard, backward compatible
- **Exact match**: proceed normally

**Task 3.3: Test ML RR consistency**
- File: `tests/test_trading_app/test_ml/test_predict_live.py`
- Test: Aggressive RR mismatch returns take=False (not fail-open)
- Test: Conservative RR mismatch logs warning and proceeds
- Test: `training_rr is None` (legacy) → skip guard (backward compat)
- Test: Exact match → proceed normally

### Phase 4: Downstream Updates
**Goal:** All reporting and integrity checks use locked RR.

**Task 4.1: Update `generate_trade_sheet.py`**
- File: `scripts/tools/generate_trade_sheet.py`
- Function `_load_best_by_expr()` (lines 183-223) is a **separate copy** of variant selection logic — NOT a call to `_load_best_regime_variant()`
- Replace `ORDER BY expectancy_r DESC` at line ~213 with JOIN to `family_rr_locks`
- Show locked RR and method in output

**Task 4.2: Update `gen_playbook.py`**
- File: `scripts/tools/gen_playbook.py`
- **Note:** gen_playbook reads from CSV (`research/output/_playbook_data.csv`), NOT the database
- Either: (a) filter CSV at generation time to only include locked RR rows, or (b) have gen_playbook query `family_rr_locks` from DB to filter
- Recommended: (a) — modify the CSV generation step (wherever `_playbook_data.csv` is created) to JOIN `family_rr_locks` and only output locked-RR strategies

**Task 4.3: Add drift checks for `family_rr_locks`**
- File: `pipeline/check_drift.py`
- New check #59: Every family in `LIVE_PORTFOLIO` (resolved per instrument) must have a row in `family_rr_locks`
- New check #60: Every `locked_rr` in `family_rr_locks` must exist in `validated_setups` for that family
- New check #61: `family_rr_locks.updated_at` must be within 30 days (staleness guard)

**Task 4.4: Add `select_family_rr.py` to rebuild chain**
- Update `scripts/tools/run_rebuild_with_sync.sh` — add `select_family_rr.py` after `strategy_validator.py` and before `build_edge_families.py`
- Update `validation-workflow.md` rebuild chain documentation

**Task 4.5: Verify downstream propagation (audit only, no code change expected)**
- `trading_app/execution_engine.py` — Uses `trade.strategy.rr_target` for target price. No change needed IF upstream feeds correct locked RR.
- `scripts/tools/build_edge_families.py` — Family hash is trade-day based (`{instrument}_{orb_min}m_{md5}`), does NOT include `rr_target`. RR locking is orthogonal to family hashing. No change needed.
- `trading_app/strategy_validator.py` — Validates all RR levels independently. No change needed — SharpeDD selection happens AFTER validation.
- Display-only consumers (no execution impact, low priority): `pipeline/dashboard.py`, `trading_app/ai/sql_adapter.py`, `ui/db_reader.py` — all use `ORDER BY expectancy_r DESC` for display. Note for future cleanup but not blocking.

### Phase 5: Validation
**Goal:** Prove the system works end-to-end.

**Task 5.1: Run full test suite**
```bash
python -m pytest tests/ -x -q
```

**Task 5.2: Run drift checks**
```bash
python pipeline/check_drift.py
```

**Task 5.3: Run behavioral audit**
```bash
python scripts/tools/audit_behavioral.py
```

**Task 5.4: Populate `family_rr_locks` for all instruments**
```bash
python scripts/tools/select_family_rr.py
```

**Task 5.5: Build live portfolio with locked RR and verify**
```bash
python -m trading_app.live_config --db-path gold.db
```

### Phase 6: ML Retrain + Paper Trader (Post-Implementation)
**Goal:** Retrain ML with locked RR populations and validate.

**Task 6.1: Retrain ML for all 4 instruments**
- After `family_rr_locks` populated
- Each model trains on exactly the locked RR population
- Verify `training_rr` in model bundle matches `family_rr_locks`

**Task 6.2: Paper trader comparison**
- Build two portfolios explicitly via `build_live_portfolio()` or `build_portfolio()` (both now enforce locked RR)
- For "old" comparison: temporarily remove `family_rr_locks` JOIN to get variable-RR portfolio
- Compare on 2025 data: total trades, PnL, Sharpe, MaxDD
- Expected: similar PnL (Sharpe was indistinguishable), lower DD (by design)

---

## Files Modified (Summary)

| File | Change | Phase |
|------|--------|-------|
| `pipeline/init_db.py` | Add `family_rr_locks` table schema | 1 |
| `scripts/tools/select_family_rr.py` | NEW — compute and store locked RR per family | 1 |
| `tests/test_rr_selection.py` | NEW — test SharpeDD selection logic | 1 |
| `trading_app/live_config.py` | Add `rr_target` to spec, enforce locked RR in `_load_best_regime_variant` | 2 |
| `trading_app/rolling_portfolio.py` | Enforce locked RR in `load_rolling_validated_strategies` | 2 |
| `trading_app/portfolio.py` | Enforce locked RR in `build_portfolio` / `load_validated_strategies` | 2 |
| `tests/test_trading_app/test_live_config.py` | Test locked RR enforcement | 2 |
| `trading_app/ml/meta_label.py` | Source `training_rr` explicitly from config/DB | 3 |
| `trading_app/ml/predict_live.py` | Aggressive mismatch → error+skip; conservative → warn+proceed | 3 |
| `tests/test_trading_app/test_ml/test_predict_live.py` | Test RR guard behavior | 3 |
| `scripts/tools/generate_trade_sheet.py` | JOIN `family_rr_locks` in `_load_best_by_expr()` | 4 |
| `scripts/tools/gen_playbook.py` | Filter to locked RR (via CSV source or DB query) | 4 |
| `pipeline/check_drift.py` | Add checks #59-61 for `family_rr_locks` | 4 |
| `scripts/tools/run_rebuild_with_sync.sh` | Add `select_family_rr.py` to rebuild chain | 4 |

## Files Verified (No Change Expected)

| File | Why |
|------|-----|
| `trading_app/execution_engine.py` | Uses `trade.strategy.rr_target` — correct if upstream is correct |
| `scripts/tools/build_edge_families.py` | Family hash is trade-day based, RR is orthogonal. No change. |
| `trading_app/strategy_validator.py` | Validates all RR levels — selection is post-validation |
| `pipeline/dashboard.py` | Display only — `ORDER BY expectancy_r DESC` for top-10 display |
| `trading_app/ai/sql_adapter.py` | MCP query templates — display only, not execution |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `family_rr_locks` stale after rebuild | Drift check #61 (staleness), added to rebuild chain (Task 4.4) |
| Family in LIVE_PORTFOLIO but not in `family_rr_locks` | Drift check #59, fail-closed in all variant resolution paths |
| ML models trained on wrong RR | `training_rr` from explicit source; aggressive mismatch → error+skip |
| LIVE_PORTFOLIO becomes instrument-specific | Option A avoids this — resolve at build time from DB |
| Rolling eval bypasses RR lock | Task 2.3 adds JOIN to `rolling_portfolio.py` |
| Paper trader uses different portfolio builder | Task 2.4 adds JOIN to `portfolio.py:build_portfolio()` |
| gen_playbook reads CSV not DB | Task 4.2 addresses CSV source filtering |
