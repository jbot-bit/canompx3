# Bloomey Statistical Hardening — Design & Implementation Plan

**Date:** 2026-03-08
**Trigger:** Mr. Bloomey project review grounded against docs/reference/ literature (15 PDFs)
**Grade:** B- → target A-
**Spec:** Extends `docs/specs/STATISTICAL_HARDENING.md` (FIX 4-6 already done)

## Problem Statement

The project computes DSR, FST hurdle, and BH FDR correctly but does NOT enforce them
in the validation pipeline. Walk-forward + yearly + stress-test is a strong heuristic
but does not substitute for proper multiple testing correction when searching ~2,376
combinations per instrument. Additionally, the discovery grid includes E3 (soft-retired,
0/50 FDR survivors) and 0.75× tight stop (16/228 BH FDR survivors), inflating the trial
count unnecessarily.

## Reconciliation with Existing Spec

| Spec Item | Status | Bloomey Alignment |
|-----------|--------|-------------------|
| FIX 1-3 (ML wrong-fit) | NOT DONE | Out of Bloomey scope (ML-specific) |
| FIX 4 (DSR Mertens) | DONE | Prerequisite satisfied |
| FIX 5 (Record K) | DONE | Prerequisite satisfied |
| FIX 6 (FST hurdle) | DONE | Prerequisite satisfied |
| FIX 7 (BHY stress test) | NOT DONE | Extended: enforce BH FDR at discovery |
| FIX 8 (Rejection rate log) | NOT DONE | Included as Phase 4 |
| FIX 9 (WFE) | NOT DONE | Bloomey #4 |
| FIX 10 (ML threshold) | NOT DONE | Out of Bloomey scope |
| FIX 11 (Synthetic null) | NOT DONE | Deferred to future |
| FIX 12 (PBO) | NOT DONE | Bloomey #5 |

## PHASE 0: Grid Reduction — CANCELLED

**Decision:** Keep E3 in grid, keep 0.75× stop in grid. Let the new statistical gates
(DSR/FST) filter on merit instead of manually excluding.

**Rationale:**
- E3: 0/50 FDR survivors, but removing it pre-hoc is a form of reverse cherry-picking.
  If DSR/FST gates kill all E3 strategies, that's confirmation. If one survives, we learn.
  E3 adds only ~264 combos (~10% of grid) — modest n_trials inflation.
- 0.75× stop: Kill ratio is 10:1 to 20:1 (structural, not curve-fit). Deliberately
  designed NOT to inflate n_trials (correlated overlay on same hypothesis). Has 0 validated
  strategies currently because it hasn't been through validation yet. Removing it before
  giving it a chance is premature.
- The whole point of statistical gates is to let the math decide, not our priors.

---

## Phases

### PHASE 1: Enforce Statistical Gates

**Why:** DSR, FST, and BH FDR are computed but not used for decisions. This is the #1
Bloomey finding. "You built a fire suppression system and left the sprinkler valves closed."

> **AMENDED (2026-03-08, commit bc03c02):** P4c (DSR) and P4d (FST) demoted to
> informational-only logging. Root causes: (1) DSR uses raw n_trials without N_eff
> correlated-trial adjustment (BLP 2014 Appendix A.3), inflating the threshold;
> (2) FST hurdle is in Z-score units but compared against per-trade Sharpe (unit mismatch,
> BLP 2018 Theorem 1). Both gates rejected 100% of strategies on first production run.
> BH FDR at discovery remains as the primary multiple-testing defense. See risk table
> line 255 — this was the anticipated failure mode with prescribed mitigation.

#### 1a: DSR ~~Hard~~ Informational Gate (Phase 4c in validation)

**File:** `trading_app/strategy_validator.py` → `validate_strategy()`
**Location:** After Phase 4 (stress test), before Phase 5 (optional Sharpe)
**Logic:**
```python
# Phase 4c: Deflated Sharpe gate (Bailey & Lopez de Prado 2014)
sharpe_haircut = row.get("sharpe_haircut")
if sharpe_haircut is not None and sharpe_haircut < 0:
    return ("REJECTED", f"Phase 4c: DSR below noise floor (haircut={sharpe_haircut:.4f})", [])
```
**Rationale:** DSR < 0 means the observed Sharpe doesn't exceed what K random strategies
would produce. Strategy is indistinguishable from noise.

#### 1b: FST ~~Hurdle~~ Informational Gate (Phase 4d in validation)

**File:** `trading_app/strategy_validator.py` → `validate_strategy()`
**Location:** After Phase 4c
**Logic:**
```python
# Phase 4d: False Strategy Theorem hurdle (Lopez de Prado 2018)
fst_hurdle = row.get("fst_hurdle")
if fst_hurdle is not None and row.get("sharpe_ratio", 0) < fst_hurdle:
    return ("REJECTED", f"Phase 4d: Sharpe {row['sharpe_ratio']:.4f} < FST hurdle {fst_hurdle:.4f}", [])
```
**Rationale:** FST hurdle = expected max per-trade Sharpe from K trials with zero skill.
Below this = noise.

#### 1c: BH FDR at Discovery (pre-validation filter)

**File:** `trading_app/strategy_discovery.py` → `run_discovery()`
**Location:** After grid search generates all_strategies, before DB write
**Logic:**
```python
# Apply BH FDR correction to discovery p-values
from trading_app.strategy_validator import benjamini_hochberg
p_pairs = [(s["strategy_id"], s["p_value"]) for s in all_strategies if s.get("p_value") is not None]
fdr_results = benjamini_hochberg(p_pairs, alpha=0.05)
for s in all_strategies:
    fdr = fdr_results.get(s["strategy_id"], {})
    s["fdr_significant_discovery"] = fdr.get("fdr_significant", False)
    s["fdr_adjusted_p_discovery"] = fdr.get("adjusted_p", None)
```
**Schema:** Add `fdr_significant_discovery BOOLEAN` and `fdr_adjusted_p_discovery DOUBLE`
to experimental_strategies (init_db.py). Distinct from the existing `fdr_significant`
column on validated_setups (which is post-validation BH FDR).

**NOT a hard gate:** BH FDR at discovery is informational. The DSR/FST gates in
validation are the hard filters. This stores the correction for auditability.
Rationale: BH FDR assumes independence; our strategies share outcomes. DSR already
accounts for K trials with non-normality correction. Using BH FDR as a hard gate
on top of DSR would be double-counting the multiple testing penalty.

**Verify:** `python pipeline/check_drift.py` + `pytest tests/ -x -q`

---

### PHASE 2: Walk-Forward Efficiency

**File:** `trading_app/walkforward.py` → `run_walkforward()`
**Location:** During window generation, compute IS metrics alongside OOS

**Logic:**
For each test window, also compute metrics on IS period (all outcomes before window_start):
```python
is_outcomes = [o for o in all_outcomes if o["trading_day"] < window_start]
is_metrics = compute_metrics(is_outcomes) if len(is_outcomes) >= 15 else None
window["is_exp_r"] = is_metrics["expectancy_r"] if is_metrics else None
```

After all windows:
```python
valid_is = [w for w in valid_windows if w.get("is_exp_r") and w["is_exp_r"] > 0]
if valid_is:
    mean_oos = sum(w["test_exp_r"] for w in valid_is) / len(valid_is)
    mean_is = sum(w["is_exp_r"] for w in valid_is) / len(valid_is)
    wfe = mean_oos / mean_is if mean_is > 0 else None
```

**Store:** Add `wfe` field to WalkForwardResult dataclass. Write to validated_setups.
**Schema:** Add `wfe DOUBLE` to validated_setups in init_db.py.
**Threshold:** WFE > 0.50 = healthy (Pardo). Store but don't gate (informational first).

**Verify:** `pytest tests/test_trading_app/test_walkforward.py -x -q`

---

### PHASE 3: Yearly Robustness Relaxation

**File:** `trading_app/strategy_validator.py` → `validate_strategy()`
**Current:** `min_years_positive_pct: float = 1.0` (ALL years positive)
**Change:** Default to `0.75` (75% of years positive)

**Also:** `run_validation()` passes this parameter — update its default too.

**Rationale:** Fitschen (Building Reliable Trading Systems): 17/20 top CTAs had an
entire year with no gain. ALL-years-positive is stricter than professional fund standards.
75% is still strict (4/5 years or 6/8 years must be positive).

**Blast radius:**
- validate_strategy() already accepts this as a parameter
- run_validation() passes it through
- No other code references this threshold
- Tests may assert on specific rejection messages — update

**Verify:** `pytest tests/test_trading_app/test_strategy_validator.py -x -q`

---

### PHASE 4: Rejection Rate Logging (Spec FIX 8)

**New table:** `validation_run_log` in init_db.py
```sql
CREATE TABLE IF NOT EXISTS validation_run_log (
    run_id TEXT PRIMARY KEY,
    run_ts TIMESTAMPTZ DEFAULT now(),
    instrument TEXT NOT NULL,
    orb_minutes INTEGER NOT NULL,
    combos_tested INTEGER,
    phase1_survivors INTEGER,    -- sample size gate
    phase2_survivors INTEGER,    -- post-cost ExpR gate
    phase3_survivors INTEGER,    -- yearly robustness gate
    phase4_survivors INTEGER,    -- stress test gate
    phase4c_survivors INTEGER,   -- DSR gate
    phase4d_survivors INTEGER,   -- FST gate
    wf_survivors INTEGER,        -- walk-forward gate
    fdr_significant_count INTEGER,
    final_validated INTEGER,
    rejection_rate DOUBLE,       -- 1 - (final / combos_tested)
    notes TEXT
)
```

**Write:** At end of run_validation(), log one row per run.
**Read:** By audit scripts, Pinecone sync, health checks.

**Verify:** `pytest tests/ -x -q`

---

### PHASE 5: Approximate PBO (Spec FIX 12)

**New function:** `compute_pbo()` in walkforward.py or new module
**Paper:** Bailey et al (2014) — Probability of Backtest Overfitting

**Simplified approach (permutation-based):**
1. Take all outcomes for an instrument × session × aperture
2. Partition into S=8 time blocks (chronological)
3. For each of C(8,4)=70 combinatorial train/test splits:
   a. Rank all strategies by IS performance (train blocks)
   b. Measure OOS performance (test blocks) of the IS-best strategy
4. PBO = fraction of 70 splits where IS-best strategy has negative OOS
5. PBO > 0.50 = likely overfit

**Scope:** Per edge family, not per individual strategy. This measures whether the
SELECTION PROCESS (picking the best strategy from the family) is robust.

**Store:** `pbo DOUBLE` on edge_families table.

**This is the most complex item.** Design in detail during implementation.

**Verify:** Unit tests with synthetic data (known-overfit vs known-robust scenarios)

---

### PHASE 6: Full Rebuild + Verification

After all code changes, rebuild all 4 instruments × 3 apertures:

```bash
# For each instrument: MGC, MNQ, MES, M2K
python trading_app/outcome_builder.py --instrument $INST --orb-minutes 5
python trading_app/outcome_builder.py --instrument $INST --orb-minutes 15
python trading_app/outcome_builder.py --instrument $INST --orb-minutes 30
python trading_app/strategy_discovery.py --instrument $INST --orb-minutes 5
python trading_app/strategy_discovery.py --instrument $INST --orb-minutes 15
python trading_app/strategy_discovery.py --instrument $INST --orb-minutes 30
python trading_app/strategy_validator.py --instrument $INST
python scripts/migrations/retire_e3_strategies.py
python scripts/tools/build_edge_families.py
```

**Verification checklist:**
- [ ] Drift checks pass: `python pipeline/check_drift.py`
- [ ] All tests pass: `pytest tests/ -x -q`
- [ ] Behavioral audit: `python scripts/tools/audit_behavioral.py`
- [ ] Strategy counts: before vs after (expect ~30-50% reduction from new gates)
- [ ] No INVALID strategies in validated_setups
- [ ] No negative ExpR in validated_setups
- [ ] WFE distribution: median should be 0.40-0.70 (Pardo healthy range)
- [ ] Rejection rate logged per instrument
- [ ] Pinecone sync: `python scripts/tools/sync_pinecone.py`

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DSR/FST gates reject ALL strategies for an instrument | Low | High | Check distribution of DSR values before enforcing; soft-launch as logging-only first |
| Yearly relaxation admits noise strategies | Medium | Medium | DSR/FST gates catch these |
| PBO computation too slow | Medium | Low | Limit to edge family heads only |
| Full rebuild reveals zero validated strategies for M2K | Medium | Medium | M2K already marginal (12-15 strategies); accept if gates reject all |
| E3 strategies all die under new gates | High | None | Expected outcome; confirms soft-retirement decision |
| 0.75× stop strategies mostly die under new gates | Medium | Low | Structural kill ratio means survivors are real |

## Rollback Plan

Each phase is independently revertable:
- Phase 1: Remove Phase 4c/4d from validate_strategy()
- Phase 2: Remove WFE computation (informational only)
- Phase 3: Change min_years_positive_pct back to 1.0
- Phase 4: Drop validation_run_log table
- Phase 5: Drop PBO computation

No destructive schema changes. All additions are additive columns/tables.
No grid reduction — E3 and 0.75× stop stay in grid, filtered by gates.

## Guardian Prompt Assessment

- **ENTRY_MODEL_GUARDIAN:** No entry model changes. E3 stays in grid. No entry model
  LOGIC changes. Guardian not triggered.

- **PIPELINE_DATA_GUARDIAN:** No changes to ingestion, aggregation, or feature computation.
  Changes are validation-layer only (strategy_validator.py, walkforward.py).
  No data flow changes. Guardian not triggered.
