# ML Methodology Fix — Design Doc

**Status:** APPROVED for implementation
**Date:** 2026-03-21
**Context:** ML meta-labeling system has 3 methodology FAILs (EPV, negative baselines, selection bias). Bootstrap resolution already FIXED (5K perms). This design fixes the remaining 3 to produce honest ML results.

**Key constraint:** Layer 1 (O5 RR1.0 raw baseline) is CLEAN (p<1e-9). Do NOT touch it.

---

## Problem Statement

The current ML system produces numbers that cannot be trusted because:

1. **EPV = 2.4** (need ≥10). 55 positives / 23 features. Model memorizes noise. (Peduzzi 1996, van Smeden 2019)
2. **Negative baselines.** ML trains on sessions where raw ExpR < 0. De Prado (AIFML Ch 3.6) requires positive-edge primary model.
3. **Selection bias.** Data loaded from `validated_setups` (pre-selected winners). Bootstrap tests on same subset. (White 2000)

All 5K bootstrap p-values (p=0.0016 to p=0.1190) are from the BROKEN system and will change after fixes.

---

## Fix Sequence: B → A → C → D

### Fix B: Replace Data Source (selection bias)
- Use `load_single_config_feature_matrix()` with `bypass_validated=True`
- Loads from `orb_outcomes` directly — all 12 sessions, all break days
- Bootstrap test must also use `bypass_validated=True` (currently False at line 89)
- Forces `skip_filter=True` (ML sees full population, learns filters from features)

**Files:** `meta_label.py` (change default call), `ml_bootstrap_test.py` (line 89)
**Academic basis:** White (2000) — test the full search space, not just survivors

### Fix A: Hard Gates Before Training (negative baselines + EPV)
Add 3 checks in `train_per_session_meta_label()` BEFORE the training loop:

1. **Baseline gate:** Compute raw session ExpR from full unfiltered data. If ExpR ≤ 0, skip session → `model_type="NONE", reason="negative_baseline_expr={value}"`
2. **EPV gate:** After feature selection (Fix C), check `n_positives / n_features ≥ 10`. If not, skip → `model_type="NONE", reason="epv_{value}_below_10"`
3. **Pre-registered sessions:** New `ML_PRE_REGISTERED_SESSIONS` in config.py. Only these sessions train. Committed to git before running.

**Files:** `meta_label.py` (new gates), `config.py` (new constant)
**Academic basis:** De Prado AIFML Ch 3.6 (positive baseline), Peduzzi 1996 (EPV ≥ 10)

### Fix C: Cut Features 23 → ≤5 (EPV fix)
Two-step process:

1. **Univariate scan (train split only):** For each feature × session, compute quintile spread of ExpR. Rank by signal strength. This is a NEW script.
2. **Select top 5:** Features with structural mechanism (RESEARCH_RULES.md) AND univariate signal. Store as `ML_CORE_FEATURES` in config.py.

Expert prior (all have structural mechanisms):
- `orb_size_norm` — ORB size IS the edge
- `atr_20_pct` — vol regime rank
- `gap_open_points_norm` — overnight institutional repositioning
- `orb_pre_velocity` — pre-session momentum
- `prior_sessions_broken` — cross-session flow

The univariate scan CONFIRMS or OVERRIDES this prior. If a feature has no univariate signal, it doesn't make the cut regardless of mechanism.

**Files:** `config.py` (new `ML_CORE_FEATURES` list), new script `scripts/tools/ml_univariate_scan.py`
**Academic basis:** Hastie/Tibshirani/Friedman ESL Ch 7.10 — feature selection on train split only

### Fix D: Retrain + Re-bootstrap + Replay
1. Retrain with fixes B+A+C in place
2. Bootstrap 5K perms with Phipson & Smyth correction
3. Replay vs raw baseline (paper_trader --use-ml vs --raw-baseline)
4. Accept results — 0 survivors is a valid and useful answer

**Files:** `ml_bootstrap_test.py` (update SURVIVORS list after retrain)

---

## Safety: ML Kill Switch During Fix Window

Between changing config and retraining, `predict_live.py` would silently pad 78% of features with -999 sentinels. To prevent garbage predictions:

- Add `ML_METHODOLOGY_VERSION: int = 2` to config.py (bump from implicit v1)
- Store version in model bundle at training time
- `predict_live.py` checks bundle version matches config version — if not, fail-open with WARNING
- This ensures no ML inference runs against stale models during the fix window

---

## Blast Radius Summary

| Component | Impact | Action |
|-----------|--------|--------|
| `config.py` | Feature lists change → hash changes → drift check fires | Expected. Retrain immediately after. |
| `meta_label.py` | New gates + data source change | Core fix. |
| `features.py` | No changes needed | `bypass_validated` path already exists. |
| `predict_live.py` | Version check added | Safety gate. |
| `ml_bootstrap_test.py` | `bypass_validated=True` + updated SURVIVORS | Must match training. |
| `check_drift.py` | `check_ml_config_hash_match` fires until retrain | Expected behavior. |
| 11 research scripts | Import from config — silently get new feature set | Research scripts are historical artifacts. |
| 4 test files | No round-trip feature count test | Add one. |
| `paper_trader.py` | ML predictions change | Only with `--use-ml`. Raw baseline unaffected. |

**Zero impact on:** Layer 1 raw baseline, gold.db schema, pipeline/, any non-ML code.

---

## Pre-Registration (committed before running)

Sessions to test ML on (all 12, full universe):
CME_REOPEN, TOKYO_OPEN, BRISBANE_1025, SINGAPORE_OPEN, LONDON_METALS, EUROPE_FLOW,
US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE

Only sessions with positive raw baseline ExpR will train models (Fix A).
All sessions are reported in results regardless (White 2000).

RR targets to test: 1.0, 1.5, 2.0
Apertures to test: O5, O15, O30 (per-aperture mode)
Entry model: E2 only (E1 optional second pass)

Significance threshold: p < 0.05 (Phipson & Smyth corrected, 5000 perms)
EPV minimum: 10
Multiple testing correction: BH FDR across all tested session × aperture × RR combos

---

## Success Criteria

1. All 3 methodology FAILs are resolved (hard gates in code)
2. Univariate scan completed on train split only
3. Models retrained with ≤5 features on positive-baseline sessions only
4. Bootstrap 5K re-run with honest p-values
5. Results accepted — even if 0 survivors (that's information)
6. Drift checks pass after retrain
7. Layer 1 raw baseline completely unaffected

---

## Risk: Zero Survivors

If no session survives after honest methodology:
- ML for ORB is DEAD — add to NO-GO registry
- Focus shifts to univariate filters (confluence features design, already drafted)
- Layer 1 raw baseline remains the tradeable portfolio
- This is the honest answer and worth having
