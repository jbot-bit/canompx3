# ML Improvement — 3-Phase Design Document (V2: Post-Audit)

**Status:** DESIGN — awaiting "go" to implement
**Date:** 2026-03-27 (V2: re-examined 2x, grounded against live codebase)
**Prerequisite:** `docs/plans/2026-03-21-ml-methodology-fix-design.md` (approved Fix A-F design)
**Kill gate:** If Phase 1 produces <2 BH FDR survivors at K=12 → ML is DEAD. Stop.

---

## Audit Trail — What V1 Got Wrong

Before reading the plan, know what changed between V1 and V2:

| # | V1 Error | V2 Fix |
|---|----------|--------|
| 1 | Feature names assumed but not verified | Ran `load_single_config_feature_matrix` — confirmed all 5 core features exist in E6-filtered matrix (25 columns total) |
| 2 | LightGBM monotone constraints were positional `[+1,+1,0,0,+1]` | Must be name-mapped dict, not positional list — early sessions drop `prior_sessions_broken`, breaking positional mapping |
| 3 | "Retrain all 108 configs" was hand-waved | Specified wrapper script: 9 invocations of `meta_label.py` (3 RR × 3 aperture flags), collect CPCV AUC per session from logs |
| 4 | E6 filter interaction unexamined | Confirmed: none of the 5 core features match E6 noise patterns. E6 is harmless but kept as defense-in-depth |
| 5 | Bootstrap compute cost unspecified | 5000 perms × 12 sessions ≈ 60K model fits ≈ 8-10 hours. Overnight run. |
| 6 | Kill condition was binary (DEAD vs ALIVE) | Refined: 0 = DEAD, 1 = CONDITIONAL (investigate, don't proceed to Phase 2), ≥2 = ALIVE |
| 7 | `MLPrediction` NamedTuple change was breaking | Python 3.11 supports defaults on NamedTuple — new fields get defaults, existing callers unaffected |
| 8 | MAPIE `method='lac'` asserted without verification | Verified: `method='lac'` IS correct (default for binary in MAPIE ≥1.2); `cv='prefit'` confirmed valid. Pin to ≥1.2. |
| 9 | Man AHL cited for monotone constraints | Corrected: Man AHL discusses parameter stability and model confidence sets, NOT monotone constraints. Monotone constraints are UNSUPPORTED by local resources. |
| 10 | Fix C (train-only column drop) overstated | With 5 core features, constant columns within a session are unlikely. Fix is still correct but low-impact. |
| 11 | Config selection CPCV cost not computed | 108 configs × 10 CPCV splits = 1080 model fits for selection. ~20 min. Acceptable. |
| 12 | `compute_config_hash` update not specified | Must add `ML_CORE_FEATURES` to hash input. Existing hash includes feature lists that won't change — need explicit core features in hash. |

---

## Dependency Graph

```
Phase 1: Fix A-F (Methodology Rehabilitation)
  ├── Fix A (deterministic config)     ──┐
  ├── Fix B (EF/LM cross-session)     ──┤── independent, any order
  ├── Fix C (train-only column drop)   ──┤
  ├── Fix F (5 features + hash update) ──┘
  ├── Fix E (positive baseline gate)   ── after F (features must be final)
  ├── Pre-registration commit          ── after A+B+C+E+F
  ├── Retrain 108 configs (wrapper)    ── after pre-reg (~20 min)
  ├── Config selection (12 best)       ── after retrain, CPCV AUC ranking
  ├── Bootstrap 12 sessions (5K each)  ── after selection (~8-10 hrs overnight)
  └── Fix D (BH FDR at K=12)          ── after bootstrap
        │
        ▼
  KILL GATE: how many sessions survive BH?
    0     → ML DEAD, add to NO-GO §5, STOP
    1     → ML CONDITIONAL, investigate only, do NOT proceed to Phase 2
    ≥2    → ML ALIVE, proceed to Phase 2
        │
        ▼
Phase 2: LightGBM Challenger + Conformal Prediction
  ├── 2A: LightGBM challenger          ── only on surviving sessions
  │     ├── Optuna hyperopt (20 trials per session, ~40 min total)
  │     ├── Champion/challenger eval on frozen test set
  │     └── Winner must independently pass bootstrap at K=12
  ├── 2B: Conformal prediction (MAPIE)  ── wraps Phase 2A winner
  │     ├── Alpha sweep on val set (conformal calibration)
  │     └── OOS comparison: conformal zones vs threshold sweep
  └── DECISION: which model + decision method to deploy
        │
        ▼
Phase 3: Monitoring + AI Coach
  ├── 3A: ML monitoring stack           ── needs deployed model from Phase 2
  └── 3B: AI Trading Coach              ── ZERO ML coupling, can start anytime
```

---

## Phase 1: Fix A-F (Methodology Rehabilitation)

### Goal

Implement the 6 approved fixes, then honestly evaluate whether ML adds value.
This phase either produces validated ML configs or kills ML permanently.

### FDR K-Selection Design

**Problem:** K=108 (12 sessions × 3 apertures × 3 RR) makes BH FDR impossible.
Minimum achievable p-value with 5000 perms: (0+1)/(5000+1) = 0.0002.
BH rank-1 at K=108: 0.05/108 = 0.000463. Effectively only 1 config could ever
survive, and only by beating ALL 5000 permutations.

**Resolution:** Session is the discovery unit (K=12). Aperture and RR are hyperparameters.

**Codebase precedent:**
`strategy_validator.py:1211-1240` — BH FDR already runs per-session:
```python
session_p_pools: dict[str, list[tuple[str, float]]] = {}
# ... pools strategies by orb_label ...
for session_name, pool in sorted(session_p_pools.items()):
    k_session = len(pool)
    session_fdr = benjamini_hochberg(pool, alpha=0.05, total_tests=k_session)
```
K is per-session, NOT global. This is the canonical FDR method for this project.

`RESEARCH_RULES.md` § Multiple Testing: "Use instrument/family K for promotion
decisions, global K for headline claims." Session = family.

**Academic grounding:**
- BH (1995) Theorem 1 (p.293 in `resources/benjamini-and-Hochberg-1995-fdr.pdf`):
  FDR control proven for **independent** test statistics. Sessions test different
  market microstructures — genuinely independent hypotheses.
- Within-session configs are **correlated**: O5⊂O15⊂O30 (nested apertures),
  RR1.0/1.5/2.0 share entries (correlated outcomes). Under positive dependence,
  BH is conservative (Benjamini & Yekutieli 2001 — not in local resources but
  standard result).
- Chordia et al (2020, `resources/Two_Million_Trading_Strategies_FDR.pdf` p.3):
  testing families defined by the hypothesis, not the parameter grid. They apply
  FDR across strategies testing the same anomaly.
- Bailey & de Prado (2014, `resources/deflated-sharpe.pdf` p.1): correction
  accounts for the number of **independent** trials.

**The protocol:**

1. **Retrain** all 108 configs (12 sessions × 3 apertures × 3 RR) with the fixed methodology.
2. **Select** the best (aperture, RR) per session by CPCV AUC computed on the 60% train split.
   This is hyperparameter tuning, not discovery. Selection committed BEFORE bootstrap.
3. **Bootstrap** the 12 selected configs (one per session). 5000 perms each. One p-value per session.
4. **BH FDR** at K=12, q=0.05. Report K=108 as conservative footnote.
5. **Cross-session consistency:** ≥2 survivors required for ML ALIVE.

**Why selection before bootstrap is defensible:** CPCV AUC is computed entirely within
the train split. The bootstrap evaluates honest_delta_r on the test split. These are
independent — selecting the best config on train does not bias the test-set bootstrap.
This is standard hyperparameter tuning within cross-validation.

**Why ≥2 survivors:** With 12 independent tests at FDR q=0.05, the expected proportion
of false positives among rejections is ≤5%. But a single rejection leaves 5% chance it's
false. Two rejections with expected FP rate 5% means ~0.1 expected false among them —
overwhelmingly likely both are real. A single survivor is investigated but not enough to
declare ML a production system.

### Fix A: Deterministic Config Selection

**File:** `trading_app/ml/features.py` lines 909 and 981
**Change:** Add tiebreaker to both `ORDER BY v.cnt DESC`:
```sql
ORDER BY v.cnt DESC, v.rr_target ASC, v.entry_model ASC, v.orb_minutes ASC
```
**Breaks if wrong:** Non-deterministic model selection across retrains.
**Verify:** Call `load_single_config_feature_matrix` twice, assert identical config.
**Blast radius:** Query-only. No downstream impact.

### Fix B: EUROPE_FLOW / LONDON_METALS Cross-Session Lookahead

**File:** `trading_app/ml/meta_label.py` — `_get_session_features()` lines 186-209
**Current code:** Only handles early sessions (index ≤ MAX_EARLY_SESSION_INDEX).
**Change:** Add EF/LM guard to the else branch (line 206-207):
```python
# Current:
else:
    X_session = X_e6

# Changed to:
elif session in ("EUROPE_FLOW", "LONDON_METALS"):
    # EF/LM swap chronological order by season (winter EF=17:00<LM=18:00,
    # summer LM=17:00<EF=18:00). Cross-session features from one contaminate
    # the other ~42% of the time. Drop all cross-session + level proximity.
    drop_cols = [c for c in CROSS_SESSION_FEATURES if c in X_e6.columns]
    drop_cols += [c for c in LEVEL_PROXIMITY_FEATURES if c in X_e6.columns]
    X_session = X_e6.drop(columns=drop_cols, errors="ignore")
else:
    X_session = X_e6
```
**Impact on core features:** `prior_sessions_broken` is in CROSS_SESSION_FEATURES
and gets dropped for EF/LM. These sessions train with 4 of 5 core features. EPV improves
(fewer features, same samples).
**Breaks if wrong:** ~42% of EF rows train on future LM data.
**Verify:** Call `_get_session_features(X, 'EUROPE_FLOW')`, assert `prior_sessions_broken` absent.
**Blast radius:** Only 2/12 sessions affected.

### Fix C: Train-Only Constant-Column Drop

**File:** `trading_app/ml/meta_label.py` line 361
**Change:** `session_data = X_session.iloc[session_indices]` → `session_data = X_session.iloc[train_idx]`
**Note:** With only 5 core features, constant columns are unlikely (no one-hot categoricals
left). Fix is still correct but mainly a code hygiene improvement.
**Blast radius:** Negligible.

### Fix F: Feature Reduction (EPV Fix)

**File 1: `trading_app/ml/config.py`** — Add after TRADE_CONFIG_FEATURES (line ~113):
```python
# Expert-prior features for V2 methodology.
# 5 features selected by structural mechanism, NOT data-driven scan.
# Verified present in E6-filtered matrix (25 cols → 5 selected):
#   orb_size_norm, atr_20_pct, gap_open_points_norm,
#   orb_pre_velocity_norm, prior_sessions_broken
# EPV at O5 RR1.0 MNQ: ~1300 × 55% WR / 5 = ~143 (well above 10).
# EPV at O30 RR2.0 MNQ: ~430 × 34% WR / 5 = ~29 (above 10).
#
# @research-source: expert prior selection (Hastie/Tibshirani ESL §7.10 — NOT local)
# @revalidated-for: E2
ML_CORE_FEATURES: list[str] = [
    "orb_size_norm",           # ORB size IS the edge (Blueprint §2, cost mechanism)
    "atr_20_pct",              # Vol regime rank (confirmed ATR70_VOL filter)
    "gap_open_points_norm",    # Overnight institutional repositioning (ATR-normalized)
    "orb_pre_velocity_norm",   # Pre-session momentum slope (ATR-normalized)
    "prior_sessions_broken",   # Cross-session flow (#1 importance in prior experiments)
]
```

**File 2: `trading_app/ml/config.py`** — Update `compute_config_hash()`:
```python
# Add to the config_str:
f"|{ML_CORE_FEATURES}"
```
This ensures the hash changes, invalidating old models.

**File 3: `trading_app/ml/meta_label.py`** — After E6 filter (line ~274):
```python
X_e6 = apply_e6_filter(X_all)

# V2 methodology: select only expert-prior features (EPV fix)
from trading_app.ml.config import ML_CORE_FEATURES
available_core = [f for f in ML_CORE_FEATURES if f in X_e6.columns]
if len(available_core) < len(ML_CORE_FEATURES):
    missing = set(ML_CORE_FEATURES) - set(available_core)
    logger.warning(f"Missing core features (will be absent for some sessions): {missing}")
X_e6 = X_e6[available_core]
```
**Note:** E6 filter runs first as defense-in-depth (none of the 5 core features match
E6 noise patterns, so it drops 0 columns in practice, but keeps the safety net).

**Breaks if wrong:** EPV stays at 2.2 with 23 features — model overfit by definition.
**Verify:** After fix, `X_e6.shape[1]` should be 5 (or 4 for EF/LM/CME_REOPEN after Fix B).
**Blast radius:** HIGH — `compute_config_hash()` changes, all `.joblib` models rejected.
Drift check `check_ml_config_hash_match` (check_drift.py:2380) fires. EXPECTED.

### Fix E: Positive Baseline Gate

**File:** `trading_app/ml/meta_label.py` — after line 336 (train/val/test index computation)
**Change:**
```python
# Fix E: Positive baseline gate (de Prado AIFML Ch 3.6 — NOT local)
# Meta-labeling assumes primary model has positive edge.
# Training on negative baselines produces threshold artifacts (confirmed p=0.35).
train_pnl = pnl_r[train_idx]
train_expr = float(train_pnl.mean()) if len(train_pnl) > 0 else 0.0
if train_expr <= 0:
    logger.info(f"{log_prefix} >> SKIP (negative baseline ExpR={train_expr:+.4f} on train)")
    _store({"model_type": "NONE", "reason": f"negative_baseline_expr={train_expr:+.4f}"})
    continue
```
**Ordering:** MUST come after Fix F. Baseline ExpR depends on which samples have
valid features. With 5 features and fewer NaN rows, sample counts may shift slightly.
**Breaks if wrong:** ML trains on negative-baseline sessions → threshold artifacts.
**Blast radius:** At O30 RR2.0, kills ~6/12 sessions. At O5 RR1.0, kills 0/12.

### Fix D: Universe-Wide BH FDR (Post-Bootstrap)

**File:** `scripts/tools/ml_bootstrap_test.py` — after all session bootstraps
**Change:** Reuse `benjamini_hochberg` from `strategy_validator.py`:
```python
from trading_app.strategy_validator import benjamini_hochberg

# One p-value per session (from pre-registered config selection)
session_pvalues = [(s, r["p_value"]) for s, r in bootstrap_results.items()]

# Promotion: K=12 (session-level family)
fdr_k12 = benjamini_hochberg(session_pvalues, alpha=0.05, total_tests=12)

# Conservative footnote: K=108 (global grid)
fdr_k108 = benjamini_hochberg(session_pvalues, alpha=0.05, total_tests=108)

n_survivors = sum(1 for v in fdr_k12.values() if v["fdr_significant"])
```
**Breaks if wrong:** Without FDR, 12 independent tests at p<0.05 have 46% chance
of ≥1 false positive. BH controls expected FP proportion among rejections.
**Blast radius:** Post-processing only. No production code impact.

### Retrain Wrapper Script

**Problem identified in audit:** The training script takes `--rr-target` and `--per-aperture`
flags. To train all 108 configs, we need a wrapper.

**New file:** `scripts/tools/ml_v2_retrain_all.py`
```python
"""Retrain all 108 ML configs for Phase 1 evaluation.

Invokes meta_label.py 9 times:
  3 RR targets (1.0, 1.5, 2.0) × {flat, per-aperture O5, O15, O30}

Actually: 3 RR × 1 flat + 3 RR × 1 per-aperture = 6 invocations.
Per-aperture mode trains O5/O15/O30 within one call.
Flat mode trains a single model per session across all apertures.

Collects CPCV AUC per (session, aperture, RR) into a summary CSV.
The best config per session (by CPCV AUC) is selected for bootstrap.
"""
```

**Implementation note:** The current `train_per_session_meta_label` returns a dict with
per-session results including `cpcv_auc`. The wrapper calls it 6 times (3 RR × 2 modes)
and collects results. No refactoring of the training loop needed.

**Compute cost:** 6 invocations × 12 sessions × ~10 CPCV splits = ~720 model fits.
At ~1s each with 5 features: ~12 minutes. Acceptable.

### Pre-Registration Protocol

Commit to `docs/pre-registrations/ml-v2-preregistration.md` BEFORE any retrain:
```markdown
# ML V2 Pre-Registration
Date: [commit date]
Commit: [hash]

Universe: MNQ E2, all 12 sessions, O5/O15/O30, RR 1.0/1.5/2.0 (108 configs)
Features: ML_CORE_FEATURES (5 expert priors)
Data: bypass_validated=True (full orb_outcomes)
Gates: train ExpR > 0, EPV ≥ 10
Config selection: best CPCV AUC per session on train split (committed before bootstrap)
Bootstrap: 5000 perms, Phipson & Smyth (2010)
FDR: BH at K=12 (session family), q=0.05. K=108 reported as footnote.
Cross-session: ≥2 BH survivors required.
Acceptance: whatever survives is real. <2 survivors = ML DEAD.
```

### Implementation Order

```
 1. Fix A  →  4 LOC, verify, commit
 2. Fix B  →  10 LOC, verify, commit
 3. Fix C  →  2 LOC, verify, commit
 4. Fix F  →  25 LOC (config.py + meta_label.py + hash), verify, commit
 5. Fix E  →  10 LOC, verify, commit
 6. Pre-registration  →  commit doc
 7. Retrain wrapper script  →  50 LOC, run all 108 configs (~12 min)
 8. Config selection  →  script ranks by CPCV AUC, outputs 12 selected, commit
 9. Bootstrap 12 sessions  →  run overnight (~8-10 hrs)
10. Fix D  →  30 LOC, apply BH FDR to bootstrap results
11. KILL GATE  →  evaluate: 0=DEAD, 1=CONDITIONAL, ≥2=ALIVE
```

### File-Level Change List

| File | Action | Changes | LOC |
|------|--------|---------|-----|
| `trading_app/ml/features.py:909,981` | MODIFY | ORDER BY tiebreaker (Fix A) | 4 |
| `trading_app/ml/meta_label.py:199-207` | MODIFY | EF/LM cross-session guard (Fix B) | 10 |
| `trading_app/ml/meta_label.py:361` | MODIFY | Train-only column drop (Fix C) | 2 |
| `trading_app/ml/config.py` | MODIFY | `ML_CORE_FEATURES`, update `compute_config_hash` (Fix F) | 15 |
| `trading_app/ml/meta_label.py:274` | MODIFY | Apply core feature selection (Fix F) | 8 |
| `trading_app/ml/meta_label.py:336` | MODIFY | Positive baseline gate (Fix E) | 10 |
| `scripts/tools/ml_v2_retrain_all.py` | CREATE | Wrapper: 6 invocations, collect CPCV, rank, select | 50 |
| `scripts/tools/ml_bootstrap_test.py` | MODIFY | BH FDR at K=12 and K=108 (Fix D) | 30 |
| `docs/pre-registrations/ml-v2-preregistration.md` | CREATE | Pre-registration doc | 25 |
| `tests/test_trading_app/test_ml/test_config.py` | MODIFY | Test ML_CORE_FEATURES count=5, hash changes | 10 |
| `tests/test_trading_app/test_ml/test_meta_label.py` | MODIFY | Test baseline gate, EF/LM guard | 20 |

### Blast Radius

- `compute_config_hash()` output changes → all `.joblib` bundles invalidated.
  `predict_live.py:100` checks hash, warns, fails open. `check_drift.py:2380`
  `check_ml_config_hash_match` fires. Both EXPECTED.
- `ML_METHODOLOGY_VERSION` stays at 2 (already bumped). No version gate change.
- `evaluate.py` uses model bundle's stored `feature_names` for alignment
  (`_fill_missing_features(X, feature_names)`). Works correctly with 5 features.
- No downstream impact on `strategy_validator.py`, `outcome_builder.py`, or pipeline.
  ML is a leaf node.

### Acceptance Criteria

1. All 6 fixes committed with individual verification
2. Pre-registration committed BEFORE retrain
3. Retrain produces per-session models with 5 features (4 for EF/LM/CME_REOPEN)
4. EPV ≥ 10 logged for every trained session
5. Negative-ExpR sessions skipped on train split
6. Bootstrap p-values for 12 sessions computed
7. BH FDR at K=12 and K=108 both reported
8. Kill gate evaluated
9. `pytest tests/test_trading_app/test_ml/ -x -q` passes
10. `python pipeline/check_drift.py` passes (hash mismatch is expected/documented)

### Kill Conditions

| Outcome | Action |
|---------|--------|
| 0 sessions survive BH at K=12 | **ML DEAD.** Add to Blueprint §5 NO-GO: "ML V2 with 5 expert priors: 0/12 BH survivors. Raw baselines are the portfolio." |
| 1 session survives | **ML CONDITIONAL.** Log finding. Do NOT proceed to Phase 2. Monitor in forward test. Revisit when data grows by 50%. |
| ≥2 sessions survive | **ML ALIVE.** Proceed to Phase 2. |
| EPV < 10 for ALL sessions | **ML DEAD.** (Extremely unlikely with 5 features — would need <50 positive trades per session.) |

### Estimated LOC

| Component | Code | Tests | Docs |
|-----------|------|-------|------|
| Fixes A-F | 59 | 30 | 0 |
| Retrain wrapper | 50 | 0 | 0 |
| Fix D (BH FDR) | 30 | 0 | 0 |
| Pre-registration | 0 | 0 | 25 |
| **Phase 1 total** | **~139** | **~30** | **~25** |

---

## Phase 2: LightGBM Challenger + Conformal Prediction

### Gate

Phase 1 MUST produce ≥2 BH FDR survivors. Otherwise this phase is cancelled.

### 2A: LightGBM Challenger

**Goal:** Test whether gradient boosting with monotone domain constraints outperforms
RF on the same 5 features and same evaluation framework.

#### Monotone Constraints (NAME-MAPPED, not positional)

**V1 bug:** Used positional list `[+1,+1,0,0,+1]`. This breaks when `_get_session_features`
drops `prior_sessions_broken` for early sessions (CME_REOPEN, TOKYO_OPEN) or EF/LM,
leaving 4 features with 5 constraints.

**V2 fix:** Name-based mapping, resolved at train time:
```python
LGBM_MONOTONE_MAP: dict[str, int] = {
    "orb_size_norm": 1,           # Bigger ORB = less friction drag = +
    "atr_20_pct": 1,              # Higher vol regime = more opportunity = +
    "prior_sessions_broken": 1,   # Market-wide flow = higher conviction = +
    "gap_open_points_norm": 0,    # Gap direction matters, no monotone prior
    "orb_pre_velocity_norm": 0,   # Velocity sign matters, no monotone prior
}

# At train time, build constraint list matching actual feature order:
constraints = [LGBM_MONOTONE_MAP.get(f, 0) for f in feature_names]
```

**Why these constraints:**
- `orb_size_norm` +1: G-filter mechanism. Friction as % of risk is inversely proportional
  to ORB size. Larger ORB → less friction → higher P(win). Confirmed across all instruments.
- `atr_20_pct` +1: ATR70_VOL filter is a deployed, validated filter. Higher vol regime =
  more breakout energy. Monotone prior = vol helps, never hurts.
- `prior_sessions_broken` +1: #1 feature importance in prior experiments (12.2% MES, 8.3% MNQ).
  More prior breaks = market-wide directional flow = higher breakout continuation probability.
- `gap_open_points_norm` 0: Large positive gap ≠ large negative gap for breakout quality.
  Direction matters. No monotone prior.
- `orb_pre_velocity_norm` 0: Strong up momentum ≠ strong down momentum. Sign matters.

**Academic justification for monotone constraints:** UNSUPPORTED by local resources.
Man AHL (2015, `resources/man_overfitting_2015.pdf` p.1) discusses "penalizing models for
complexity" and "model stability under parameter variation" as overfitting mitigations.
Monotone constraints are a specific form of structural regularization that reduces hypothesis
space. They encode domain knowledge (G-filter mechanism, vol regime) that IS locally
supported, even though the technique itself is not cited in local PDFs.

#### Hyperparameter Search

```python
# Optuna TPE sampler, 20 trials per session
search_space = {
    "num_leaves": (8, 64),           # Tree complexity
    "min_child_samples": (50, 200),  # Leaf size (regularization)
    "learning_rate": (0.01, 0.1),    # Step size (log-uniform)
    "subsample": (0.6, 1.0),         # Row sampling
    "colsample_bytree": (0.6, 1.0),  # Feature sampling (limited effect with 5 features)
}
# Fixed:
#   n_estimators=200 (convergence tested in existing RF experiments)
#   is_unbalance=True (native class weighting)
#   monotone_constraints=<from name map>
#   monotone_constraints_method='advanced'
#   random_state=42
```

**Inner evaluation:** Mean CPCV AUC on the 60% train split (same CPCV splits as RF).
**Compute cost:** 20 trials × 10 CPCV splits × surviving sessions (say 4) = 800 model fits.
LightGBM with 5 features and ~800 samples/fold trains in <0.5s. Total: ~7 minutes.
Optuna study persisted to `models/ml/optuna_{instrument}_{session}.db`.

#### Champion/Challenger Evaluation

1. Train RF (Phase 1 winner) and LightGBM (Optuna winner) on same 60% train split.
2. Calibrate both with isotonic regression on same 20% val set.
3. Evaluate on same 20% frozen test set.
4. **Primary metric: calibrated Brier score.**
   - Brier = mean((predicted_p - actual_outcome)^2)
   - We care about probability QUALITY because conformal (2B) relies on calibrated probabilities.
   - AUC only measures ranking, not calibration. Lower Brier = better.
5. Secondary: AUC, honest_delta_r, skip_pct.
6. **Winner must independently pass bootstrap** at session-level BH FDR q=0.05.

### 2B: Conformal Prediction (MAPIE)

**Goal:** Replace threshold sweep (36 candidates, selection bias on val set) with
conformal prediction sets that provide distribution-free coverage guarantees.

#### Method

```python
from mapie.classification import MapieClassifier

# Wrap the winning model (RF or LightGBM)
mapie = MapieClassifier(estimator=winner_model, method='lac', cv='prefit')
mapie.fit(X_val, y_val)  # Calibrate on validation set

# Predict with coverage guarantee
y_pred, y_sets = mapie.predict(X_test, alpha=0.10)
# y_sets shape: (n_samples, n_classes, len(alpha)) — boolean for each class
```

**V1 was actually correct:** `method='lac'` (Least Ambiguous Classifier) is the current
default and recommended method for binary classification in MAPIE ≥1.2. `'score'` is the
older name, still accepted for backward compatibility. LAC produces tighter prediction sets.

**`cv='prefit'`:** Model already trained. MAPIE only calibrates conformity scores on the
validation set. No re-training.

**Decision zones from prediction sets:**
- `y_sets[i, 1, 0] == True and y_sets[i, 0, 0] == False` → TAKE (only {win} in set)
- `y_sets[i, 0, 0] == True and y_sets[i, 1, 0] == False` → SKIP (only {lose} in set)
- `y_sets[i, 0, 0] == True and y_sets[i, 1, 0] == True` → ABSTAIN (both in set)
- Neither → TAKE (empty set → fail-open)

**Alpha sweep:** [0.05, 0.10, 0.15, 0.20] (95%, 90%, 85%, 80% coverage)

**Calibration set size gate:** Need N ≥ 100 per session on the val split.
Verified: at O5 RR1.0 MNQ with ~1300 samples/session, val = ~260. Gate passes.
Sessions with val < 100 fall back to threshold sweep (no conformal).

**Key evaluation (on frozen test set):**

| Metric | Target | Kill if |
|--------|--------|---------|
| Empirical coverage | ≥ (1-alpha) | < (1-alpha) - 0.05 |
| ABSTAIN rate | < 60% | > 80% |
| TAKE-only Sharpe | > threshold Sharpe | < baseline Sharpe |
| SKIP avg_r | < 0 (model correctly skips losers) | > 0 (model wrong) |

**MLPrediction backward compatibility (V1 bug fix):**
```python
class MLPrediction(NamedTuple):
    p_win: float
    take: bool
    threshold: float
    decision: str = "THRESHOLD"  # "TAKE" | "SKIP" | "ABSTAIN" | "THRESHOLD"
    prediction_set: tuple = ()   # frozenset can't be NamedTuple default; use tuple
```
Default values ensure all existing callers (execution_engine.py:603, :608, :609) work
without changes. They access `.take`, `.p_win`, `.threshold` — all unchanged.

**Academic justification:** UNSUPPORTED by local resources. Conformal prediction
(Vovk et al 2005) not in `resources/`. Justified because: (a) distribution-free coverage
guarantee replaces the 36-candidate threshold sweep which has documented selection bias
(`meta_label.py:134-155` comments), (b) MAPIE is scikit-learn compatible and production-grade.

### Phase 2 File-Level Change List

| File | Action | Changes | LOC |
|------|--------|---------|-----|
| `trading_app/ml/lightgbm_model.py` | CREATE | LGBMClassifier wrapper, Optuna objective, name-mapped constraints | 180 |
| `trading_app/ml/champion_challenger.py` | CREATE | Side-by-side eval: Brier, AUC, delta_r, bootstrap | 120 |
| `trading_app/ml/conformal.py` | CREATE | MAPIE wrapper, alpha sweep, zone logic, size gate | 150 |
| `trading_app/ml/config.py` | MODIFY | `LGBM_MONOTONE_MAP`, `OPTUNA_N_TRIALS`, search space, conformal defaults | 30 |
| `trading_app/ml/meta_label.py` | MODIFY | `--model-type` CLI flag, model selection hook | 40 |
| `trading_app/ml/predict_live.py` | MODIFY | MLPrediction defaults, conformal path, bundle compat | 45 |
| `trading_app/ml/evaluate.py` | MODIFY | Conformal zone evaluation, Brier score reporting | 60 |
| `tests/test_trading_app/test_ml/test_lightgbm.py` | CREATE | Monotone constraints, Optuna, name mapping | 60 |
| `tests/test_trading_app/test_ml/test_conformal.py` | CREATE | TAKE/SKIP/ABSTAIN, coverage, size gate | 50 |
| `tests/test_trading_app/test_ml/test_predict_live.py` | MODIFY | New MLPrediction fields, backward compat | 15 |

### Phase 2 Blast Radius

- `predict_live.py` MLPrediction gains 2 fields with defaults. All callers use named
  access (verified: execution_engine.py:603,608,609). **No breaking change.**
- Model bundle gains `conformal_calibrator` key. `predict_live.py` must handle bundles
  with/without it (check `bundle.get("conformal_calibrator")`).
- New pip deps: `lightgbm`, `optuna`, `mapie` (Phase 2 only, not Phase 1).

### Phase 2 Acceptance Criteria

1. LightGBM trains with name-mapped monotone constraints on surviving sessions
2. Optuna study persisted to SQLite, 20 trials complete
3. Champion/challenger Brier scores computed on frozen test set
4. Winner passes independent bootstrap at BH FDR K=12
5. Conformal zones produce valid coverage ≥ (1-alpha) on test set
6. ABSTAIN rate < 60% at alpha=0.10 for ≥2 sessions
7. All existing tests pass + new tests

### Phase 2 Kill Conditions

| Outcome | Action |
|---------|--------|
| LightGBM Brier worse than RF on ALL sessions | Stay with RF. LightGBM adds no value. |
| LightGBM fails bootstrap at K=12 | Stay with RF. |
| Conformal ABSTAIN > 80% on ALL sessions | Fall back to threshold sweep. |
| Conformal TAKE Sharpe < baseline Sharpe | Stay with threshold. |
| SKIP avg_r > 0 consistently | Model is wrong — skipping good trades. Investigate. |

### Phase 2 Estimated LOC

| Component | Code | Tests |
|-----------|------|-------|
| 2A: LightGBM + champion/challenger | 370 | 75 |
| 2B: Conformal + evaluate + predict | 255 | 65 |
| **Phase 2 total** | **~625** | **~140** |

---

## Phase 3: Monitoring + AI Coach

### 3A: ML Monitoring Stack

Detect model drift before it costs money. Applies to whatever model survives Phase 2.

**Feature drift (PSI):** Per-feature, rolling 60-day windows. Alert: PSI ≥ 0.25.
**Score drift (KS):** 2-sample KS on P(win) distribution. Alert: p < 0.01.
**Calibration drift:** Decile bins, predicted vs realized win rate. Alert: max deviation > 10%.
**PnL attribution:** ML-TAKE / ML-SKIP / ML-ABSTAIN / BASELINE buckets, monthly.

Storage: `ml_monitoring` table in gold.db. Dashboard: ML panel in `pipeline/dashboard.py`.

### 3B: AI Trading Coach

**Zero ML coupling.** Data source: broker APIs. Analysis: Claude API. Storage: `data/` directory.
Independently shippable. Full design: `docs/plans/2026-03-06-ai-coach-design.md`.
Phase 1 scope: broker fill fetcher + trade matcher → `data/broker_trades.jsonl`.

### Phase 3 File-Level Change List

| File | Action | LOC |
|------|--------|-----|
| `trading_app/ml/monitoring.py` | CREATE | 200 |
| `pipeline/init_db.py` | MODIFY | 15 |
| `pipeline/dashboard.py` | MODIFY | 80 |
| `scripts/tools/run_ml_monitoring.py` | CREATE | 40 |
| `tests/test_trading_app/test_ml/test_monitoring.py` | CREATE | 60 |

### Phase 3 Estimated LOC

| Component | Code | Tests |
|-----------|------|-------|
| 3A: Monitoring | 335 | 60 |
| 3B: Coach Phase 1 | 200 | 55 |
| **Phase 3 total** | **~535** | **~115** |

---

## New Dependencies

| Package | Version | Phase | Purpose |
|---------|---------|-------|---------|
| `lightgbm` | ≥4.0 | 2A | Gradient boosting with monotone constraints |
| `optuna` | ≥3.5 | 2A | Bayesian hyperparameter search (TPE) |
| `mapie` | ≥1.2 | 2B | Conformal prediction (MapieClassifier, `method='lac'`) |

Phase 1 requires NO new dependencies.

---

## Summary

| Phase | Code | Tests | Deps | Kill gate | Compute |
|-------|------|-------|------|-----------|---------|
| 1: Fix A-F | ~139 | ~30 | None | <2 BH survivors → DEAD | ~12 min retrain + ~10 hrs bootstrap |
| 2: LightGBM + Conformal | ~625 | ~140 | 3 packages | LightGBM/conformal fail → stay RF/threshold | ~50 min |
| 3: Monitoring + Coach | ~535 | ~115 | None | N/A (infrastructure) | N/A |
| **Total** | **~1299** | **~285** | **3** | | |

**Critical path:** Phase 1 bootstrap (overnight) → BH FDR → go/no-go.

---

## Academic References

| Reference | Local? | Sections cited |
|-----------|--------|----------------|
| Benjamini & Hochberg (1995) | YES (`resources/benjamini-and-Hochberg-1995-fdr.pdf`) | Theorem 1 (p.293): FDR control for independent tests. §3.1: BH procedure. |
| Chordia, Goyal & Saretto (2020) | YES (`resources/Two_Million_Trading_Strategies_FDR.pdf`) | p.3: family defined by hypothesis, not parameter grid. §4.1: FWER vs FDR. |
| Bailey & de Prado (2014) | YES (`resources/deflated-sharpe.pdf`) | p.1: independent trials correction for multiple testing. |
| Aronson (2006) | YES (`resources/Evidence_Based_Technical_Analysis_Aronson.pdf`) | p.231: Monte Carlo permutation as gold standard for strategy testing. |
| Man AHL (2015) | YES (`resources/man_overfitting_2015.pdf`) | p.1: model confidence sets, parameter stability, complexity penalties. |
| De Prado "ML for Asset Managers" | YES (`resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`) | p.22: meta-labeling reference. Book is 45-page Elements edition — full meta-labeling theory in AIFML (NOT local). |
| De Prado AIFML Ch 3.6 | **NO** | Meta-labeling positive baseline requirement. Referenced throughout codebase. |
| Peduzzi et al (1996) | **NO** | EPV ≥ 10 rule. Standard biostatistics. |
| Benjamini & Yekutieli (2001) | **NO** | BH under positive dependence (within-session configs). |
| Phipson & Smyth (2010) | **NO** | Permutation p-value: (b+1)/(m+1). Used in bootstrap. |
| Vovk et al (2005) | **NO** | Conformal prediction theory. |
| LightGBM monotone constraints | **NO** | Domain-knowledge regularization. UNSUPPORTED by local resources. |
