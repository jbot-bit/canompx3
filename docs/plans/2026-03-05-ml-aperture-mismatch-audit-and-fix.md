# ML Aperture & RR Mismatch: Design Document (Mar 5 2026)

## Status: DESIGN COMPLETE — Ready for Implementation

---

## Executive Summary

Two silent failure modes identified in the ML meta-label system. Both cause
the Random Forest to produce biased P(win) predictions without raising errors.

| Bug | Type | Severity | Fix |
|-----|------|----------|-----|
| **Aperture mismatch** | Covariate shift (features) | HIGH | Per-(session, aperture) models |
| **RR target mismatch** | Label shift (outcomes) | MEDIUM | RR guard + fail-open |

**Architecture decision**: Train separate RF models per (session, aperture).
Add RR guard to check training RR vs prediction RR. Fail-open on mismatch.

**Implementation**: Two phases.
- **Phase 1** (immediate, ~2 hours): Aperture guard + RR guard in predict_live.py.
  Stores `training_aperture` and checks it at prediction time. Zero risk.
- **Phase 2** (next session, ~4 hours): Per-(session, aperture) training in meta_label.py.
  Trains separate model per aperture. Uses all available data per aperture.

---

## 1. Bug Analysis

### 1.1 Aperture Mismatch (Covariate Shift)

**Root cause**: `features.py` config picker selects ONE aperture per session
(lines 727-771). Training data contains only that aperture's outcomes.
`orb_minutes` becomes constant → dropped by `nunique <= 1` check in
meta_label.py (line 312). Model has zero awareness of its training aperture.

At prediction time, `predict_live.py` accepts any `orb_minutes` value.
`_align_features()` silently drops columns not in `model_feature_names`.
No error raised. Predictions succeed with biased probabilities.

**Feature distributions shift across apertures** (confirmed by DB query):

| Feature | O5 avg | O15 avg | O30 avg |
|---------|--------|---------|---------|
| orb_size (MGC) | 1.52 pts | 2.43 pts (1.6x) | 3.25 pts (2.1x) |
| orb_size_norm | 0.082 | 0.131 | 0.175 |
| orb_volume | lower | higher | highest |

The RF's learned split thresholds (e.g., `if orb_size_norm < 0.10`) produce
WRONG leaf routing when features come from a different aperture's distribution.

**Affected sessions** (3 of 8 ML sessions, 41% of delta R):

| Session | Trained On | At-Risk Apertures | Risk |
|---------|-----------|-------------------|------|
| MGC TOKYO_OPEN | O15 (823 samples) | O5 (36 strats), O30 (3 strats) | HIGH |
| MNQ SINGAPORE_OPEN | O30 (674+ samples) | O5 (29 strats), O15 (strats) | HIGH |
| MES SINGAPORE_OPEN | O5 (large N) | O15 (1 strat) | LOW |

**Safe sessions** (5 of 8): MGC CME_REOPEN (all O5), MGC SINGAPORE_OPEN
(all O5), MNQ TOKYO_OPEN (all O5), MNQ BRISBANE_1025 (all O5),
M2K CME_PRECLOSE (all O5). Single-aperture = no mismatch possible.

### 1.2 RR Target Mismatch (Label Shift)

**Root cause**: Config picker also locks to ONE RR target per session
(via `rr_target_lock` in bundle). `rr_target` was removed from
TRADE_CONFIG_FEATURES (dominated 56-69% importance — tautological).
Model has zero RR awareness.

**This is NOT covariate shift**. Features (orb_size, atr_20, overnight_range)
are IDENTICAL regardless of RR target. What changes is the LABEL: "win"
means "price moved 2.5R" at RR2.5 but "price moved 1.0R" at RR1.0.
Same features, different question.

| RR Mismatch Direction | What Happens | Safe? |
|-----------------------|-------------|-------|
| Match (RR2.5 → RR2.5) | Correct P(win) | YES |
| Conservative (model RR2.5, trade RR1.0) | P(win) understated | YES (safe side) |
| Aggressive (model RR2.5, trade RR3.0+) | P(win) overstated | NO (overconfident) |

**Existing state**: `rr_target_lock: 2.5` stored in bundle but NEVER CHECKED
at prediction time. Trades at any RR are silently accepted.

### 1.3 Why Per-RR Models Are NOT the Answer

Per-aperture models are mandatory because FEATURES shift. Per-RR models are not:
- Same features, different labels → no information gain from separate feature spaces
- N fragmentation: 5-6 RR targets x 3 apertures = 15-18 models per session
- RF can't distinguish RR2.0 from RR2.5 in feature space (features identical)

Correct approach: store training RR, warn/fail-open on aggressive mismatch.

---

## 2. Academic Literature Support

### 2.1 Covariate Shift in Tree Models

**de Prado (ML for Asset Managers, Ch. 3 — Meta-Labeling):**
> "Tree models do not extrapolate. They partition feature space based on
> training distribution. Out-of-distribution points inherit probabilities
> from nearest leaves, which have no guarantee of validity."

Meta-labeling is strategy-specific. The secondary classifier learns
"P(this specific strategy wins)" — conditioned on the training distribution.

**Bailey et al. (Pseudo-Mathematics and Financial Charlatanism):**
> "Changing feature distributions violates the IID assumption underlying
> cross-validation. Calibration curves become inconsistent."

IS performance deteriorates OOS due to data mining bias. Different aperture
= different population = IID violation.

**Aronson (Evidence-Based Technical Analysis, Ch. 6):**
> Three-segment validation (train/val/test) only works when segments are
> from the SAME distribution. "Probability thresholds optimized on one
> regime are unreliable when applied to different regime."

### 2.2 Label Shift (RR Target)

**Carver (Systematic Trading, Ch. 7-8):**
> Forecast strength is continuous, not binary. A strong forecast at one
> target level provides directional information about other levels — but
> the magnitude of the probability estimate requires recalibration.

**Chan (Algorithmic Trading, Ch. 3):**
> "Backtests done using data prior to regime shifts may be quite worthless."
> RR target changes are not regime shifts per se, but they change the
> mapping from features to outcomes in the same way.

### 2.3 Consensus

Random Forests are robust to **monotonic feature transformations** (scaling,
shifting) but **NOT robust to distribution shape changes** (O5→O15 is a shape
change: tighter range → wider range, different volume profile). The right fix
is distribution-matched training, not normalization hacks.

---

## 3. Architecture: Per-(Session, Aperture) Models + RR Guard

### 3.1 Model Keying

```
BEFORE: bundle["sessions"][session] → one model per session
AFTER:  bundle["sessions"][session][f"O{aperture}"] → one model per (session, aperture)
```

Example (MGC after Phase 2):
```python
sessions = {
    "CME_REOPEN": {
        "O5":  {"model": RF, "calibrator": Iso, "threshold": 0.51,
                "training_aperture": 5, "training_rr": 2.5, ...},
        "O15": {"model": None, "reason": "N=82 < 200"},
        "O30": {"model": None, "reason": "N=14 < 200"},
    },
    "TOKYO_OPEN": {
        "O5":  {"model": RF, ...},  # Trained on O5 data only
        "O15": {"model": RF, ...},  # Trained on O15 data only
        "O30": {"model": None, "reason": "N < 200"},
    },
}
```

### 3.2 Prediction Routing

```python
def predict(instrument, session, orb_minutes, rr_target, ...):
    # 1. Look up per-(session, aperture) model
    aperture_key = f"O{orb_minutes}"
    session_data = bundle["sessions"].get(session, {})

    # Phase 1 compatibility: old bundles have flat session dict
    if "model" in session_data:
        # Old format — use aperture guard
        model_info = session_data
        training_aperture = model_info.get("training_aperture")
        if training_aperture and training_aperture != orb_minutes:
            return FAIL_OPEN  # Aperture mismatch
    else:
        # New format — per-aperture lookup
        model_info = session_data.get(aperture_key, {})

    if model_info.get("model") is None:
        return FAIL_OPEN  # No model for this aperture

    # 2. RR guard
    training_rr = model_info.get("training_rr")
    if training_rr and rr_target > training_rr:
        logger.warning("RR mismatch: model=RR%.1f, trade=RR%.1f — fail-open",
                       training_rr, rr_target)
        return FAIL_OPEN  # Model would be overconfident

    # 3. Predict (features guaranteed distribution-matched)
    ...
```

### 3.3 Training Changes (Phase 2)

In `meta_label.py`, the per-session loop becomes a per-(session, aperture) loop:

```python
for session in sessions:
    for aperture in [5, 15, 30]:
        # Load outcomes for THIS aperture only (already the case)
        # Train RF on this aperture's feature distribution
        # Quality gates apply per (session, aperture) — low N → NO_MODEL
        # Store in bundle["sessions"][session][f"O{aperture}"]
```

Quality gates (CPCV >= 0.50, AUC > 0.52, OOS positive, skip < 85%) apply
per (session, aperture). Low-N apertures get NO_MODEL — this is CORRECT
behavior, not a bug. The validated strategy stands on its own.

---

## 4. Edge Cases (25 analyzed, 0 blockers)

### 4.1 Bundle Migration (Phase 1 → Phase 2)
Old bundles without `training_aperture`: fail-open (safe default).
Old flat-format bundles (pre-Phase-2): Phase 1 guard catches mismatches.
Phase 2 bundles: per-aperture lookup, no guard needed.

### 4.2 RR Conservative Direction
Model trained on RR2.5, trade at RR1.0: P(win) is understated (model is
conservative). Take/skip decision is SAFE — if model says "take" at 2.5R,
it's almost certainly right at 1.0R. Allow prediction with info log.

### 4.3 RR Aggressive Direction
Model trained on RR2.5, trade at RR3.0+: P(win) is overstated. Fail-open.
Kelly sizing would over-bet. This is the dangerous direction.

### 4.4 Sessions with Single Aperture (5 of 8)
No change needed. Model trains on O5, predicts on O5. Guard passes.

### 4.5 Entry Model One-Hot Columns
Per-session models already handle this. If session has only E2 trades,
`entry_model_E2` is constant → dropped. Not affected by aperture changes.

### 4.6 skip_filter=True Covariate Shift
Training includes ALL break days (unfiltered). Live prediction sees only
filter-eligible days. This is a minor separate covariate shift.
LOW risk: features.py already includes `orb_size` which captures filter
boundary. Model learns "small ORB = skip" from the unfiltered data.
Not addressed in this fix (different problem, different scope).

### 4.7 Cross-Session Features for Early Sessions
CME_REOPEN and TOKYO_OPEN have near-constant cross-session features
(0 or 1 prior sessions). Already handled by MAX_EARLY_SESSION_INDEX
which drops cross-session features for these sessions. Not affected.

### 4.8 Global Feature Backfill
`_backfill_global_features()` fills from O5 row. Global features
(atr_20, overnight_range, prev_day_range) are aperture-independent.
Backfill is correct and not affected by per-aperture architecture.

---

## 5. M2.5 Second-Opinion Audit Results

Ran M2.5 in bugs mode on features.py, meta_label.py, predict_live.py.
17 findings, 65% false positive rate (consistent with M2.5 empirical rates).

### Verified TRUE

| Finding | File | Impact |
|---------|------|--------|
| DB connection leak in sweep CLI | meta_label.py main() | LOW (CLI only, not production) |
| Import inside function | predict_live.py:343 | STYLE (Python caches, no perf impact) |
| Row-by-row filter iteration | features.py:618-633 | PERF (training only, not live) |

### FALSE POSITIVES (11 of 17)
- M2.5 claimed "wrong merge key in backfill" → SQL already filters by instrument
- M2.5 claimed "timezone mismatch" → Python 3.13 fromisoformat handles offsets
- M2.5 claimed "feature_names captured before constant drop" → actual line order is correct
- M2.5 claimed "KeyError in hybrid path" → early return guard prevents this
- M2.5 claimed "division by zero in proximity" → np.where guards exist

### Design Gaps Identified by M2.5
1. Bundle migration: old bundles without `training_aperture` → **added to 4.1**
2. `rr_target_lock` never checked → **addressed by RR guard in 3.2**
3. `E6_NOISE_EXACT` drops `orb_minutes` → **confirmed, this is the root cause**

---

## 6. Implementation Plan

### Phase 1: Guards (immediate, ~2 hours)

**Goal**: Eliminate silent failures. No training changes.

| File | Change | Lines |
|------|--------|-------|
| `meta_label.py` | Capture `training_aperture` and `training_rr` per session BEFORE constant column drop. Store in session dict. | ~10 |
| `predict_live.py` | Add aperture check + RR guard in `predict()`. Fail-open on mismatch. Backward-compatible with old bundles. | ~25 |
| `test_predict_live.py` | Test: aperture mismatch → fail-open. Test: RR aggressive → fail-open. Test: RR conservative → predict with log. Test: matching → normal. | ~50 |

After Phase 1, retrain all 4 instruments to embed `training_aperture` and
`training_rr` in bundles. Existing models work (fail-open if field missing).

### Phase 2: Per-Aperture Models (~4 hours)

**Goal**: Use all available data per aperture instead of discarding mismatched trades.

| File | Change | Lines |
|------|--------|-------|
| `meta_label.py` | Nest aperture loop inside session loop. Train per (session, aperture). Bundle format: `sessions[session][f"O{aperture}"]`. | ~40 |
| `predict_live.py` | Route by `(session, aperture_key)`. Handle both old and new bundle formats. | ~20 |
| `features.py` | Config picker returns configs PER APERTURE (not just best single). Minor refactor of `load_single_config_feature_matrix`. | ~30 |
| `test_predict_live.py` | Tests for per-aperture routing, NO_MODEL apertures, format migration. | ~40 |

### Phase 3: Retrain + Validate

1. Retrain all 4 instruments with per-aperture architecture
2. Compare OOS delta R: per-aperture vs old single-aperture
3. Verify NO_MODEL apertures match quality gate expectations
4. Paper trader run with mixed apertures to verify routing + logging

---

## 7. Validation Gates

| Gate | Criterion | When |
|------|-----------|------|
| Unit tests pass | All new tests green | Phase 1 + 2 |
| Drift checks pass | `python pipeline/check_drift.py` | Phase 1 + 2 |
| Behavioral audit | `python scripts/tools/audit_behavioral.py` | Phase 1 + 2 |
| Same-aperture regression | P(win) and threshold unchanged for matching apertures | Phase 1 |
| OOS delta R comparison | Per-aperture total >= single-aperture total | Phase 2 |
| NO_MODEL count | Low-N apertures correctly rejected by quality gates | Phase 2 |

---

## 8. Decision Record

| Decision | Rationale | Alternatives Rejected |
|----------|-----------|----------------------|
| Per-aperture models | Features shift across apertures (confirmed). Only fix that uses all data. | Normalized single model (academic consensus: don't mix distributions) |
| RR guard (not per-RR models) | Features identical across RR. Only labels change. Per-RR would fragment N. | Per-RR models (same features, N collapse) |
| Fail-open on mismatch | Validated strategy stands on its own. Silence > false signal (Aronson). | Fail-closed (blocks trading unnecessarily) |
| Two-phase rollout | Phase 1 is zero-risk safety net. Phase 2 can be deferred without danger. | Single big-bang change (higher risk) |
| Quality gates per (session, aperture) | Low N at non-dominant apertures is expected. Gates catch it automatically. | Lower thresholds for non-dominant apertures (overfitting risk) |

---

## 9. Glossary

- **Aperture**: ORB time window (5, 15, or 30 minutes). `orb_minutes` in code.
- **Covariate shift**: Feature distributions differ between training and prediction.
- **Label shift**: Same features, different outcome definition (RR1.0 vs RR2.5 "win").
- **Fail-open**: On error or mismatch, trade proceeds on validated strategy alone (no ML).
- **Config picker**: `load_single_config_feature_matrix()` in features.py. Selects ONE
  config per session based on `config_selection` parameter (max_samples or best_sharpe).
- **Quality gates**: 4 post-training checks (OOS positive, CPCV >= 0.50, AUC > 0.52,
  skip < 85%). Applied per model; low-N models are automatically rejected.
