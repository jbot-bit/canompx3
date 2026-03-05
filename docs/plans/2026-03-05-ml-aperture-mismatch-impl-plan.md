# ML Aperture & RR Mismatch — Implementation Plan

**Design doc:** `docs/plans/2026-03-05-ml-aperture-mismatch-audit-and-fix.md`
**Date:** 2026-03-05

---

## Phase 1: Guards (Zero-Risk Safety Net)

### Task 1: Capture training_aperture and training_rr in meta_label.py

**File:** `trading_app/ml/meta_label.py`
**What:** Inside the per-session training loop (line ~279), after filtering to session rows
but BEFORE the constant column drop (line ~312), extract the unique `orb_minutes` and
`rr_target` values from `meta_all` for this session. Store them in `session_results`.
Then in the bundle-building section (lines ~549-572), add `training_aperture` and
`training_rr` to each session's saved dict.

**Exact changes:**

1. **After line 282** (after `session_indices = np.where(smask)[0]`), add:
   ```python
   # Capture training aperture/RR BEFORE constant column drop removes them.
   # Single-config mode loads one aperture per session, so orb_minutes is constant.
   session_orb_minutes = meta_all["orb_minutes"].iloc[session_indices]
   training_aperture = int(session_orb_minutes.iloc[0])
   session_rr = meta_all["rr_target"].iloc[session_indices]
   training_rr = float(session_rr.iloc[0])
   ```

2. **In the SESSION result dict** (line ~471, the `session_results[session] = {` block),
   add two new keys:
   ```python
   "training_aperture": training_aperture,
   "training_rr": training_rr,
   ```

3. **In the bundle-building section** (line ~551-566, where `bundle["sessions"][session]`
   is constructed for SESSION models), add:
   ```python
   "training_aperture": info["training_aperture"],
   "training_rr": info["training_rr"],
   ```

**Verification:** After this change, run:
```bash
python -m trading_app.ml.meta_label --instrument MGC --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter --no-cpcv
```
Then inspect the bundle:
```python
import joblib
b = joblib.load("models/ml/meta_label_MGC_hybrid.joblib")
for s, info in b["sessions"].items():
    print(s, info.get("training_aperture"), info.get("training_rr"))
```
Each session should show its training aperture (5, 15, or 30) and RR (2.5).

**Lines changed:** ~10

---

### Task 2: Add aperture guard + RR guard in predict_live.py

**File:** `trading_app/ml/predict_live.py`
**What:** In the `predict()` method, after resolving the session model (lines ~213-219),
add checks for aperture mismatch and aggressive RR mismatch. Both fail-open.

**Exact changes:**

1. **Add a counter** at line ~83 (in `__init__`):
   ```python
   self.aperture_mismatch_count: int = 0
   self.rr_mismatch_count: int = 0
   ```

2. **After the hybrid session check** (after line ~219, after the `return result` for
   missing session model), add aperture + RR guard:
   ```python
   # Aperture guard: check training aperture matches prediction aperture.
   # Old bundles without training_aperture → fail-open (safe default).
   if is_hybrid and session_info is not None:
       training_aperture = session_info.get("training_aperture")
       if training_aperture is not None and training_aperture != orb_minutes:
           logger.info(
               "Aperture mismatch for %s %s: model trained on O%d, "
               "prediction for O%d — fail-open",
               instrument, orb_label, training_aperture, orb_minutes,
           )
           self.aperture_mismatch_count += 1
           result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
           self._prediction_cache[cache_key] = result
           return result

       # RR guard: aggressive RR (trade RR > training RR) → fail-open.
       # Conservative RR (trade RR < training RR) is safe — P(win) is understated.
       training_rr = session_info.get("training_rr")
       if training_rr is not None and rr_target > training_rr:
           logger.info(
               "RR mismatch for %s %s: model trained on RR%.1f, "
               "prediction for RR%.1f (aggressive) — fail-open",
               instrument, orb_label, training_rr, rr_target,
           )
           self.rr_mismatch_count += 1
           result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
           self._prediction_cache[cache_key] = result
           return result

       if training_rr is not None and rr_target < training_rr:
           logger.debug(
               "RR conservative for %s %s: model RR%.1f, trade RR%.1f — "
               "P(win) may be understated",
               instrument, orb_label, training_rr, rr_target,
           )
   ```

3. **In `summary()` method** (line ~416), add the new counters:
   ```python
   "aperture_mismatch_count": self.aperture_mismatch_count,
   "rr_mismatch_count": self.rr_mismatch_count,
   ```

**Lines changed:** ~30

---

### Task 3: Add tests for aperture guard and RR guard

**File:** `tests/test_trading_app/test_ml/test_predict_live.py`
**What:** Add a new test class `TestApertureRRGuard` with 5 tests.

**Test 1: `test_aperture_mismatch_fails_open`**
- Create hybrid bundle with `training_aperture=5` on SINGAPORE_OPEN session
- Predict with `orb_minutes=15`
- Assert fail-open result (0.5, True, 0.5)
- Assert `aperture_mismatch_count == 1`
- Assert model's `predict_proba` was NOT called (short-circuits before RF)

**Test 2: `test_aperture_match_predicts_normally`**
- Create hybrid bundle with `training_aperture=5` on SINGAPORE_OPEN session
- Predict with `orb_minutes=5`
- Assert real prediction (p_win != 0.5)
- Assert `aperture_mismatch_count == 0`

**Test 3: `test_rr_aggressive_fails_open`**
- Create hybrid bundle with `training_rr=2.5` on SINGAPORE_OPEN session
- Predict with `rr_target=3.0`
- Assert fail-open result
- Assert `rr_mismatch_count == 1`

**Test 4: `test_rr_conservative_predicts_normally`**
- Create hybrid bundle with `training_rr=2.5` on SINGAPORE_OPEN session
- Predict with `rr_target=1.5`
- Assert real prediction (p_win != 0.5)
- Assert `rr_mismatch_count == 0`

**Test 5: `test_old_bundle_without_guards_predicts_normally`**
- Create hybrid bundle WITHOUT `training_aperture` or `training_rr` keys
- Predict with any orb_minutes and rr_target
- Assert real prediction (backward compatible — old bundles don't fail-open)

**Implementation pattern:** Follow existing `_make_hybrid_bundle()` helper.
Add `training_aperture` and `training_rr` to session dicts when creating bundles.

**Lines changed:** ~80

---

### Task 4: Run tests + drift checks

**Commands:**
```bash
python -m pytest tests/test_trading_app/test_ml/test_predict_live.py -x -q -v
python -m pytest tests/ -x -q
python pipeline/check_drift.py
```

**Pass criteria:** All tests green, drift checks pass.

---

### Task 5: Retrain all 4 instruments (embed training_aperture/training_rr)

**Commands:**
```bash
python -m trading_app.ml.meta_label --instrument MGC --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter
python -m trading_app.ml.meta_label --instrument MNQ --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter
python -m trading_app.ml.meta_label --instrument MES --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter
python -m trading_app.ml.meta_label --instrument M2K --single-config --rr-target 1.0 \
  --config-selection max_samples --skip-filter
```

**Verification:** Inspect each bundle and confirm `training_aperture` and `training_rr`
are present in every session dict.

**NOTE:** Results will vary slightly from previous run due to RF randomness (n_jobs=-1).
The +250-300R range should be stable. Session composition may shift +-1 session.

---

## Phase 2: Per-Aperture Models

### Task 6: Refactor features.py config picker to support per-aperture loading

**File:** `trading_app/ml/features.py`
**What:** Add a new function `load_per_aperture_feature_matrices()` that returns a dict
of `{(session, aperture): (X, y, meta)}`. This wraps the existing config picker logic
but iterates over apertures instead of picking the best single one.

**Approach:** The existing `load_single_config_feature_matrix()` already filters to one
aperture per session (via `bc.rn = 1`). For per-aperture, we modify the PARTITION to
`PARTITION BY v.orb_label, v.orb_minutes` so it picks the best config PER (session, aperture).

**Changes:**
1. Add parameter `per_aperture: bool = False` to `load_single_config_feature_matrix()`
2. When `per_aperture=True`, change the PARTITION clause:
   ```sql
   PARTITION BY v.orb_label, v.orb_minutes  -- instead of just v.orb_label
   ```
3. Return the full DataFrame — the training loop handles splitting by (session, aperture)
4. Add `aperture` column to `meta` from the loaded data (it's already there as `orb_minutes`)

**Lines changed:** ~15 (the change is small — just the SQL PARTITION clause)

---

### Task 7: Refactor meta_label.py training loop for per-(session, aperture) models

**File:** `trading_app/ml/meta_label.py`
**What:** Add `--per-aperture` CLI flag. When set, the per-session loop at line ~279
becomes a per-(session, aperture) loop. The inner training logic is unchanged — it's
the same RF, same quality gates, same calibration.

**Changes:**
1. Add `per_aperture: bool = False` parameter to `train_per_session_meta_label()`
2. If `per_aperture`, call `load_single_config_feature_matrix(per_aperture=True)`
3. Replace `for session in sessions:` loop with:
   ```python
   for session in sessions:
       session_results[session] = {}
       for aperture in sorted(meta_all["orb_minutes"].unique()):
           aperture_key = f"O{aperture}"
           # Filter to this (session, aperture) combination
           mask = (meta_all["orb_label"] == session) & (meta_all["orb_minutes"] == aperture)
           ...  # same training logic, using mask instead of smask
           session_results[session][aperture_key] = { ... }
   ```
4. Bundle format changes from `sessions[session] = {...}` to
   `sessions[session] = {"O5": {...}, "O15": {...}, "O30": {...}}`
5. Add `"bundle_format": "per_aperture"` to top-level bundle metadata
6. Add `--per-aperture` to argparse in `main()`

**Lines changed:** ~40

---

### Task 8: Update predict_live.py routing for per-aperture bundles

**File:** `trading_app/ml/predict_live.py`
**What:** In the `predict()` method, detect the bundle format and route accordingly.

**Changes:**
1. After resolving `session_info` (line ~214), detect format:
   ```python
   if is_hybrid:
       session_data = bundle.get("sessions", {}).get(orb_label)
       if session_data is None:
           return FAIL_OPEN  # session not in bundle

       # Detect per-aperture vs flat format
       aperture_key = f"O{orb_minutes}"
       if bundle.get("bundle_format") == "per_aperture":
           # New format: sessions[session][aperture_key]
           session_info = session_data.get(aperture_key, {})
       elif "model" in session_data:
           # Old flat format: sessions[session] = {model, ...}
           session_info = session_data
           # Phase 1 aperture guard still applies for old bundles
       else:
           session_info = {}

       if session_info.get("model") is None:
           return FAIL_OPEN
   ```
2. RR guard logic from Phase 1 remains — applies to per-aperture models too
3. Update `get_model_info()` to handle nested session format

**Lines changed:** ~25

---

### Task 9: Add tests for per-aperture routing

**File:** `tests/test_trading_app/test_ml/test_predict_live.py`
**What:** Add `TestPerApertureModel` class.

**Tests:**
1. `test_per_aperture_routes_to_correct_model` — O5 prediction hits O5 model
2. `test_per_aperture_no_model_for_aperture` — O30 has no model → fail-open
3. `test_per_aperture_rr_guard_still_works` — aggressive RR → fail-open even with per-aperture
4. `test_old_flat_format_still_works` — backward compatibility with Phase 1 bundles
5. `test_per_aperture_different_models_per_aperture` — O5 and O15 have different thresholds

**Lines changed:** ~80

---

### Task 10: Run full test suite + drift checks

**Commands:**
```bash
python -m pytest tests/ -x -q
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
```

---

## Phase 3: Retrain + Validate

### Task 11: Retrain all 4 instruments with --per-aperture flag

**Commands:**
```bash
python -m trading_app.ml.meta_label --instrument MGC --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter --per-aperture
python -m trading_app.ml.meta_label --instrument MNQ --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter --per-aperture
python -m trading_app.ml.meta_label --instrument MES --single-config --rr-target 2.5 \
  --config-selection max_samples --skip-filter --per-aperture
python -m trading_app.ml.meta_label --instrument M2K --single-config --rr-target 1.0 \
  --config-selection max_samples --skip-filter --per-aperture
```

### Task 12: Compare per-aperture vs single-aperture results

**Verification:**
1. Inspect each bundle — confirm per-aperture format with nested session dicts
2. Compare total honest delta R to Phase 1 baseline
3. Count models per (session, aperture) — expect low-N apertures get NO_MODEL
4. Verify SAFE sessions (single-aperture) produce identical results

### Task 13: Code review

Run the `superpowers-extended-cc:code-reviewer` subagent on all changed files.

---

## Dependency Graph

```
Task 1 (meta_label captures) → Task 2 (predict_live guards)
Task 2 → Task 3 (guard tests)
Task 3 → Task 4 (verify all pass)
Task 4 → Task 5 (retrain with new fields)
Task 5 → PHASE 1 COMPLETE

Task 6 (features.py per-aperture) → Task 7 (meta_label loop)
Task 7 → Task 8 (predict_live routing)
Task 8 → Task 9 (per-aperture tests)
Task 9 → Task 10 (verify all pass)
Task 10 → Task 11 (retrain per-aperture)
Task 11 → Task 12 (compare results)
Task 12 → Task 13 (code review)
Task 13 → PHASE 2 COMPLETE
```

## Batching for Review Checkpoints

- **Batch 1 (Phase 1, Tasks 1-3):** Code changes — commit after Task 4 passes
- **Batch 2 (Phase 1, Tasks 4-5):** Verification + retrain — commit bundles
- **Batch 3 (Phase 2, Tasks 6-9):** Code changes — commit after Task 10 passes
- **Batch 4 (Phase 2, Tasks 10-13):** Verification + retrain + review
