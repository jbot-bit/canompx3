---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ML Fix Execution Plan

**Purpose:** Fix 6 proven bugs in ML pipeline. Each fix is isolated, verified, committed independently.
**Approach:** Understand → implement → verify → commit. No batching. No rushing.
**Outcome bias check:** Fixes remove verified bugs. They do not add anything that steers toward "ML works" or "ML doesn't work." BH FDR is the final arbiter.

---

## Phase 1: Correctness Fixes (no model impact, no retraining needed)

### 1.1 Fix C — Train-only constant-column drop

**Current code** (`meta_label.py:361`):
```python
session_data = X_session.iloc[session_indices]  # ALL indices (train+val+test)
const_cols = [c for c in X_session.columns if session_data[c].nunique() <= 1]
```

**Fix:**
```python
session_data = X_session.iloc[train_idx]  # TRAIN only
const_cols = [c for c in X_session.columns if session_data[c].nunique() <= 1]
```

**Why this is honest:** Feature selection decisions must not see test data. Period.
**Why it's low-risk:** In practice, the same columns drop (entry_model one-hots are constant everywhere). But correctness matters.
**Blast radius:** Zero. Same columns drop. No hash change. No model invalidation.
**Verify:** Log dropped columns before/after. Expect identical sets.

### 1.2 Fix A — Deterministic config selection tiebreaker

**Current code** (`features.py` lines 798, 800, 909, 981):
```sql
ORDER BY v.cnt DESC              -- lines 909, 981
ORDER BY v.sample_size DESC      -- line 798
ORDER BY v.sharpe_ratio DESC     -- line 800
```

**Fix:** Add deterministic tiebreaker to all 4:
```sql
ORDER BY v.cnt DESC, v.entry_model DESC, v.confirm_bars ASC, v.orb_minutes ASC   -- lines 909, 981
ORDER BY v.sample_size DESC NULLS LAST, v.entry_model DESC, v.confirm_bars ASC   -- line 798
ORDER BY v.sharpe_ratio DESC NULLS LAST, v.entry_model DESC, v.confirm_bars ASC  -- line 800
```

`entry_model DESC` → prefers E2 over E1 (alphabetical, E2 is primary).
`confirm_bars ASC` → prefers CB1 (E2's default).

**Why this is honest:** Non-determinism means different runs can produce different training data from the same DB. That's a bug.
**Blast radius:** Low. Query change. No schema. No hash change.
**Verify:** Run `load_single_config_feature_matrix` twice, diff output.

---

## Phase 2: Lookahead Fix (affects feature computation)

### 2.1 Fix B — EUROPE_FLOW / LONDON_METALS DST cross-session lookahead

**Current code** (`meta_label.py:190-209`, `_get_session_features()`):
- Drops cross-session features only for index ≤ 1 (CME_REOPEN, TOKYO_OPEN)
- EUROPE_FLOW (index 5) keeps cross-session features including LONDON_METALS (index 4)
- In winter, EF (17:00) precedes LM (18:00) — LM data is lookahead for EF

**Also affected:** `features.py:247` — level proximity uses same `SESSION_CHRONOLOGICAL_ORDER[:session_idx]`

**Fix:** In `_get_session_features()`, add a condition for the EF/LM pair:
```python
# DST swap: EUROPE_FLOW and LONDON_METALS swap chronological order by season.
# Neither can safely include the other as "prior." Drop cross-session + level
# features for both to prevent seasonal lookahead contamination.
DST_SWAP_SESSIONS = {"EUROPE_FLOW", "LONDON_METALS"}
if session in DST_SWAP_SESSIONS:
    drop_cols = [c for c in CROSS_SESSION_FEATURES if c in X_e6.columns]
    drop_cols += [c for c in LEVEL_PROXIMITY_FEATURES if c in X_e6.columns]
    X_session = X_e6.drop(columns=drop_cols, errors="ignore")
```

**Why this is honest:** Using future data as a feature is cheating. Doesn't matter if it "helps" or "hurts" ML.
**Blast radius:** EF and LM models lose ~9 features. With Fix F (5 expert features), they lose `prior_sessions_broken` + all 6 level proximity = train on 4 features. Still EPV-compliant at O5 RR1.0 (~430 positives / 4 features = EPV 107).
**Verify:** Training logs for EF/LM show cross-session + level features dropped. Other sessions unaffected.

---

## Phase 3: Methodology Gates (change which sessions train)

### 3.1 Fix E — Positive baseline gate

**Current code:** No baseline check exists. Sessions with negative ExpR train models, then get rejected by Gate 1 (OOS delta < 0) at line 466. This wastes compute and allows negative-baseline models into the training process.

**Fix:** In `meta_label.py`, after line 336 (train/val/test split computed), before line 345 (minimum samples check):
```python
# De Prado gate: primary model must have positive edge
train_pnl = pnl_r[train_idx]
train_expr = float(train_pnl.mean()) if len(train_idx) > 0 else 0.0
if train_expr <= 0:
    logger.info(f"{log_prefix} >> NO_MODEL (negative baseline: ExpR={train_expr:+.4f}R on {len(train_idx)} train trades)")
    _store({"model_type": "NONE", "reason": f"negative_baseline_expr={train_expr:+.4f}"})
    continue
```

**Why this is honest:** De Prado (AIFML Ch 3.6) — meta-labeling requires positive-edge primary model. Not an optimization. Not a filter. A prerequisite.
**Blast radius:** Blocks negative-baseline sessions. At O5 RR1.0: 0/12 blocked. At O30 RR2.0: ~6/12 blocked. Medium.
**Verify:** Run on O5 RR1.0. All 12 sessions should proceed. Run on O30 RR2.0. Confirm blocked sessions match DB query results (NYSE_OPEN -0.136, US_DATA_1000 -0.125, etc.).

---

## Phase 4: Feature Reduction (highest blast radius — do last)

### 4.1 Fix F — ML_CORE_FEATURES (23 → 5)

**Current code:** `apply_e6_filter()` drops ~20 noise features → ~25 remain. All 25 go to RF.

**Fix:** Add to `config.py`:
```python
ML_CORE_FEATURES: list[str] = [
    "orb_size_norm",
    "atr_20_pct",
    "gap_open_points_norm",
    "orb_pre_velocity_norm",
    "prior_sessions_broken",
]
```

In `_get_session_features()`, after existing drop logic and before return:
```python
from trading_app.ml.config import ML_CORE_FEATURES
keep = [c for c in X_session.columns if c in ML_CORE_FEATURES]
if keep:
    X_session = X_session[keep]
```

**Why this is honest:** EPV ≥ 10 is a statistical requirement (Peduzzi 1996). Not an optimization. Features chosen on structural mechanism, not data-mining. The old importance rankings from the broken model are NOT cited as evidence.
**Why expert prior over scan:** A univariate scan is its own multiple-testing problem (276 tests). Expert features with mechanisms are cleaner. If 0 survivors after retrain, we accept it — not scan for "better" features.
**Blast radius:** HIGH. Changes `compute_config_hash()` → all .joblib invalidated → drift check fires. Expected. V2 version gate already rejects old models.
**Verify:** `X_session.shape[1]` = 5 (or 4 for EF/LM after Fix B). EPV at O5 RR1.0 ≈ 430/5 = 86.

---

## Phase 5: Post-Processing Fix (after retrain)

### 5.1 Fix D — Universe-wide BH/FDR

**Current code:** `ml_bootstrap_test.py:200` reports PASS/FAIL per-config at p < 0.05. No FDR.

**Fix:** After the summary loop in `main()`, add:
```python
# BH FDR across full tested family
p_vals = sorted([r["p_value"] for r in all_results if "p_value" in r])
n_tests = len(p_vals)
for i, p in enumerate(p_vals):
    bh_threshold = (i + 1) / n_tests * 0.05
    q_value = p * n_tests / (i + 1)  # BH adjusted
    survives = p <= bh_threshold
    log.info(f"  BH rank {i+1}/{n_tests}: p={p:.4f}, threshold={bh_threshold:.4f}, q={min(q_value,1):.4f}, {'SURVIVES' if survives else 'KILLED'}")
```

**Why this is honest:** Reporting per-config p-values without FDR is p-hacking. BH at q=0.05 is the standard correction.
**Blast radius:** Zero — post-processing in standalone script. No production code.
**Verify:** Apply BH to old 5K results. Confirm at most 1 survivor.

---

## Phase 6: Retrain + Bootstrap (execution, not code change)

After ALL fixes committed and verified:
1. Pre-register universe in a committed doc (108 configs)
2. Run training sweep (O5/O15/O30 × RR1.0/RR1.5/RR2.0)
3. Update SURVIVORS in bootstrap script
4. Run bootstrap 5K
5. Apply BH FDR
6. Report ALL results
7. Accept 0 survivors if that's the answer

---

## Phase 7: Docs Update (real numbers from logs, not memory)

Update STRATEGY_BLUEPRINT.md, MEMORY.md, HANDOFF.md with actual log output.
If 0 FDR survivors: add ML to NO-GO registry.
