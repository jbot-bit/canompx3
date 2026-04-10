# ML V3 — Stage 0 Verification Report

**Date:** 2026-04-11
**Author:** Claude Code institutional audit session
**Status:** COMPLETE — Stage 1 unblocked
**Sprint plan:** `TaskList` IDs #1-5

---

## Purpose

Before writing the V3 pre-registered hypothesis file (Stage 2), verify:
1. The pooled universe is what we think it is (the 52-vs-45 memory discrepancy)
2. The candidate V3 features are actually available AND free of look-ahead bias
3. The current `LiveMLPredictor` empty-models behavior matches the code-comment claim
4. The Mode A holdout split is computable and non-trivial in size

No production code was touched. All findings are read-only queries and source inspection.

---

## Q1 — Validated setups status clarification

**Query:** `SELECT status, COUNT(*) FROM validated_setups GROUP BY status`

| status | count | avg ExpR | avg WR | notes |
|---|---|---|---|---|
| `active`  | 29 | 0.122 | 53.5% | MNQ=27, MES=2, GC=0 |
| `retired` | 23 | 0.143 | 55.2% | GC=17, MNQ=6 |

**The 17 GC strategies are ALL retired.** They were promoted and retired on 2026-04-11 — consistent with memory note `gc_mgc_cross_validation_results.md`: "9/10 FAIL. Edge does NOT transfer. MGC proxy deployment path DEAD." GC strategies have genuine positive edge on GC itself (avg ExpR 0.150, WR 55%) but failed MGC cross-validation so they're not deployable.

**V3 pooled universe decision:** MNQ + MES only (29 strategies). GC retired strategies are NOT included in training pool because (a) they are not the deployment target, (b) pooling cross-instrument mixes regimes, (c) their retirement signals poor transfer which would bias the pooled model toward noise.

---

## Q2 — Pooled universe composition

Query used precomputed `sample_size` and `win_rate` from `validated_setups` (already correctly computed at discovery time; no need to re-derive via triple-join).

```
Total active strategies: 29
Pooled sample_size:      29,488 trades
Pooled positives (est):  15,280 trades
Pooled WR:               51.82%
Pooled ExpR:             +0.108R
```

**By session (pooled):**

| instrument | session | n_strats | trades | pct of pool | avg_wr | avg_expR |
|---|---|---|---|---|---|---|
| MES | CME_PRECLOSE | 2 | 481 | 1.6% | 63.1% | 0.185 |
| MNQ | CME_PRECLOSE | 1 | 596 | 2.0% | 62.4% | 0.170 |
| MNQ | COMEX_SETTLE | 8 | 7,992 | 27.1% | 54.1% | 0.138 |
| MNQ | EUROPE_FLOW  | 7 | 7,567 | 25.7% | 52.2% | 0.103 |
| MNQ | NYSE_OPEN    | 6 | 7,348 | 24.9% | 51.5% | 0.109 |
| MNQ | TOKYO_OPEN   | 4 | 4,810 | 16.3% | 49.6% | 0.102 |
| MNQ | US_DATA_1000 | 1 | 694   | 2.4% | 57.2% | 0.100 |

**Observations:**
- **78% of the pool is in 3 US sessions** (COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN). The pooled model will learn US-session regime patterns most strongly. This matches deployment reality — these sessions are where the validated edge lives.
- **TOKYO_OPEN has the lowest average WR (49.6%)** — below 50%. This is the one session where the primary model is near-negative baseline. Risk of dragging the pooled positive baseline. *Stage 2 decision:* still include or drop?
- **Pooled WR 51.82% is POSITIVE** — de Prado Ch 3 meta-labeling precondition is satisfied for the pooled model. This is the first time in the project's ML history.

**EPV for a 6-feature pooled RF:**
```
EPV = 15,280 positives / 6 features = 2,547
```
Well above the Peduzzi 1996 ≥ 10 threshold. Well above the "statistically comfortable" ≥ 100. **The EPV trap that killed V2 per-session (EPV=2.4) is structurally broken by pooling.**

---

## Q3 — Holdout window (Mode A sacred, 2026-01-01 onwards)

**Raw orb_outcomes 2026+ pool (upper bound, before filter drop-through):**

| instrument | orb_minutes | setup rows | trading days |
|---|---|---|---|
| MES | 5 | 25,884 | 71 |
| MNQ | 5 | 28,296 | 72 |

Available holdout = ~72 trading days (2026-01-02 → 2026-04-06). For the 29 active strategies with filter drop-through, estimated pooled holdout trade count:

```
Rough estimate: 29,488 × (72 / 1,905) ≈ 1,115 pooled holdout trades
```

**This is a thin holdout per strategy (~40 trades/strategy, below Criterion 7's N ≥ 100 per-strategy threshold).** However, the V3 evaluation is on the POOLED model (one model for all strategies), so the effective N for the holdout test is the full ~1,115. That's plenty for a Sharpe / ExpR comparison test and a 5K bootstrap.

**Criterion 8 kill criterion restated for pooled V3:** "Pooled holdout ExpR ≥ 0.40 × pooled IS ExpR (= 0.40 × 0.108 = 0.043R)." Locked in Stage 2.

---

## Q4 — Candidate feature availability & look-ahead audit

### Q4a — V2 core features (all verified pre-break safe)

Existing ML V2 uses 5 hand-picked theory-grounded features (`trading_app/ml/config.py:127-133`). All are pre-break safe per the architectural comment at lines 56-72 ("ML PREDICTION ARCHITECTURE: PRE-BREAK — decide before placing stop").

| feature | source | coverage (MNQ O5, pre-holdout) | pre-break safe? |
|---|---|---|---|
| `orb_size_norm`          | transform-time (features.py) | 100% (derived from orb_{SESSION}_size) | ✓ known at ORB close |
| `atr_20_pct`             | `daily_features.atr_20_pct`  | 96.9% | ✓ stored, computed from prior days |
| `gap_open_points_norm`   | transform-time               | 99.9% (raw gap_open_points) | ✓ known at session start |
| `orb_pre_velocity_norm`  | transform-time               | ~99% (raw orb_{SESSION}_pre_velocity) | ✓ computed at ORB close |
| `prior_sessions_broken`  | transform-time (features.py) | computed | ✓ prior sessions only |

### Q4b — NEW candidate features

| feature | status | verdict |
|---|---|---|
| `rel_vol_{SESSION}` | **DROPPED** | `rel_vol = break_bar_volume / 20d median`. `break_bar_volume` is in `LOOKAHEAD_BLACKLIST` (config.py:173). Computing rel_vol requires the break bar, which is unknown pre-break. Confluence program used it as an at-break filter (safe in that context), not as a pre-break ML feature. Including it in V3 would leak the break into the prediction. |
| `pit_range_atr` | **DROPPED** | Column exists in `daily_features` schema but is **0% populated** across MNQ/MES/MGC (0/1169 rows each). Memory note `exchange_range_signal.md` referenced a feature that was either never backfilled or lives in a different table. Not usable. |
| `orb_{SESSION}_volume` | **KEPT** | Per `config.py:77` SESSION_FEATURE_SUFFIXES: "Total ORB-window volume — known at ORB close (pre-break)". Coverage 85-88% across all 6 active sessions for MNQ pre-holdout. Will be used as `orb_{SESSION}_volume_norm` via 20-day session-median normalization in transform. |
| `overnight_range` | **DROPPED** | In `LOOKAHEAD_BLACKLIST` (config.py:160-164) — session-dependent look-ahead for Asian sessions. Note: used as a FILTER (`OVNRNG_100`) on post-17:00-Brisbane sessions where it's pre-ORB safe, but NOT safe as a mixed-session ML feature. |

### Q4c — Final V3 feature set (LOCKED)

6 features, all pre-break safe, all empirically available:

1. `orb_size_norm` — transform-time, theory: ORB size is the primary edge mechanism (Blueprint §2 cost-mechanism)
2. `atr_20_pct` — stored, theory: vol regime rank matters (confirmed by ATR70_VOL filter)
3. `gap_open_points_norm` — transform-time, theory: overnight institutional repositioning
4. `orb_pre_velocity_norm` — transform-time, theory: pre-session momentum matters
5. `prior_sessions_broken` — transform-time, theory: cross-session flow is top feature in V2 experiments
6. `orb_{SESSION}_volume_norm` — **NEW**, transform-time, theory: session volume vs its 20d norm is a confluence signal (confluence program 48 BH survivors univariately)

EPV with 6 features: 15,280 / 6 = **2,547**. Safe.

---

## Q5 — LiveMLPredictor empty-models behavior verification

Source inspection of `trading_app/ml/predict_live.py:216-261`:

```python
def predict(self, instrument, trading_day, ...):
    ...
    # Fail-open: no model for this instrument
    if instrument not in self._models:
        self.fail_open_count += 1
        result = MLPrediction(p_win=0.5, take=True, threshold=0.5)
        self._prediction_cache[cache_key] = result
        return result
```

**Confirmed:** when `self._models` is empty (e.g., `models/ml/` directory missing or empty), every predict() call returns `take=True` with `p_win=0.5`. The current live bot state (no models) is equivalent to "ML gate disabled, all trades pass through." This is the silent fail-open path that Stage 1 will replace with explicit behavior.

**Observed risk (confirmed):** `models/ml/` directory **does not exist at all** on disk — not just empty. On `LiveMLPredictor.__init__`, `MODEL_DIR` is referenced via `pathlib.Path(...)` but never checked for existence. If someone drops a rogue `.joblib` into the dir, it gets silently loaded on next orchestrator startup. Drift checks catch structure + 90-day freshness but run at commit time, not at live startup.

**Stage 1 fix specification:**
- `trading_app/ml/config.py` add `ML_ENABLED: bool = os.getenv("ML_ENABLED", "0") == "1"` (default disabled)
- `trading_app/live/session_orchestrator.py:241-254` gate on `ML_ENABLED`:
  - `ML_ENABLED=False` → `self._ml_predictor = None`, log INFO "ML disabled (ML_ENABLED=0)"
  - `ML_ENABLED=True` AND `MODEL_DIR` empty or missing required instrument model → `RuntimeError` at startup (fail-closed: refuse to boot, not silent fail-open)
  - `ML_ENABLED=True` AND models present → current path
- `trading_app/paper_trader.py:271-279` same gate
- `LiveMLPredictor.__init__` assert `MODEL_DIR.exists()` — create or raise, not silent reference
- Tests for all three states

---

## Kill-criterion pre-registration (binding for Stage 2 hypothesis file)

The following kill criteria will be written into `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence.yaml` and are **binding** — no post-hoc relaxation. If any trigger in Stage 3 evaluation, ML V3 is DEAD and Stage 4 deletes `trading_app/ml/`.

1. **BH FDR:** 0/K survive at q=0.05 with K = pre-registered trial count → DEAD
2. **WFE:** best survivor WFE < 0.50 (Criterion 6) → DEAD
3. **DSR:** best survivor DSR < 0.95 per Bailey-LdP 2014 Eq. 2 (Criterion 5) → DEAD
4. **Holdout ExpR:** pooled 2026 sacred holdout `ExpR < 0.40 × pooled IS ExpR` (Criterion 8 = 0.043R minimum) → DEAD
5. **Era stability:** any era (2019-22, 2023, 2024-25) with ≥50 trades shows pooled ExpR < −0.05 → era-dependent, cannot deploy as general case (Criterion 9)
6. **Chordia t:** final survivor `t < 3.00` (theory-supported, Criterion 4) → DEAD
7. **Beat-the-baseline test:** per LdP 2020 institutional audit, must beat a 1-line ATR>50th percentile filter. If simple filter matches ML delta → ML is unnecessary, DEAD.

---

## Architectural choices (documented for Stage 2)

**V3 = pre-break pooled model.** Not per-session (breaks EPV trap). Not at-break (major scope change).

Rationale:
- Pre-break reuses existing `predict_live.py` / `features.py` / `cpcv.py` / `ExecutionEngine` infrastructure. Zero architecture-level refactoring.
- Pooling across all 29 validated strategies breaks the EPV=2.4 trap that killed V2 per-session (now EPV=2,547).
- At-break migration would require (a) new ExecutionEngine integration point to pass break-bar context to predictor, (b) new feature pipeline for break-bar features, (c) re-testing of `rel_vol` as a true ML feature. Deferred to V4 if V3 passes and at-break is still of interest.

**Training window:** 2019-05-06 → 2025-12-31 real-micro bars (Phase 3c canonical, 1,951 pre-holdout trading days, 6.65 clean years).

**Holdout window:** 2026-01-01 → present (Mode A sacred, currently 72 trading days = ~3.8% of pre-holdout). Touched ONCE at the very end of Stage 3 evaluation. No peeking, no re-running on the holdout.

**Model type:** RandomForestClassifier with `RF_PARAMS` from existing `trading_app/ml/config.py:277-285`. Conservative hyperparameters (max_depth=6, min_samples_leaf=100, n_estimators=200).

**Validation:** CPCV-45 (10 groups, k=2 test, purge=1d, embargo=1d) on train set; single OOS shot on holdout.

**Bootstrap:** 5,000 permutations with Phipson-Smyth (avoids resolution floor artifact). Null = shuffled labels.

**MinBTL budget:** Bailey et al 2013 strict bound at E[max_N]=1.0 for 6.65yr MNQ/MES clean data: `N ≤ 28 trials`. V3 target K = **8 trials** (well under strict Bailey budget). Trial enumeration to be finalized in Stage 2.

---

## What Stage 0 did NOT do

- Did not re-derive per-strategy trade lists via `build_eligibility_report` triple-join — used `validated_setups.sample_size` precomputed values instead. These were computed at discovery time with correct filter application. If Stage 3 produces suspect results, a re-derivation via `build_eligibility_report` is the first debugging step.
- Did not empirically test `LiveMLPredictor` with an actually-empty `models/ml/` dir in a unit test — verified by source inspection only. Stage 1 will add the test as part of the fail-closed fix.
- Did not audit GC retired strategies for "genuinely dead vs prematurely retired" — out of V3 scope. If V3 fails and Stage 4 delete happens, GC residual research is a separate sprint.
- Did not verify `features.py:transform_to_features` correctly computes `orb_pre_velocity_norm` and `prior_sessions_broken` on current `daily_features` schema — Stage 3 will validate at runtime.

---

## Stage 1 is unblocked. Proceeding.
