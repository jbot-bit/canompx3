# ML Layer Bloomey Fixes — Design Document

Date: 2026-03-09
Source: `/bloomey-review ML layer` — Grade B+
Scope: 5 findings across `trading_app/ml/`, ordered by severity.

## Literature Grounding

Every fix in this document is grounded in published methodology:

| Fix | Primary Source | Secondary |
|-----|---------------|-----------|
| OOS selection bias | White (2000) "Reality Check"; HLZ (2016) | Romano-Wolf (2005) |
| Sharpe delta p-values | Jobson-Korkie (1981); Lo (2002) | Memmel (2003) |
| Meta-label architecture | de Prado AIFML (2018) Ch.3 | — |
| CPCV validation | de Prado AIFML (2018) Ch.12 | — |
| Isotonic calibration | Niculescu-Mizil & Caruana (2005) | — |
| Feature importance | de Prado AIFML (2018) Ch.8 | — |

Engineering fixes (Finding 1: model format, Finding 5: vectorization) are
software correctness — no lit required.

---

## Finding 1: evaluate_validated.py Broken for Hybrid Models [IMPORTANT]

### Problem
`evaluate_validated.py:68` checks only legacy model path:
```python
model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
```
Production now uses hybrid per-session models (`_hybrid.joblib`). The script
either evaluates the wrong (stale legacy) model or reports "no model".

Same issue in `evaluate.py:78`.

### Blast Radius
- `evaluate_validated.py` — standalone CLI tool, no downstream importers
- `evaluate.py` — `_fill_missing_features()` imported by evaluate_validated.py:82
- `predict_live.py:104-107` — correct pattern (prefers hybrid → legacy fallback)
- Tests: `tests/test_trading_app/test_ml/test_meta_label.py` — does NOT test evaluate scripts
- No scripts import from evaluate_validated.py
- Impact: evaluation scripts produce misleading results when only hybrid models exist

### Fix Plan
1. **Both evaluate scripts**: add hybrid model detection matching predict_live.py pattern:
   ```python
   hybrid_path = MODEL_DIR / f"meta_label_{instrument}_hybrid.joblib"
   legacy_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
   model_path = hybrid_path if hybrid_path.exists() else legacy_path
   ```
2. **Hybrid evaluation**: for hybrid models, iterate per-session sub-models:
   - For each session with a model: predict on that session's data using the
     session's own RF + threshold + feature_names
   - Apply E6 filter + session-specific feature drops (matching meta_label.py training)
   - Report per-session and aggregate metrics
3. **Per-aperture support**: if `bundle_format == "per_aperture"`, iterate
   per (session, aperture) units

### Fortification
- Add drift check: "evaluate scripts can load current production model format"
- Add test: `test_evaluate_hybrid_model()` that creates a mock hybrid bundle
  and verifies evaluate_validated handles it correctly

---

## Finding 2: Selection-Biased OOS Aggregation [IMPORTANT]

### Literature Basis
This is the **multiple testing problem at model selection level**:
- White (2000) "Reality Check for Data Snooping" — testing K models on the
  same OOS data and keeping winners inflates reported performance.
- Romano-Wolf (2005) "Stepwise Multiple Testing" — formal framework for
  sequential model rejection.
- Harvey, Liu & Zhu (2016) "...and the Cross-Section of Expected Returns" —
  the principle: always report the FULL set of tests, not just survivors.
  Same paper we use for BH FDR in strategy discovery.

### Problem
`meta_label.py:559`:
```python
total_honest_delta = sum(r.get("honest_delta_r", 0) for _, _, r in all_results if r["model_type"] == "SESSION")
```
This sums honest_delta_r ONLY from sessions that PASSED OOS gates. Sessions
where ML hurt (honest_delta_r < 0) are rejected (model_type="NONE") and
excluded from the total. The reported aggregate is cherry-picked upward.

With 8+ sessions and random noise, ~4 sessions will be OOS-positive by chance.
The total overstates true lift per White (2000).

### Blast Radius
- `meta_label.py:559` — total_honest_delta_r computation
- `meta_label.py:596` — stored in model bundle as `total_honest_delta_r`
- `meta_label.py:708-709` — returned in result dict
- `predict_live.py:531` — reads `total_honest_delta_r` from bundle for display
- `meta_label.py:733` — printed in per-session results
- Bundle `.joblib` files on disk — will have old format until retrained
- Tests: no assertions on total_honest_delta_r values

### Fix Plan
1. **Add `total_full_delta_r`**: compute OOS delta for ALL sessions, including
   rejected ones. For rejected sessions where ML was trained but failed gates,
   include their honest_delta_r (which is negative). For sessions with NO_MODEL
   due to insufficient data, use 0 (no model = no impact).

   ```python
   # Sessions that trained but failed OOS gates: they HAVE honest OOS data
   # — we just chose not to deploy. Include their negative delta.
   total_full_delta = sum(
       r.get("honest_delta_r", 0)
       for _, _, r in all_results
       if r.get("test_auc") is not None  # trained, regardless of outcome
   )
   ```

2. **Report BOTH metrics**:
   - `total_honest_delta_r` (deployed models only) — what you actually get
   - `total_full_delta_r` (all trained sessions) — honest aggregate
   - `selection_uplift` = honest - full — the cherry-pick bias

3. **Store in bundle**: add `total_full_delta_r` alongside existing field.
   predict_live.py can optionally display it.

4. **Update print_per_session_results**: show both numbers with clear labels.

### Fortification
- Log warning if `selection_uplift > 0.5 * total_full_delta_r` — more than
  half the reported lift is from session selection
- Add to behavioral audit: "ML total_honest_delta_r must have companion
  total_full_delta_r in model bundles"

---

## Finding 3: REL_VOL_SESSIONS Canonical Source [DOWNGRADED — ALREADY GUARDED]

### Problem (as reported)
`config.py:130-142` hardcodes REL_VOL_SESSIONS instead of importing from
SESSION_CATALOG.

### Actual State
**Already defended.** Drift check at `check_drift.py:2238-2244` verifies
REL_VOL_SESSIONS matches SESSION_CATALOG dynamic entries. Test at
`test_config.py:188-196` also guards this.

### Fix Plan
**No change needed.** The hardcoding with drift-check guard is the correct
pattern — it's an explicit, deliberate list with automatic staleness detection.
Deriving dynamically would be over-engineering (some sessions may not have
rel_vol data, so the list is intentionally curated).

### Fortification
Already fortified. No action.

---

## Finding 4: Sharpe Delta Without Statistical Test [MINOR]

### Literature Basis
- Jobson & Korkie (1981) "Performance Hypothesis Testing with the Sharpe
  and Treynor Measures" — the canonical test for Sharpe ratio comparison.
- Memmel (2003) "Performance Hypothesis Testing with the Sharpe Ratio" —
  corrected version for overlapping/correlated return streams.
- Lo (2002) "The Statistics of Sharpe Ratios" — shows Sharpe ratio SE is
  O(1/sqrt(N)), so visual differences are often noise with N < 500.

### Problem
`evaluate.py:230-236` and `evaluate_validated.py:163` display Sharpe before/after
deltas without a p-value. The delta could be noise. Violates the project's
"every quantitative claim needs a p-value" rule (MEMORY.md, RESEARCH_RULES.md).
Lo (2002) shows that Sharpe ratio standard errors are large — a 0.05 delta
is meaningless without a formal test.

### Blast Radius
- `evaluate.py:230-236` — Sharpe delta display
- `evaluate_validated.py:163` — Sharpe delta display
- `meta_label.py:991` — validation set Sharpe delta display (less critical:
  val set is for threshold optimization, not claims)
- `_jobson_korkie_p()` exists in `scripts/tools/select_family_rr.py:40-57` —
  NOT importable from a shared utility location
- `_sharpe()` is defined 3x: evaluate.py:31, evaluate_validated.py:59,
  scripts/tools/backtest_1100_early_exit.py:334

### Fix Plan
1. **Extract shared utilities**: Create `pipeline/stats.py` (or add to existing
   utility) with:
   - `per_trade_sharpe(pnl: pd.Series) -> float` — replaces 3 duplicate `_sharpe()`
   - `jobson_korkie_p(sharpe_a, sharpe_b, n_a, n_b, rho=0.0) -> float` — moved
     from select_family_rr.py

2. **Add JK p-value to evaluate scripts**: After computing Sharpe delta,
   compute and display JK p-value:
   ```python
   jk_p = jobson_korkie_p(b['sharpe'], f['sharpe'], n_total, n_kept, rho=0.7)
   print(f"  {'Sharpe':<18} ... {delta:>+12.3f}  (JK p={jk_p:.3f})")
   ```
   Use rho=0.7 (same trades, different subset = high correlation).

3. **Label clearly**: if JK p > 0.05, append "(not significant)" to the delta.

4. **Update select_family_rr.py**: import from shared utility instead of
   local `_jobson_korkie_p`.

### Fortification
- Drift check: "evaluate scripts must display p-value alongside Sharpe delta"
  (grep for Sharpe delta without JK)
- Behavioral audit rule: "Sharpe comparisons require JK test"

---

## Finding 5: iterrows() Performance in Filter Application [MINOR]

### Problem
`features.py:632` and `features.py:892` use:
```python
for idx, row in df.iterrows():
    ft = row["filter_type"]
    filt = ALL_FILTERS.get(ft)
    if filt.matches_row(row.to_dict(), orb_label):
        keep_mask[idx] = True
```
With 50K+ outcomes (MNQ has 100K+), this is O(N) Python loop with dict
allocation per row. Training takes longer than necessary.

### Blast Radius
- `features.py:632` — `load_validated_feature_matrix()`
- `features.py:892` — `load_single_config_feature_matrix()`
- `trading_app/config.py` — `matches_row()` interface on all filter subclasses
- Tests: `test_features.py` — tests feature loading end-to-end
- `meta_label.py` — calls both load functions during training

### Filter Types (vectorization analysis)
| Filter | Logic | Vectorizable? |
|--------|-------|---------------|
| NoFilter | Always True | Trivial (all pass) |
| OrbSizeFilter | `orb_{label}_size >= min_size` | Yes: `df[col] >= min` |
| VolumeFilter | `rel_vol_{label} >= min` | Yes: `df[col] >= min` |
| DirectionFilter | `orb_{label}_break_dir == dir` | Yes: `df[col] == dir` |
| DayOfWeekSkipFilter | `day_of_week not in skip_days` | Yes: `~df[col].isin(days)` |
| ATRVelocityFilter | Complex: session + vel_regime + comp_tier | Yes: 3 column checks |
| CompositeFilter | base AND overlay | Yes: vectorize each, AND masks |
| BreakSpeedFilter | `orb_{label}_break_delay_min <= max` | Yes: `df[col] <= max` |
| BreakBarContinuesFilter | `orb_{label}_break_bar_continues == bool` | Yes: `df[col] == val` |

**All filter types are simple column comparisons — fully vectorizable.**

### Fix Plan
1. **Add `matches_df()` to StrategyFilter base class**:
   ```python
   def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
       """Vectorized version: return boolean Series. Default falls back to iterrows."""
       return pd.Series(
           [self.matches_row(row.to_dict(), orb_label) for _, row in df.iterrows()],
           index=df.index,
       )
   ```

2. **Override in performance-critical filters** (OrbSizeFilter, VolumeFilter,
   CompositeFilter, ATRVelocityFilter):
   ```python
   # OrbSizeFilter example
   def matches_df(self, df: pd.DataFrame, orb_label: str) -> pd.Series:
       col = f"orb_{orb_label}_size"
       if col not in df.columns:
           return pd.Series(False, index=df.index)
       mask = pd.Series(True, index=df.index)
       if self.min_size is not None:
           mask &= df[col] >= self.min_size
       if self.max_size is not None:
           mask &= df[col] < self.max_size
       return mask
   ```

3. **Update features.py** — replace iterrows with group-by + vectorized:
   ```python
   keep_mask = pd.Series(False, index=df.index)
   for ft_name, group_idx in df.groupby("filter_type").groups.items():
       filt = ALL_FILTERS.get(ft_name)
       if filt is None:
           continue
       group = df.loc[group_idx]
       # Use first orb_label from group (all same within validated combo)
       orb_labels = group["orb_label"]
       for orb_label, sub_idx in group.groupby("orb_label").groups.items():
           sub = df.loc[sub_idx]
           keep_mask.loc[sub_idx] = filt.matches_df(sub, orb_label)
   ```

4. **Backward compatible**: `matches_df()` default implementation uses
   `matches_row()` loop, so existing/custom filters work without changes.

### Fortification
- Performance benchmark in test: "filter application on 50K rows completes
  in < 2 seconds" (currently ~15-30s with iterrows)
- No functional change — vectorized output must match iterrows output exactly.
  Add a test that runs both paths and asserts equality.

---

## Implementation Order

| Phase | Finding | Risk | Time | Dependency |
|-------|---------|------|------|------------|
| 1 | #2 OOS selection bias | None (additive metric) | 30 min | None |
| 2 | #1 evaluate_validated hybrid | None (fix broken script) | 1 hr | None |
| 3 | #4 Sharpe JK p-values | None (additive display) | 45 min | None |
| 4 | #5 iterrows vectorization | LOW (must match existing) | 1 hr | None |
| 5 | #3 REL_VOL_SESSIONS | **NONE** | 0 | Already guarded |

Phases 1-3 are independent and can be parallelized.
Phase 4 requires careful testing (vectorized must match iterrows exactly).
Phase 5 is a no-op.

---

## Drift Checks to Add

1. **"evaluate scripts support hybrid model format"** — grep evaluate*.py for
   `_hybrid.joblib` path check
2. **"ML bundles include total_full_delta_r"** — check newest .joblib files
3. **"Sharpe delta displays include JK p-value"** — grep evaluate*.py for
   `jobson_korkie` call near Sharpe display
4. **"No iterrows in features.py filter application"** — grep for `.iterrows()`
   in features.py load functions (after Phase 4)
