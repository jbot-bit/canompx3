---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ML Pipeline Zero-Context Audit

**Date:** 2026-03-21
**Type:** Adversarial, evidence-only, zero prior context
**Status:** AUDIT FINDINGS — requires response to each question

---

I now have sufficient code and log evidence to produce the full audit. Let me produce it.

---

## Ground-Truth Audit: MNQ ML Meta-Labeling Pipeline

Reviewing: C:/Users/joshd/canompx3/trading_app/ml/, scripts/tools/ml_bootstrap_test.py, scripts/tools/ml_exhaustive_sweep.py, logs/ml_sweep_*.log, logs/ml_bootstrap_5k_overnight.log, models/ml/meta_label_MNQ_hybrid.joblib (existence only).

---

### AUDIT QUESTION 1

**STAGE / ASSUMPTION**
The raw ORB baseline is positive at the exact configs being meta-labeled (NYSE_OPEN O30 RR2.0, US_DATA_1000 O30 RR2.0).

**QUESTION**
Open `logs/ml_sweep_rr20_flat.log` and `logs/ml_sweep_rr20_aperture.log`. For each survivor session-aperture combination, extract the reported Sharpe and baseline ExpR at the exact (session, aperture, RR, filter_type, entry_model) config the model trained on. Provide the raw numbers.

**WHAT A CORRECT ANSWER MUST SHOW**
For each PASS survivor: positive Sharpe and positive ExpR at the exact config, computed on the full dataset before splitting. Configs reporting `Sharpe=nan` or negative ExpR indicate a negative or flat baseline â€” the meta-label cannot improve what does not exist.

**FAILURE THIS WOULD EXPOSE**
The logs at lines 8-19 of `ml_sweep_rr20_flat.log` show `Sharpe=nan` for NYSE_OPEN E2 RR2.0 O5 NO_FILTER (N=1,219) and for US_DATA_1000 E2 RR2.0 O5 NO_FILTER (N=1,246). `Sharpe=nan` occurs when the strategy breaks even or produces all-loss rows where stddev is zero or returns are flat. The per-aperture sweep trained on NYSE_OPEN O30 and US_DATA_1000 O30, not O5, but those sessions' O30 Sharpe values are also `nan` in the per-aperture log (line 30, 40 in `ml_bootstrap_5k_overnight.log`). Sharpe=nan means the baseline ExpR divided by its standard deviation is undefined â€” i.e., either flat or the baseline is not profitable. The sweep proceeds and finds "honest_delta" without first confirming the baseline was positive. If the baseline is ~0 or slightly negative, a small threshold-selection artifact can produce a positive delta on the test set.

---

### AUDIT QUESTION 2

**STAGE / ASSUMPTION**
The config selection is deterministic: the "max_samples" picker always selects the same (entry_model, filter_type, confirm_bars, orb_minutes) per session across training and bootstrap runs.

**QUESTION**
In `load_single_config_feature_matrix` (`trading_app/ml/features.py`, lines 797-800), `ORDER BY v.sample_size DESC NULLS LAST` with a `ROW_NUMBER()` window picks the first row by sample size. If two configs have identical sample_size, the tie-breaking is undefined (no secondary sort). Show that the config printed in `ml_sweep_rr20_flat.log` at training time matches exactly the config loaded in `ml_bootstrap_test.py` for the same session. Quote the config lines from both logs for NYSE_OPEN.

**WHAT A CORRECT ANSWER MUST SHOW**
Both runs must show the same entry_model, RR, aperture, filter_type, and N for NYSE_OPEN. Any difference means the bootstrap is testing a different data population than the sweep trained on, invalidating the p-value.

**FAILURE THIS WOULD EXPOSE**
Non-deterministic config selection. The `ORDER BY v.sample_size DESC NULLS LAST` has no tiebreaker (file `trading_app/ml/features.py`, line 798). DuckDB does not guarantee row order within ties. If two configs for NYSE_OPEN have identical sample_size, the selected config can differ across runs. Bootstrap would shuffle labels from a different dataset than the model was trained on, producing a null distribution that does not correspond to the actual training run. The p-value would be invalid.

---

### AUDIT QUESTION 3

**STAGE / ASSUMPTION**
The 60/20/20 split used in the bootstrap exactly matches the split used in the exhaustive sweep for each survivor.

**QUESTION**
The bootstrap in `ml_bootstrap_test.py` (line 107-113) applies global index boundaries `n_train_end = int(n_total * 0.60)` across the full dataset of 33,046 rows, then filters to session indices. The sweep in `meta_label.py` (line 279-281) does the same. Both use the same query: `load_single_config_feature_matrix(..., bypass_validated=True)`. Confirm that `n_total` is identical in both runs by quoting it from both logs. For NYSE_OPEN O30: sweep reported 644 session rows; bootstrap reported 644 session rows. Show both lines.

**WHAT A CORRECT ANSWER MUST SHOW**
`n_total = 33,046` in both runs (per-aperture). `n_session = 644` for NYSE_OPEN O30 in both runs. Any difference means the test set composition differs from training.

**FAILURE THIS WOULD EXPOSE**
Split boundary drift. If the DB was rebuilt or outcomes were added between the sweep and the bootstrap run, n_total changes, which shifts the 60/20/20 boundaries. The test set in the bootstrap would include rows the model was trained on, or exclude rows it wasn't. This is confirmed as a risk since the bootstrap warns "1,418 rows still missing global features after backfill" â€” a count that indicates live data state, which can change. If n_total changed between sweep and bootstrap, reported delta and p-value are not from the same experiment.

---

### AUDIT QUESTION 4

**STAGE / ASSUMPTION**
The bootstrap shuffles only training labels, leaving val and test labels intact.

**QUESTION**
In `ml_bootstrap_test.py` lines 155-156: `shuffled_y_train = y_all.iloc[train_idx].values.copy(); np.random.shuffle(shuffled_y_train)`. The val and test labels come from `pnl_r[val_idx]` and `pnl_r[test_idx]` which are the real PnL from the full `y_all` target. However, the threshold is re-optimized on the val set with real pnl_r but shuffled-label model predictions. Show that this procedure correctly tests what is claimed: that the model's discrimination ability (not val-set threshold fitting luck) is responsible for the test-set delta.

**WHAT A CORRECT ANSWER MUST SHOW**
The null distribution correctly represents a model with zero discrimination but full ability to find a good threshold on val data. The test must show that the null distribution's mean is near 0R and that the null model cannot systematically beat the baseline.

**FAILURE THIS WOULD EXPOSE**
The null model is trained on shuffled labels but its threshold is optimized on real val pnl_r. A null model that happens to partition val data favorably (even by chance) can still show positive test-set delta. The null mean in the bootstrap results confirms this: NYSE_OPEN O30 null mean = +4.9R (log line 153), US_DATA_1000 O30 null mean = +14.7R (log line 1043 of summary). A null mean of +14.7R on a supposed null distribution is a red flag â€” it means random models are consistently finding +14.7R improvement on the test set, which inflates the threshold for the real model to clear. The real delta of +38.5R clears it, but a positive null mean indicates that the test set itself may have a positive skew or that threshold mining on val can extract real-pnl structure from any model, not just discriminating ones.

---

### AUDIT QUESTION 5

**STAGE / ASSUMPTION**
No FDR correction is applied across the 7 survivors tested in the bootstrap.

**QUESTION**
The SURVIVORS list in `ml_bootstrap_test.py` (lines 46-57) contains 7 configs. The bootstrap tests each independently at alpha=0.05. No Benjamini-Hochberg or Bonferroni correction is applied in the summary code (lines 255-264). With 7 independent tests at alpha=0.05, the family-wise false positive rate is 1-(0.95^7) = 30%. Show what the BH-corrected q-values would be for the 7 observed p-values from `ml_bootstrap_5k_overnight.log`.

**WHAT A CORRECT ANSWER MUST SHOW**
BH-corrected q-values: rank p-values (0.0016, 0.0176, 0.0376, 0.0512, 0.0546, 0.0930, 0.1190), apply BH(q) = p Ã— 7/rank at each step. The 3 reported PASSes at p < 0.05 may not all survive BH correction at q < 0.05. At minimum, p=0.0376 (rank 3) needs q < 0.05 Ã— 3/7 = 0.0214 â€” it does not pass. Only p=0.0016 and p=0.0176 survive BH at q < 0.05 (q=0.0112, q=0.0616 respectively â€” US_DATA_1000 O30 is borderline).

**FAILURE THIS WOULD EXPOSE**
Per-config p-values reported as if independent tests are the only tests performed. The 7 survivors were themselves selected from the full exhaustive sweep, which tested multiple RR levels, sessions, and apertures. The pre-bootstrap selection step (picking survivors by honest_delta > 0 and passing 4 OOS gates) is itself a selection procedure. Bootstrap p-values computed only on survivors are not corrected for the number of configurations in the sweep that were discarded. This is selection bias: White (2000) Reality Check requires testing ALL configurations, not only the survivors.

---

### AUDIT QUESTION 6

**STAGE / ASSUMPTION**
The bootstrap universe matches the sweep universe: all 7 survivors came from the same population of tested configs.

**QUESTION**
The SURVIVORS list in `ml_bootstrap_test.py` (lines 46-57) includes `("NYSE_OPEN", None, 2.0, 3.3)` (flat, all apertures combined) AND `("NYSE_OPEN", 30, 2.0, 33.5)` (per-aperture O30). These are overlapping â€” the O30 data is a subset of the flat config data. Both were trained on the same 33,046-row dataset but selecting different subsets for the session. Show whether the test-set rows for NYSE_OPEN flat (test_idx from 13,337 rows) and NYSE_OPEN O30 (test_idx from 33,046 rows) overlap.

**WHAT A CORRECT ANSWER MUST SHOW**
Non-overlapping test sets (different datasets: flat uses 13,337 rows, per-aperture uses 33,046). The p-values are therefore not independent: the NYSE_OPEN O30 signal drives both the flat and per-aperture results, and counting them as two independent PASS results overstates evidence.

**FAILURE THIS WOULD EXPOSE**
Double-counting of the same signal. NYSE_OPEN flat `p=0.0546 MARGINAL` and NYSE_OPEN O30 `p=0.0016 PASS` both measure whether NYSE_OPEN has a discrimination signal. If the same underlying data pattern drives both, reporting "2 of 7 passed" inflates the apparent evidence. The flat run uses 13,337 rows (O5 only), the per-aperture run uses 33,046 rows (all apertures), so they are not literally the same data, but the O30 subslice's signal is a driver in both.

---

### AUDIT QUESTION 7

**STAGE / ASSUMPTION**
Feature selection (E6 filter, constant-column drops) is performed on the full dataset before splitting, not on train-only data.

**QUESTION**
In `meta_label.py` line 274, `apply_e6_filter(X_all)` is called on the complete `X_all` before the 60/20/20 split is applied at lines 279-281. Then at lines 362-364, constant columns are dropped using `session_data = X_session.iloc[session_indices]` â€” the full session data across all splits. Show that this constant-column check uses only `session_indices[session_indices < n_train_end]` (train only) rather than all session indices.

**WHAT A CORRECT ANSWER MUST SHOW**
Line 361: `session_data = X_session.iloc[session_indices]` uses ALL session indices (train + val + test combined) to determine which columns are constant. A correct implementation would use only `X_session.iloc[train_idx]`.

**FAILURE THIS WOULD EXPOSE**
Feature selection on full data leaks test-set information. If a column is constant across ALL session rows (including test), it gets dropped. If it were NOT constant on the test set alone (i.e., it only appears constant because of the train distribution), it would be incorrectly excluded from the model. More critically, any feature that IS variable in test but constant in train would also be dropped incorrectly. The E6 noise filter (`apply_e6_filter`) is similarly applied to `X_all` before splitting (line 274). While E6 drops columns based on a fixed list defined in config, the constant-column drop (lines 362-364) is data-driven and uses full-session data â€” this is confirmed look-ahead in feature selection.

---

### AUDIT QUESTION 8

**STAGE / ASSUMPTION**
The `rsi_14_at_CME_REOPEN` feature, listed in `session_guard.py` `_ALWAYS_SAFE` (line 62), is actually known before all sessions and does not leak information.

**QUESTION**
`session_guard.py` line 62 claims `rsi_14_at_CME_REOPEN` is "computed from prior-day 5m bars, known before any session." But `trading_app/ml/config.py` lines 27-29 removed this feature from `GLOBAL_FEATURES` with the comment "RSI is mean-reverting (opposite of breakout logic). Computed at CME_REOPEN only â€” no mechanism for predicting LONDON_METALS ORB hours later." Show whether this feature is currently excluded from the ML feature pipeline OR included via `_ALWAYS_SAFE` in session_guard. Verify by running `grep -r "rsi_14_at_CME_REOPEN" trading_app/ml/` and checking whether it appears in the query that builds the feature matrix.

**WHAT A CORRECT ANSWER MUST SHOW**
`rsi_14_at_CME_REOPEN` does not appear in `GLOBAL_FEATURES`, `SESSION_FEATURE_SUFFIXES`, `TRADE_CONFIG_FEATURES`, or `CATEGORICAL_FEATURES` in `config.py`. It is listed in `_ALWAYS_SAFE` in `session_guard.py` but this only means it is safe IF it appears in a feature set. Since it was removed from `GLOBAL_FEATURES`, it should not be present in the feature matrix.

**FAILURE THIS WOULD EXPOSE**
Inconsistency between `session_guard.py` (declares it safe) and `config.py` (explicitly removed it). The risk is that a future refactoring adds it back to `GLOBAL_FEATURES` without noticing the config.py exclusion commentary, silently reintroducing it. The more immediate concern is whether the daily_features table still computes this column and whether `d.*` in the query SELECT brings it in as an unused column that nevertheless passes through `transform_to_features`. Confirmed from code: `load_single_config_feature_matrix` selects `d.*` from daily_features. If `rsi_14_at_CME_REOPEN` is in the daily_features table, it enters the DataFrame but is not in `GLOBAL_FEATURES`, so it is not added to `X` in `transform_to_features` line 383. However it could appear via the catch-all for `gap_type` and `atr_vel_regime` at lines 410-413. This needs verification.

---

### AUDIT QUESTION 9

**STAGE / ASSUMPTION**
The calibrator (IsotonicRegression) is fit on val data and used for inference. The take/skip decision uses raw RF probability vs the raw threshold, not the calibrated probability.

**QUESTION**
In `meta_label.py` lines 422-423, the calibrator is fit on `(val_prob_raw, y_val)`. In `predict_live.py` lines 399-406, the take/skip decision is `p_win_raw >= threshold` (raw), then lines 415-418 calibrate for display only. Confirm this matches the bootstrap, which at line 142-144 applies `test_prob_raw >= best_t` for the null delta computation â€” no calibration in the bootstrap. The real model's take/skip in deployment also uses raw. But the saved threshold was optimized on val raw probs at line 411. Show that the calibrator at inference time does not change the take/skip decision.

**WHAT A CORRECT ANSWER MUST SHOW**
The `take` boolean in `MLPrediction` is derived from `p_win_raw >= threshold` (raw), not from calibrated probability. This is confirmed by `predict_live.py` line 405: `take = p_win_raw >= threshold`. Calibrated `p_win` is returned but not used for the take decision.

**FAILURE THIS WOULD EXPOSE**
If a caller uses `p_win` (calibrated) instead of `take` for the skip/proceed decision, it would apply a different threshold to a different probability scale. The threshold was optimized at raw probability scale. Applying it to isotonically-calibrated probabilities would produce different skip rates. This is not a code bug in the current implementation but is a latent risk: `MLPrediction` exposes both `p_win` and `take`, and a caller that checks `p_win >= threshold` would be comparing calibrated probability to a raw-scale threshold.

---

### AUDIT QUESTION 10

**STAGE / ASSUMPTION**
The paper trader / replay system actually uses the ML filter in the path that produced the `+272.56R, 2606 trades` reported raw-baseline figure, and the ML layer is layered on top of this in a separate path.

**QUESTION**
In `paper_trader.py` the import line 32 shows `build_raw_baseline_portfolio`. Confirm that `--raw-baseline` mode bypasses `LiveMLPredictor` entirely and that no ML model is instantiated in that path. Then confirm the ML path (`--multi-rr` or standard) instantiates `LiveMLPredictor` and applies `MLPrediction.take` to gate trades. Show the code path in `execution_engine.py` or `paper_trader.py` where ML take/skip is applied.

**WHAT A CORRECT ANSWER MUST SHOW**
Two clearly separated code paths: `--raw-baseline` with no ML instantiation, and the standard path with `LiveMLPredictor.predict()` called per trade before execution.

**FAILURE THIS WOULD EXPOSE**
If both paths instantiate ML (even fail-open), the "raw baseline" benchmark is contaminated. Conversely, if the ML path silently falls back to fail-open for all sessions (because no model file exists for instruments other than MNQ), the ML path and raw baseline path produce identical results â€” making any reported ML uplift unverifiable.

---

### AUDIT QUESTION 11

**STAGE / ASSUMPTION**
The deployed model at `models/ml/meta_label_MNQ_hybrid.joblib` corresponds to the sweep/bootstrap runs described in the logs.

**QUESTION**
The model artifact exists at `C:/Users/joshd/canompx3/models/ml/meta_label_MNQ_hybrid.joblib`. The bootstrap was run on a dataset of 33,046 rows (per-aperture) from `ml_bootstrap_5k_overnight.log` at timestamp starting `04:16:11`. The sweep was run at `18:16:26`. Load the bundle and extract: `bundle['trained_at']`, `bundle['config_hash']`, `bundle['rr_target_lock']`, `bundle['bundle_format']`, `bundle['n_total_samples']`, and list all sessions with `model_type == 'SESSION'`. Verify that `config_hash` matches `compute_config_hash()` from the current `config.py`.

**WHAT A CORRECT ANSWER MUST SHOW**
`trained_at` must match the sweep timestamp. `n_total_samples` must match 33,046 (per-aperture run). `bundle_format` must be `"per_aperture"`. `rr_target_lock` must be `2.0`. Sessions with models must be the exact survivors from the sweep log.

**FAILURE THIS WOULD EXPOSE**
Model on disk may have been saved from a different run than what the bootstrap tested. The sweep produced a model; the bootstrap re-trained the same architecture independently without saving. If the deployed model was saved from a flat (O5-only) run (13,337 rows) but the bootstrap tested the per-aperture run (33,046 rows), they are testing different models. The config_hash drift warning in `predict_live.py` only warns but does not block inference.

---

### AUDIT QUESTION 12

**STAGE / ASSUMPTION**
Sessions from the 10-of-12 "fallback" path in `load_single_config_feature_matrix` (those coming from `orb_outcomes` directly with `filter_type='NO_FILTER'`) have a positive raw ORB baseline at the selected config.

**QUESTION**
In `ml_bootstrap_5k_overnight.log` lines 9 and 165, 10 sessions are sourced from `orb_outcomes` fallback with `filter_type='NO_FILTER'`. These include LONDON_METALS, EUROPE_FLOW, BRISBANE_1025, TOKYO_OPEN, SINGAPORE_OPEN, NYSE_CLOSE, CME_REOPEN. The fallback assigns `filter_type='NO_FILTER'`. For each of these fallback sessions, query `orb_outcomes` directly: what is the mean `pnl_r` for E2 RR2.0 CB1 O30 NO_FILTER? Provide the query and result.

**WHAT A CORRECT ANSWER MUST SHOW**
A SQL query like `SELECT orb_label, AVG(pnl_r), COUNT(*) FROM orb_outcomes WHERE symbol='MNQ' AND entry_model='E2' AND rr_target=2.0 AND confirm_bars=1 AND orb_minutes=30 GROUP BY orb_label ORDER BY orb_label`. Results must show positive mean pnl_r for each fallback session that received a model.

**FAILURE THIS WOULD EXPOSE**
Sessions with negative raw baselines entering the ML training pool. From the sweep log (lines 25-50 of `ml_sweep_rr20_flat.log`): LONDON_METALS OOS negative -12.9R, EUROPE_FLOW AUC=0.476, BRISBANE_1025 OOS negative -9.0R, COMEX_SETTLE OOS negative -11.8R. These sessions have demonstrably negative baselines but their outcomes are still in the 33,046-row training pool for the per-aperture models. The per-aperture model trains on all sessions jointly and then per-session, meaning the training data contains rows from sessions where pnl_r is systematically negative. At RR2.0, most sessions have ~37% win rate (baseline negative), and only NYSE_OPEN and US_DATA_1000 showed positive test deltas. The model was trained to discriminate within a pool where most sessions are losing money â€” this is the negative-baseline problem flagged in the audit.

---

### AUDIT QUESTION 13

**STAGE / ASSUMPTION**
The 3-way split (60/20/20) creates a test set that is chronologically separated from training data.

**QUESTION**
The global split at `n_train_end = int(33046 * 0.60) = 19,827` and `n_val_end = int(33046 * 0.80) = 26,436`. Confirm that the data is ordered `ORDER BY o.trading_day` in the query (confirmed at `features.py` line 546). For the per-aperture NYSE_OPEN O30 session (N=644), the split produces train=384, val=146, test=114 (from log line 48). What are the actual date ranges for train, val, and test? Run `SELECT trading_day FROM orb_outcomes WHERE symbol='MNQ' AND orb_label='NYSE_OPEN' AND orb_minutes=30 AND entry_model='E2' AND rr_target=2.0 ORDER BY trading_day` and identify the 384th, 530th, and last rows.

**WHAT A CORRECT ANSWER MUST SHOW**
Strictly increasing date ranges with no overlap between train/val/test. The cutoffs should be coherent with the overall 33,046-row dataset's date boundaries. Test set should cover recent data.

**FAILURE THIS WOULD EXPOSE**
The split is applied by global row index in the 33,046-row dataset, then filtered to session rows. This means the session-level split is NOT guaranteed to be chronological within the session. The global ordering ensures `trading_day` is monotonically non-decreasing in the full dataset, so `session_indices[session_indices < n_train_end]` gives earlier rows in session. However, if rows are ordered by `trading_day` globally and multiple sessions per day exist, the per-day ordering within a date may not be deterministic. This is an edge-case risk, not confirmed as a bug, but requires verification.

---

### AUDIT QUESTION 14

**STAGE / ASSUMPTION**
The `bypass_validated=True` flag in the bootstrap correctly replicates the training data source used in the sweep.

**QUESTION**
The sweep `ml_exhaustive_sweep.py` calls `trading_app.ml.meta_label` with `--skip-filter`. Inside `meta_label.py` line 224, `bypass_validated: bool = True` is the default. The bootstrap at line 89 explicitly sets `bypass_validated=True`. Both should produce the same "HYBRID config: 2 sessions from validated_setups, 10 from orb_outcomes fallback" message. Confirm that the 2 sessions from validated_setups are the same in both sweep and bootstrap runs by quoting the log lines.

**WHAT A CORRECT ANSWER MUST SHOW**
Both logs must show `HYBRID config: 2 sessions from validated_setups`. The 2 sessions must be the same sessions (CME_PRECLOSE with ATR70_VOL and COMEX_SETTLE with ATR70_VOL, inferred from Sharpe=0.249 and 0.247 in the per-session list).

**FAILURE THIS WOULD EXPOSE**
If the DB was modified between sweep and bootstrap (new validated_setups entries added/removed), the "2 sessions from validated_setups" count or identity could change. The bootstrap would then train on a different set of configs than the sweep, making the p-values invalid. The logs confirm "HYBRID config: 2 sessions from validated_setups" in both runs (bootstrap log line 9, sweep log line 4), so this appears consistent â€” but the identity of those 2 sessions is only inferable from the Sharpe values (CME_PRECLOSE and COMEX_SETTLE) and should be explicitly verified.

---

### AUDIT QUESTION 15

**STAGE / ASSUMPTION**
The `CPCV_PURGE_DAYS = 1` and `CPCV_EMBARGO_DAYS = 1` in `config.py` (lines 277-278) are actually enforced in the CPCV implementation.

**QUESTION**
Read `trading_app/ml/cpcv.py` and confirm whether `purge_days=1` and `embargo_days=1` are passed to the CPCV scoring function from `meta_label.py` at lines 378-388. The call is `cpcv_score(RandomForestClassifier, ..., n_groups=5, k_test=2, max_splits=10)` â€” no `purge_days` or `embargo_days` arguments are passed.

**WHAT A CORRECT ANSWER MUST SHOW**
If `cpcv_score` has `purge_days` and `embargo_days` parameters with defaults of 1, the CPCV is correctly purging and embargoing. If the parameters are missing or default to 0, there is no temporal separation between CV folds.

**FAILURE THIS WOULD EXPOSE**
CPCV without purge/embargo inflates AUC by allowing the model to see future trade outcomes in adjacent folds. The CPCV is used as Gate 2 (line 484) to reject models with AUC < 0.50. If the AUC is optimistically biased upward by 0.01-0.02 due to missing purge/embargo, models with true random-chance discrimination (0.50) may show apparent 0.51-0.52 AUC and pass the gate incorrectly.

---

### AUDIT QUESTION 16

**STAGE / ASSUMPTION**
All 7 bootstrap survivors were selected AFTER reviewing the full sweep logs â€” not pre-registered before the sweep ran.

**QUESTION**
The `SURVIVORS` list in `ml_bootstrap_test.py` lines 46-57 includes specific (session, aperture, RR, honest_delta) tuples with comments like `# p=0.005 PASS` â€” referring to the 200-permutation bootstrap in `ml_bootstrap_results.log`. This means the 5000-permutation bootstrap in `ml_bootstrap_5k_overnight.log` tested survivors that already had 200-permutation p-values attached. Were the 7 survivors pre-registered before the sweep, or were they selected by inspecting the sweep output? Show the git commit that created the SURVIVORS list.

**WHAT A CORRECT ANSWER MUST SHOW**
Either (a) the SURVIVORS list was committed before the sweep ran and the sweep results were not used to select them (pre-registration), or (b) the list was built post-sweep and this is acknowledged as a two-stage selection procedure.

**FAILURE THIS WOULD EXPOSE**
Post-hoc survivor selection without correcting the p-values for the full number of configs tested in the sweep. The sweep tested configs across 12 sessions Ã— 3 apertures = 36 config slots for the per-aperture run, plus 12 for the flat runs, plus RR1.5 = approximately 60 total (session, aperture, RR) combinations. Only 7 survivors were bootstrapped. The bootstrap p-values do not account for the ~53 discarded configurations. This is equivalent to testing 60 strategies and correcting for only 7.

---

### AUDIT QUESTION 17

**STAGE / ASSUMPTION**
The `session_guard.py` correctly prevents future-session features from entering the training and inference pipelines.

**QUESTION**
`features.py` lines 443-456 call `from pipeline.session_guard import is_feature_safe` and apply it to numeric columns. The guard runs within `transform_to_features`. However, the `d.*` SELECT in `load_single_config_feature_matrix` brings all `daily_features` columns into the DataFrame before `transform_to_features` is called. The `GLOBAL_FEATURES` list (5 features) is extracted at line 383, and session-prefixed columns are processed in `_extract_session_features`. But columns like `orb_NYSE_OPEN_size`, `orb_LONDON_METALS_break_dir`, etc., arrive in `df` via `d.*`. After `transform_to_features` runs `_extract_session_features`, it drops or processes them. Confirm that `_extract_session_features` correctly maps only the CURRENT session's columns to generic names and does NOT carry raw future-session columns into `X`.

**WHAT A CORRECT ANSWER MUST SHOW**
`_extract_session_features` (features.py lines 96-147) iterates over `df["orb_label"].unique()` and creates `values` only for the current session's mask. Other sessions' raw columns (e.g. `orb_NYSE_OPEN_size` when current session is TOKYO_OPEN) are NOT added to the returned `result` DataFrame.

**FAILURE THIS WOULD EXPOSE**
If any raw `orb_{FUTURE_SESSION}_{suffix}` column survives into `X` through a path other than `_extract_session_features`, the session_guard check at lines 447-454 would catch it â€” but only if `is_feature_safe(col, session)` correctly identifies the future-session prefix. This is the correct second-line defense, but the primary concern is whether the first-line extraction already isolates same-session data only. Confirmed from code: `_extract_session_features` correctly extracts only the traded session's columns. The session_guard is a defense-in-depth. However, `orb_vwap` and `orb_pre_velocity` appear in `SESSION_FEATURE_SUFFIXES` â€” these become generic `orb_vwap` and `orb_pre_velocity` after extraction. The question is whether these features for the current session are computed as of ORB close (pre-break). The config comment says "Session VWAP from trading day start to ORB start" â€” this is pre-break. Confirmed safe.

---

### AUDIT QUESTION 18

**STAGE / ASSUMPTION**
No model file exists for MGC, MES, or M2K. The live predictor fails-open for all instruments except MNQ.

**QUESTION**
The glob `models/ml/*.joblib` returns only `meta_label_MNQ_hybrid.joblib`. `predict_live.py` line 114 logs "No ML model for {inst} â€” will fail-open" for any missing instrument. `LiveMLPredictor` then returns `MLPrediction(p_win=0.5, take=True, threshold=0.5)` for all non-MNQ instruments. Confirm: does the paper_trader or execution_engine log show fail-open counts for MGC/MES? If the paper_trader for MGC produces trades under the ML path, every single one is a fail-open with `take=True`, meaning the ML layer has zero effect on MGC.

**WHAT A CORRECT ANSWER MUST SHOW**
Either (a) MGC/MES paper_trader is explicitly run with `--raw-baseline` bypassing ML entirely, or (b) it runs the ML path but logs show 100% fail-open for those instruments.

**FAILURE THIS WOULD EXPOSE**
If performance metrics reported for a "ML portfolio" include MGC/MES without noting they are full fail-open, the reported ML portfolio performance is identical to the raw baseline for those instruments. Any comparison of "ML portfolio" to "raw baseline portfolio" is invalid for instruments without a trained model â€” the ML wrapper adds overhead but zero signal.

---

### AUDIT QUESTION 19

**STAGE / ASSUMPTION**
The `validated_setups` table contains active rows for all instruments/sessions referenced in `LIVE_PORTFOLIO` in `live_config.py`.

**QUESTION**
`live_config.py` lines 160-199 define 30 specs (comment line 151: "30 unique specs"). Each spec has (orb_label, entry_model, filter_type). Each must resolve to at least one row in `validated_setups` with `status='active'` for each instrument not in `exclude_instruments`. Run: `SELECT COUNT(*) FROM validated_setups WHERE status='active'` and cross-reference against the 30 specs Ã— 3 instruments (MGC, MNQ, MES). Report how many (spec, instrument) pairs have zero matching rows in validated_setups.

**WHAT A CORRECT ANSWER MUST SHOW**
An explicit row count for each (spec, instrument) combination. Any zero-match pair means `build_live_portfolio` will log a warning and skip that spec for that instrument. A large number of skips indicates the strategy inventory has been partially depleted (E0 purge, E3 retirement, filter retirements) and the live portfolio is running on fewer strategies than the spec count implies.

**FAILURE THIS WOULD EXPOSE**
Broken spec references. E0 purge and E3 soft-retirement removed historical validated rows. If validated_setups was not rebuilt after these purges, specs referencing E2 strategies may have no rows for instruments where only E0 variants existed. The system logs warnings but does not fail â€” so a partially broken live portfolio can run silently with 60% of specs producing zero trades.

---

### AUDIT QUESTION 20

**STAGE / ASSUMPTION**
The `E6_NOISE_EXACT` list in `config.py` correctly identifies near-constant features and the rationale for each exclusion is verified by an actual importance analysis at the current data state.

**QUESTION**
`E6_NOISE_EXACT` (config.py lines 235-239) lists `confirm_bars`, `orb_break_bar_continues`, `orb_minutes`. The comment states "<1% importance, near-constant." However, `orb_minutes` appears in `TRADE_CONFIG_FEATURES` (line 113) where it would be added to `X`, then immediately removed by E6 filter (line 350). This is a no-op for `orb_minutes`. More critically, `confirm_bars` is in both `TRADE_CONFIG_FEATURES` (added to X at line 399) and `E6_NOISE_EXACT` (dropped at E6 filter). Confirm that the net effect is that `confirm_bars` and `orb_minutes` are added then immediately dropped â€” show the code path through `transform_to_features` then `apply_e6_filter`.

**WHAT A CORRECT ANSWER MUST SHOW**
`transform_to_features` adds `confirm_bars` and `orb_minutes` at lines 399-401. `apply_e6_filter` at lines 346-352 then removes them (they are in `E6_NOISE_EXACT`). Net: they enter and immediately exit. This is a known no-op, but it means the `TRADE_CONFIG_FEATURES` list contains dead entries that produce no ML features.

**FAILURE THIS WOULD EXPOSE**
If someone removes `apply_e6_filter` from a future training path (e.g., in a simplified experiment), `confirm_bars=1` (constant at the config being tested) would dominate as a near-constant column, potentially creating a spurious constant-value predictor that passes some sklearn checks but adds no information.

---

### AUDIT QUESTION 21

**STAGE / ASSUMPTION**
The calibrator (IsotonicRegression) is fit on val data but the val data was also used for threshold selection, making the calibrator's output unreliable.

**QUESTION**
In `meta_label.py` lines 422-424, the calibrator is fit on `(val_prob_raw, y_val)` â€” the same val set used for threshold optimization at lines 409-415. The calibrator maps raw probabilities to calibrated probabilities on the same data used to find the best threshold. Confirm: is the calibrator evaluated on any held-out data, or only on the data it was fit on?

**WHAT A CORRECT ANSWER MUST SHOW**
The calibrator is saved in the bundle and used at inference time. Its accuracy as a probability estimator (does P(win)=0.65 mean 65% wins?) is never evaluated on the test set â€” only on the val set it was trained on. The calibrated probability is used for "display/Kelly sizing" (line 408 comment).

**FAILURE THIS WOULD EXPOSE**
Using a calibrator fit on the same data as the threshold selection step biases the calibrated probabilities toward the threshold range. If a Kelly sizer uses `p_win` to size trades, it will use miscalibrated probabilities. The current code comments acknowledge this is "for display only" (predict_live.py line 408), but if any downstream code sizes contracts based on `p_win`, it will use unreliable estimates.

---

### AUDIT QUESTION 22

**STAGE / ASSUMPTION**
The `EUROPE_FLOW` and `LONDON_METALS` session order in `SESSION_CHRONOLOGICAL_ORDER` (config.py lines 188-201) is consistent with `session_guard.py` `_SESSION_ORDER` (lines 39-51), despite seasonal swapping.

**QUESTION**
`trading_app/ml/config.py` line 186 comment: "EUROPE_FLOW/LONDON_METALS swap order by season (winter EF=17:00 before LM=18:00, summer LM=17:00 before EF=18:00). Static ordering here uses summer convention." `pipeline/session_guard.py` `_SESSION_ORDER` (lines 44-45): `"LONDON_METALS"` before `"EUROPE_FLOW"` â€” summer convention. In `_extract_cross_session_features` (features.py lines 172-197), the `SESSION_CHRONOLOGICAL_ORDER` from `trading_app/ml/config.py` is used. For a winter day where EUROPE_FLOW (17:00) precedes LONDON_METALS (18:00), the static order places LONDON_METALS before EUROPE_FLOW â€” meaning EUROPE_FLOW's break direction is counted as a "prior session" for LONDON_METALS' training, when in winter EUROPE_FLOW has NOT yet occurred when LONDON_METALS breaks. Show whether the cross-session feature computation for LONDON_METALS includes EUROPE_FLOW break data in winter.

**WHAT A CORRECT ANSWER MUST SHOW**
`_extract_cross_session_features` uses `SESSION_CHRONOLOGICAL_ORDER[:session_idx]` where `session_idx` for LONDON_METALS = 4 (summer order: CME_REOPEN, TOKYO, BRISBANE_1025, SINGAPORE_OPEN, LONDON_METALS). Prior sessions are CME_REOPEN, TOKYO, BRISBANE, SINGAPORE â€” not EUROPE_FLOW. So cross-session contamination does NOT apply here.

**FAILURE THIS WOULD EXPOSE**
Static ordering is safe for LONDON_METALS (EUROPE_FLOW is position 5, after LM at position 4). However, EUROPE_FLOW at position 5 would see LONDON_METALS (position 4) as a prior session â€” but in winter, LONDON_METALS has NOT yet occurred when EUROPE_FLOW starts. In winter, EUROPE_FLOW (17:00) precedes LONDON_METALS (18:00). The cross-session feature `prior_sessions_broken` for EUROPE_FLOW would include `orb_LONDON_METALS_break_dir` as a prior session break count, when in winter LM is a FUTURE session relative to EF. This is a seasonal look-ahead contamination in the cross-session feature for approximately half the trading days.

---

### AUDIT QUESTION 23

**STAGE / ASSUMPTION**
The bootstrap's null distribution correctly represents the behavior of random models at the same sample size and the observed positive null means (+4.9R to +14.7R) do not indicate structural bias.

**QUESTION**
The null distribution for US_DATA_1000 O30 shows `null_mean = +14.7R, null_std = 24.4R` (from summary line 1043: `+38.5  +14.7  0.0176`). A null mean of +14.7R is 60% of the real delta of +38.5R. This is extremely high for a null distribution â€” random models should have null mean â‰ˆ 0. What causes the null mean to be positive? Inspect the `_optimize_threshold_profit` function (meta_label.py lines 134-183): the function only returns a threshold if `delta > best_delta = 0.0` (line 179) â€” meaning it only returns thresholds that beat the baseline on the val set. If no threshold beats the baseline, it returns `None` and `null_delta = 0.0` (bootstrap line 174-176). This produces a truncated null distribution where null_delta is floored at 0.

**WHAT A CORRECT ANSWER MUST SHOW**
The floor at null_delta = 0.0 (bootstrap line 174-176: `else: null_delta = 0.0`) means the null distribution is one-sided truncated: it can only be 0 or positive, never negative. This makes the null mean positive by construction. The p-value `(null_deltas >= verified_delta).sum()` is correct for a one-sided test, but the null mean statistic is misleading because it reflects the truncation, not genuine signal in the null models.

**FAILURE THIS WOULD EXPOSE**
A truncated null distribution systematically inflates the null mean and makes the p-value conservative (harder to pass). The p-value computation itself (`(null_deltas >= verified_delta)`) is still valid for a one-sided test. However, reporting "null mean = +14.7R" as evidence that random models can find +14.7R is misleading â€” it is entirely a consequence of the 0-floor. The US_DATA_1000 O30 case (real delta +38.5R, null max +36.6R) is particularly concerning: the null distribution's maximum of +36.6R is close to the real delta of +38.5R, and the p-value of 0.0176 means ~88 of 5000 random models produced deltas >= +38.5R. This is a marginally significant result driven partly by a high-variance null distribution.

---

### AUDIT QUESTION 24

**STAGE / ASSUMPTION**
The existing MNQ model artifact `meta_label_MNQ_hybrid.joblib` was saved from the per-aperture run (33,046 rows), and `predict_live.py` will correctly apply it only when `orb_minutes` matches `training_aperture` stored in each session entry.

**QUESTION**
`predict_live.py` lines 291-312 check `training_aperture != orb_minutes` and fail-open on mismatch. The per-aperture bundle format stores aperture keys like "O5", "O15", "O30" per session. At inference time, `aperture_key = f"O{orb_minutes}"` (line 280) is used to look up the sub-model. If the deployed strategies trade NYSE_OPEN at O5 (not O30), but the only trained aperture model for NYSE_OPEN is O30, what happens? The O5 key lookup returns `{}`, the model is None, and inference fails-open (take=True). Confirm this behavior by tracing lines 279-289.

**WHAT A CORRECT ANSWER MUST SHOW**
For a per-aperture bundle, `session_data.get("O5")` returns `{}` or missing, `session_info.get("model")` returns None, and the code at line 285-289 fails-open. The ML layer has zero effect on O5 NYSE_OPEN trades even if a valid O30 model exists.

**FAILURE THIS WOULD EXPOSE**
The ML layer provides no filtering for apertures not trained. If the live portfolio trades NYSE_OPEN at O5 and the model only trained on O30, 100% of NYSE_OPEN O5 trades are taken regardless of ML signal. This is documented fail-open behavior, but it means the reported "ML-filtered" portfolio includes unfiltered trades that inflate trade count and may dilute the ML uplift.

---

### AUDIT QUESTION 25

**STAGE / ASSUMPTION**
The constant-column drop at training time (meta_label.py lines 362-364) produces the same feature set as at inference time, because `_get_session_features` uses the same `feature_names` list saved in the bundle.

**QUESTION**
`meta_label.py` lines 362-364: `const_cols = [c for c in X_session.columns if session_data[c].nunique() <= 1]; X_session = X_session.drop(columns=const_cols)`. The resulting feature set is saved as `feature_names = list(X_session.columns)` at line 369. At inference (predict_live.py), `_align_features` uses this `feature_names` list to reconstruct the feature vector, filling missing columns with -999. If a feature was constant (e.g., `entry_model_E1 = 0` always) at training time but is variable at inference (because a new entry model is introduced), the feature would be absent from `feature_names` and thus absent from inference â€” the model was never trained on it. This is the correct behavior. But confirm the inverse: if a feature was NOT constant at training time but IS constant at inference (e.g., single session, single entry model in live), will `_align_features` fill it with -999 if it is absent from the inference row?

**WHAT A CORRECT ANSWER MUST SHOW**
`_align_features` at predict_live.py lines 510-519 fills missing columns: one-hot prefixes get 0.0, others get -999.0. If `entry_model_E1` is present in `feature_names` but the live row is E2 only (so `entry_model_E1 = 0`), the live transform produces `entry_model_E1 = 0` normally. The issue only arises if the live feature matrix's categoricals produce a different set of one-hot columns than training.

**FAILURE THIS WOULD EXPOSE**
The live `transform_to_features` call on a single row produces one-hot columns only for categories present in that row. If training had `entry_model_E1` and `entry_model_E2` (because both appeared in training data), but a live E2 row only produces `entry_model_E2`, then `entry_model_E1` is absent from `X` and filled with 0.0 by `_align_features` â€” which is the correct value. However, the E6 filter drops both `entry_model_E1` and `entry_model_E2` anyway (E6_NOISE_PREFIXES includes `"orb_label_"` but NOT `"entry_model_"`). Wait â€” `entry_model_E1` and `entry_model_E2` are observed dropped as constant columns in the sweep log (line 25: `dropped 2 constant cols: ['entry_model_E1', 'entry_model_E2']`), not by E6 filter but by the constant-column drop. At inference time on a single row, both `entry_model_E1=0` and `entry_model_E2=1` are produced â€” only one is 1.0. Since training dropped both as constants (all rows in training were the same entry model for that session), they appear in `feature_names = []` as absent. `_align_features` would fill them with 0.0. This is technically fine but the model was never trained with these features, so their 0.0 fill is irrelevant.

---

### AUDIT QUESTION 26

**STAGE / ASSUMPTION**
The EPV (Events Per Variable) constraint claimed in `config.py` comment line 302 ("V2 = â‰¤5 features") is actually enforced in training.

**QUESTION**
`config.py` line 302 comment: `V2 = methodology fix (â‰¤5 features, full universe, baseline+EPV gates)`. The actual E6-filtered feature count from the sweep log: `Single-config feature matrix: 33,046 rows x 45 features (win rate: 37.2%)`, then `E6 filter: dropping 20 noise columns` â†’ 25 E6 features remain (line 23 of `ml_sweep_rr20_flat.log`). For NYSE_OPEN O30 with train N=384, EPV = 384/25 = 15.4 (acceptable). But for CME_PRECLOSE with N=129 total (train roughly 78), EPV = 78/25 = 3.1. `config.py` cites "EPV=2.4" in the memory file. Show whether any gate in the training code checks EPV >= 10 (de Prado threshold) before training.

**WHAT A CORRECT ANSWER MUST SHOW**
No EPV gate exists in `meta_label.py`. The only sample-size gates are: `n_session < effective_min` (where `effective_min = 200` for single_config) and `len(val_idx) < 20` and `len(test_idx) < 20` (lines 339-346). CME_PRECLOSE with N=129 fails the `n_session < 200` gate â€” confirmed by sweep log showing CME_PRECLOSE came from `validated_setups` with ATR70_VOL filter (N=129), but it still trained (log line 30: `CME_PRECLOSE >> ML t=0.35 ... HonestDelta=+2.2`). Wait â€” 129 < 200. Check if `min_session_samples` is overridden.

**FAILURE THIS WOULD EXPOSE**
CME_PRECLOSE N=129 < effective_min=200, but the sweep trained a model for it and it passed gates. Either `effective_min` was lower (the `200 if single_config` logic may have been different), or the session-level check is bypassed for validated sessions. This indicates either the EPV comment is aspirational (not enforced in code) or the minimum sample size was met through a different path. The bootstrap final results show CME_PRECLOSE RR1.5 passed at p=0.0376 â€” this is the only PASS result with an acknowledged N<200 concern.

---

### AUDIT QUESTION 27

**STAGE / ASSUMPTION**
The `EUROPE_FLOW` and `BRISBANE_1025` sessions received E1 entry model configs from the fallback path, while all validated sessions use E2. This inconsistency may bias the feature matrix.

**QUESTION**
From sweep log lines 8, 12: `BRISBANE_1025 E1 RR2.0 CB1 O15 NO_FILTER N=1313` and `EUROPE_FLOW E1 RR2.0 CB1 O5 NO_FILTER N=1313`. Two sessions use E1, all others use E2. `entry_model` is in `CATEGORICAL_FEATURES` (config.py line 98) and gets one-hot encoded. After the constant-column drop (which drops `entry_model_E1` and `entry_model_E2` when all rows in a session are the same entry model), the model for E2-only sessions sees only `entry_model_E2` one-hot. For E1 sessions (BRISBANE, EUROPE_FLOW), only `entry_model_E1` is constant. Does this asymmetry cause any issue at inference if E2 is the only entry model in the live portfolio?

**WHAT A CORRECT ANSWER MUST SHOW**
At inference for BRISBANE_1025 (E1 session), the live predictor would receive `entry_model='E2'` if E2 strategies are deployed for that session â€” but the training model for BRISBANE_1025 was trained on E1 data. This is a training-inference mismatch in entry model for the BRISBANE_1025 session. The aperture guard (check training_aperture) would catch aperture mismatch but there is NO entry_model guard in `predict_live.py`. The model trained on E1 BRISBANE_1025 outcomes would receive E2 BRISBANE_1025 features at inference â€” different entry mechanics, different win rate, different fill behavior.

**FAILURE THIS WOULD EXPOSE**
Entry model mismatch between training and inference for sessions where the fallback path selected E1 but the live portfolio runs E2. The trained model for BRISBANE_1025 learned to discriminate E1 win/loss patterns. Applied to E2 trades, its predictions are miscalibrated. No guard in the code catches this â€” the aperture guard only checks `orb_minutes`, not `entry_model`. The `training_rr` guard (lines 321-343) only checks RR. Entry model mismatch is unguarded.

---

### AUDIT QUESTION 28

**STAGE / ASSUMPTION**
The exhaustive sweep and bootstrap confirm ML skill for NYSE_OPEN O30 RR2.0 only â€” no other configuration reached p < 0.05 under BH correction.

**QUESTION**
From `ml_bootstrap_5k_overnight.log` summary:
- NYSE_OPEN O30 RR2.0: p=0.0016, PASS
- US_DATA_1000 O30 RR2.0: p=0.0176, PASS
- CME_PRECLOSE RR1.5: p=0.0376, PASS
- US_DATA_830 O30 RR2.0: p=0.0512, MARGINAL
- NYSE_OPEN flat RR2.0: p=0.0546, MARGINAL
- US_DATA_1000 O15 RR2.0: p=0.0930, MARGINAL
- CME_PRECLOSE flat RR2.0: p=0.1190, FAIL

Apply BH correction to these 7 p-values (sorted ascending): p1=0.0016 (q_critical=0.05Ã—1/7=0.007), p2=0.0176 (q=0.05Ã—2/7=0.014), p3=0.0376 (q=0.05Ã—3/7=0.021), p4=0.0512 (q=0.05Ã—4/7=0.029), p5=0.0546 (q=0.05Ã—5/7=0.036), p6=0.0930 (q=0.05Ã—6/7=0.043), p7=0.1190 (q=0.05Ã—7/7=0.050). Which configurations survive BH at q < 0.05?

**WHAT A CORRECT ANSWER MUST SHOW**
p1=0.0016 < q_critical=0.007: PASS
p2=0.0176 > q_critical=0.014: FAIL (0.0176 > 0.0143)
p3-p7: all fail.

Under BH correction, **only NYSE_OPEN O30 RR2.0 survives as a genuine ML signal** from the 7-config family. US_DATA_1000 O30 and CME_PRECLOSE RR1.5 do not survive. Furthermore, the 7 configs are themselves the result of selection from a much larger sweep family (~60 configs), meaning even this BH correction is insufficient â€” it should be applied to the full sweep family.

**FAILURE THIS WOULD EXPOSE**
Reporting "3 of 7 passed bootstrap (p < 0.05)" as evidence of ML skill is incorrect under multiple-testing correction. The system is treating 7 independent hypothesis tests as if they are a single test. Under BH with 7 tests, only the most extreme p-value (0.0016) survives. The actual deployable ML signal is: one config (NYSE_OPEN O30 RR2.0), N_test=114 trades, delta=+33.5R on the test set.

---

## TOP 5 KILL-SHOT QUESTIONS

**KILL SHOT 1 (Question 28 / Audit Question 28)**
Apply BH correction to all 7 bootstrap p-values. The answer kills the "3 survivors" claim: only NYSE_OPEN O30 RR2.0 at p=0.0016 survives BH(7). But the 7 configs were themselves selected from ~60 sweep configs â€” BH should be applied to the full family. Under the full family, even p=0.0016 may not survive if the effective number of tested configs is large enough. This single computation determines whether the pipeline has any deployable ML signal at all.

**KILL SHOT 2 (Question 1 / Audit Question 1)**
Confirm that the Sharpe=nan at training time for all sessions except CME_PRECLOSE (Sharpe=0.249) and COMEX_SETTLE (Sharpe=0.247) means the raw baseline ExpR is approximately 0 or negative for the session-configs being meta-labeled. A negative baseline means any positive test-set delta is not "ML uplift on a profitable strategy" â€” it is "ML partially recovering a losing strategy." This violates de Prado's meta-labeling prerequisite and invalidates the conceptual framing of the entire approach.

**KILL SHOT 3 (Question 22 / Audit Question 22)**
The EUROPE_FLOW cross-session feature contains look-ahead contamination for LONDON_METALS in winter. `SESSION_CHRONOLOGICAL_ORDER` places LONDON_METALS before EUROPE_FLOW (summer convention), so EUROPE_FLOW is "prior" to LONDON_METALS in the static order. But in winter, LONDON_METALS (18:00) comes after EUROPE_FLOW (17:00). The `prior_sessions_broken` feature for LONDON_METALS includes EF break data in winter â€” which has not yet happened. This is verifiable by checking the EUROPE_FLOW vs LONDON_METALS DST start times and confirming which is position 4 vs 5 in `SESSION_CHRONOLOGICAL_ORDER`. If LONDON_METALS rows in winter have `prior_sessions_broken` that includes EUROPE_FLOW data, training data is contaminated for ~50% of LONDON_METALS rows.

**KILL SHOT 4 (Question 11 / Audit Question 11)**
Load the deployed model bundle at `C:/Users/joshd/canompx3/models/ml/meta_label_MNQ_hybrid.joblib` and inspect: `bundle['n_total_samples']`, `bundle['rr_target_lock']`, `bundle['trained_at']`, and which sessions have `model_type == 'SESSION'`. If `n_total_samples` is 13,337 (flat run) rather than 33,046 (per-aperture run), the deployed model is a different model than the one whose bootstrap was run at 5000 permutations â€” the p-values in the log do not apply to the deployed artifact. This is the single most important provenance check: does the on-disk model match the tested model?

**KILL SHOT 5 (Question 4 / Audit Question 4)**
The null distribution mean for US_DATA_1000 O30 is +14.7R with real delta +38.5R. The `_optimize_threshold_profit` function floors null_delta at 0 (line 174-176 of bootstrap). This means the null distribution is one-sided truncated. The effective null median is +0.0R (log line showing null_median close to zero), confirming the floor effect. Run the bootstrap but compute `null_delta = float(test_pnl[null_kept].sum()) - float(test_pnl.sum())` without the `else: null_delta = 0.0` floor â€” allowing negative null deltas. If the p-value drops significantly (e.g., from 0.0176 to 0.05+), the US_DATA_1000 O30 result is an artifact of the truncated null distribution. This computation directly determines whether the second-strongest ML signal is real or a truncation artifact.