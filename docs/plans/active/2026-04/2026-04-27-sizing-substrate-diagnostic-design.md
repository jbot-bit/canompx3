---
status: active
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Sizing-Substrate Diagnostic — Design Spec

**Date:** 2026-04-27
**Status:** APPROVED design (pre-implementation), v0.2 hardened with literature grounding pass
**Revision:** v0.2 corrects literature citations (Carver Ch. 7 not Ch. 8; Lopez Ch. 19 not in local resources, replaced with ML4AM §1.7; Aronson Ch. 6 was wrong, dropped). Adds five gaps: ex-ante direction-of-edge, forecast-stability gate, selection-bias-compounding acknowledgment, power-analysis per cell, NULL-coverage guard, reproducibility seed.
**Risk tier:** high (research-stage; not deployment)
**Authority:** Stage-1 falsifier for the "convert binary filters into continuous sizing" thesis
**Companion artifact (to be written at pre-reg time):** `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`

---

## 1. Purpose

Decide, on falsifiable evidence, whether the live system has *substrate* for a Carver-style continuous-forecast sizing layer — or whether the binary filters already capture all available information.

The thesis under test (refined from external feedback, then grounded against repo truth): the live system trades fixed `qty=1`. Each deployed lane uses a binary filter that is itself a thresholded version of an underlying continuous feature. If the continuous feature has monotonic predictive power for realized R that the binary collapses, then per-trade sizing proportional to the continuous signal can extract more EV from existing edges without new discovery.

The thesis is false if continuous versions of the deployed-filter substrates do not show monotonic R-lift, or if simulated rank-based sizing does not produce a portfolio-EV improvement over flat qty=1 with bootstrap CI excluding zero.

This is a **diagnostic**, not a deployment artifact. Passing cells qualify a Stage-2 sizing-model pre-registration; they do not authorize live changes.

## 2. Scope and non-scope

### In scope
- Read-only analysis over `orb_outcomes` and `daily_features`.
- The 6 deployed lanes in `docs/runtime/lane_allocation.json` (rebalance 2026-04-18, profile `topstep_50k_mnq_auto`).
- Two feature tiers (Tier-A: filter-substrate continuous originals; Tier-B: orthogonal continuous features). Total K = 48 cells.
- Per-cell metrics: Spearman ρ, Q1→Q5 mean-R lift, monotonicity, sized-vs-flat ExpR delta with bootstrap CI, split-half sign stability.
- BH-FDR control across the full K=48 family at q=0.05.
- One markdown result artifact + one JSON twin.

### Explicitly out of scope
- Cross-feature interactions (NO-GO mined: see registry entries for "Pre-velocity × atr_vel_regime", "New-on-new confluence stacking", IBS, NR7).
- Adding new features to `daily_features`. Diagnostic uses existing canonical columns only.
- Production code changes in `pipeline/` or `trading_app/`. The diagnostic is a stand-alone script under `research/` or `scripts/audit/`.
- Any allocator change, lane change, or qty change in live config.
- Discovery on 2026 holdout data. Holdout boundary 2026-01-01 is sealed.

## 3. Live ground truth (verified at design time)

Source: `docs/runtime/lane_allocation.json`, `rebalance_date: 2026-04-18`, `profile_id: topstep_50k_mnq_auto`, MNQ-only, `qty=1` (verified at `trading_app/live/session_orchestrator.py`).

| # | strategy_id (short) | ORB | RR | Filter | N (12mo) | ExpR | WR |
|---|---|---|---|---|---|---|---|
| 1 | EUROPE_FLOW_E2_O5 | O5 | 1.5 | ORB_G5 | 264 | +0.189 | 51.5% |
| 2 | SINGAPORE_OPEN_E2_O15 | O15 | 1.5 | ATR_P50 | 137 | +0.241 | 53.3% |
| 3 | COMEX_SETTLE_E2_O5 | O5 | 1.5 | ORB_G5 | 253 | +0.176 | 50.2% |
| 4 | NYSE_OPEN_E2_O5 | O5 | 1.0 | COST_LT12 | 262 | +0.120 | 57.3% |
| 5 | TOKYO_OPEN_E2_O5 | O5 | 1.5 | COST_LT12 | 236 | +0.093 | 46.6% |
| 6 | US_DATA_1000_E2_O15 | O15 | 1.5 | ORB_G5 | 257 | +0.079 | 44.7% |

Three distinct filter classes are deployed: `ORB_G5` (3 lanes), `COST_LT12` (2 lanes), `ATR_P50` (1 lane). Each is a binary derivative of a continuous quantity already in `daily_features` (289 columns; key continuous features include `atr_20`, `atr_20_pct`, `atr_vel_ratio`, `rel_vol_<SESSION>`, `overnight_range_pct`, `pit_range_atr`, `garch_forecast_vol_pct`, `gap_open_points`).

Trade tape `orb_outcomes`: 8.73M rows, 2010-06-07 → 2026-04-26, 8 symbols, with per-trade `pnl_r`, `mae_r`, `mfe_r`, plus the join keys (`trading_day`, `symbol`, `orb_label`, `orb_minutes`, `rr_target`, `confirm_bars`, `entry_model`).

## 4. Hypothesis

**H0 (null):** For every deployed lane, every continuous feature in scope satisfies all of:
- Spearman |ρ| < 0.10 against realized `pnl_r`,
- Q1→Q5 mean-R lift is non-monotonic or absolute Q5−Q1 difference < 0.20R,
- Sized-vs-flat ExpR delta 95% bootstrap CI includes zero.

In plain words: existing binary filters already capture the predictive content; sizing on continuous features adds variance without EV.

**H1 (substrate exists):** At least one (lane, feature) cell passes all three quantitative gates AND survives BH-FDR at q=0.05 across the full K=48 family AND survives split-half sign stability.

**Decision rule:**
- ≥3 lanes have at least one passing cell ⇒ **substrate confirmed**, pre-register Stage-2 sizing model.
- 1–2 lanes have a passing cell ⇒ **substrate weak**, document and park; do not Stage-2.
- 0 lanes pass ⇒ **thesis killed**; record as NO-GO with this design as the entry, reopen requires new mechanism citation.

The decision rule is binding once pre-reg is locked. No re-running with different feature lists, different K, or different thresholds.

## 5. Test design

### 5.1 Cell definition

A *cell* is a (lane, feature) pair. K = 48 cells total = 6 lanes × 8 features.

**Tier-A — filter substrate (3 features per lane = 18 cells).**
For each lane, the continuous original of its deployed binary filter, expressed three ways to capture different functional forms:
- The raw substrate value at trade time (lane-relative if the filter is lane-specific).
- The same substrate normalized by `atr_20_pct` (vol-adjusted form).
- The same substrate as a percentile rank over the trailing 252 trading days (regime-relative form).

Substrate identification (verified at design time against `trading_app/config.py`):
- `ORB_G5` (lanes 1, 3, 6) → `OrbSizeFilter`, `min_size=5.0` ("ORB size >= 5 points"). Continuous substrate: `orb_<SESSION>_size` (raw points). Lane-relative form: divided by `atr_20`. Vol-adjusted form: divided by `atr_20_pct`.
- `ATR_P50` (lane 2) → `OwnATRPercentileFilter`, median-split on `atr_20`. Continuous substrate: `atr_20_pct` (the trailing-percentile rank used for the binary cut).
- `COST_LT12` (lanes 4, 5) → `CostRatioFilter`. Continuous substrate: cost-to-target ratio in R units, computed via `pipeline.cost_model.get_session_cost_spec(instrument, orb_label)` divided by the lane's expected R distance. Final formula must be re-derived from the filter class implementation at pre-reg time and the line numbers cited in YAML.

**Tier-B — orthogonal continuous features (5 features per lane = 30 cells).**
The same five lane-conditional features applied to all 6 lanes:
- `rel_vol_<lane_session>` (the lane-specific relative-volume column).
- `overnight_range_pct`.
- `atr_vel_ratio`.
- `garch_forecast_vol_pct`.
- `pit_range_atr`.

Tier-B exists so the diagnostic does not collapse to "test only the variable you already trust." A passing Tier-B cell would mean the lane has continuous structure orthogonal to its deployed filter — a stronger sizing case than Tier-A alone.

**Correlation note for Tier-A.** The 3 functional forms per substrate (raw, vol-normalized, percentile-rank) are correlated by construction. BH-FDR remains valid under positive dependence (Benjamini-Yekutieli 2001 §2 — BH controls FDR under "positive regression dependence"), but effective independent K is smaller than 18 in Tier-A. This is acknowledged, not corrected: BH-FDR is conservative under dependence, so passing cells are if anything stronger evidence than the q value suggests. We do NOT switch to BY 2001 because BY's correction is excessively conservative for our case.

### 5.2 Per-cell procedure

For one cell (lane L, feature f):

1. Pull the lane's IS trade tape from `orb_outcomes` filtered by L's (instrument, orb_label, **orb_minutes** [O5 or O15 per lane], rr_target, confirm_bars, entry_model). Sample window: 2010-06-07 to 2025-12-31 inclusive. **Raise `RuntimeError`** if any returned row has `trading_day >= 2026-01-01`. Note that this trade tape is already conditional on the lane's deployed binary filter being TRUE — the diagnostic tests for continuous structure *among trades that already pass the filter*, which is the exact substrate that a sizing layer would act on.
2. Join `daily_features` to attach feature `f` per (`trading_day`, `symbol`, `orb_minutes`). Use the lane's `orb_minutes` for the join — O5 and O15 features are stored in distinct rows.
3. Drop rows with NULL `f`. Record drop count and drop fraction. **NULL-coverage guard:** if drop_fraction > 0.20 the cell is marked **INVALID (insufficient feature coverage)** rather than passing or failing — it produces no decision signal. This prevents NULL-correlated edge from contaminating ρ.
4. **Pre-cell power check:** with N rows after NULL drop, the standard error of Spearman ρ is approximately `1/√(N-2)`. Require N such that detecting `|ρ|=0.10` has implied `t = 0.10·√(N-2) ≥ 3.00` (Harvey-Liu-Zhu 2015 with-theory threshold via Chordia 2018, per `pre_registered_criteria.md` §Criterion 3 t-rule). This requires `N ≥ 902`. Cells with N < 902 are marked **UNDERPOWERED** (decision: cannot pass, can fail definitively only if ρ point estimate is near zero with tight CI; otherwise UNVERIFIED). Document per-lane IS-N in the result.
5. **Pre-cell ex-ante direction:** before computing ρ, declare the predicted sign of edge per (lane, feature) in the pre-reg YAML based on mechanism. Examples: ORB_G5 substrate `orb_size` predicted POSITIVE (Crabel "compression precedes expansion" → bigger ORB has bigger trend post-break, but actual sign needs explicit Carver/Crabel mechanism citation in YAML); `rel_vol` predicted POSITIVE (higher participation → cleaner break). Cells whose realized sign matches the pre-registered prediction get a *prediction-confirmed* flag; cells where realized sign opposes the prediction may still pass numerically but are flagged *prediction-flipped* and require a written mechanism-revision note before any Stage-2 use. This blocks the post-hoc "size up into whichever direction the data shows" loophole.
6. Compute Spearman ρ and two-sided p-value against `pnl_r` (the canonical outcome; not `ts_pnl_r` which is the trailing-stop variant).
7. Bin trades into quintiles by `f`. Compute mean `pnl_r` per quintile (Q1 = lowest f, Q5 = highest f). Record monotonicity flag (Q1≤Q2≤Q3≤Q4≤Q5 OR Q1≥Q2≥Q3≥Q4≥Q5).
8. Compute baseline ExpR (mean `pnl_r`, qty=1, current live behavior).
9. Compute sized ExpR using linear-rank weights `w = {0.6, 0.8, 1.0, 1.2, 1.4}` indexed by quintile. **Direction is set by the pre-registered prediction in step 5**, not by realized data: if predicted positive, Q5 gets 1.4; if predicted negative, Q1 gets 1.4. The mean weight is 1.0 by construction → dollar-vol matched. This is a Stage-1 *detection* proxy; it is NOT Carver's actual continuous-forecast recipe (Carver Ch. 7: forecast scalar normalizes to abs-mean=10, cap=±20). The bounded ratio max/mean = 1.4 < 2.0 is the linear-rank analog of Carver's 2× cap.
10. **Forecast-stability gate (Carver Ch. 7 fn 78):** compute the rolling 252-trading-day standard deviation of `f` across the IS window. If `(SD_max - SD_min) / SD_median > 0.50` (i.e. >50% relative variation in scale across the window), the cell is marked **UNSTABLE** — passes only with mandatory annotation that any Stage-2 sizing must include explicit forecast normalization, not raw-feature use.
11. Bootstrap (10,000 resamples, with replacement, preserving trade rows) the sized-ExpR minus baseline-ExpR delta. **Random seed = 42 for reproducibility**; the same seed produces the same CI on re-run. Record 95% CI.
12. Split-half stability: split the IS trade rows by median trade date for this lane. Recompute ρ and sized-vs-flat delta on each half. Record sign of each.

### 5.3 Cell pass criteria

A cell passes only if ALL hold:
- Cell is not marked INVALID (NULL-coverage > 20%) or UNDERPOWERED (N < 902 after NULL drop, see §5.2 step 4).
- `|ρ| ≥ 0.10`.
- Quintile lift is monotonic AND `|Q5_mean_R − Q1_mean_R| ≥ 0.20`.
- Sized-vs-flat ExpR delta 95% bootstrap CI is strictly positive.
- Two-sided p-value for ρ survives BH-FDR at q=0.05 over the full K=48 family.
- Split-half: sign of ρ matches in both halves AND sign of sized-vs-flat delta matches in both halves.
- Realized sign of ρ matches the pre-registered prediction (step 5). A *prediction-flipped* cell never passes for Stage 2 promotion; it can only be re-eligible after a written mechanism-revision in a fresh pre-reg.
- Forecast-stability gate (step 10) — UNSTABLE cells may pass numerically but cannot be promoted to Stage 2 without an explicit forecast-normalization pre-reg in Stage 2.

Any single failure ⇒ cell fails. No partial credit.

### 5.4a Diagnostic-validity guardrail

If, after the run completes, **any single tier has ≥50% INVALID + UNDERPOWERED cells**, the diagnostic itself is declared **INCONCLUSIVE for that tier** rather than confirming H0. This prevents the failure mode where a substrate is undetectable for instrumental reasons (NULL-heavy feature, low-N lane) but H0 is reported as confirmed.

### 5.4 Lane and global pass

- Lane has substrate ⇔ ≥1 of its 8 cells passes.
- Substrate confirmed globally ⇔ ≥3 lanes have substrate.

## 6. Boundary discipline

- **Read-only.** No DB writes. No edits to `pipeline/` or `trading_app/`. The diagnostic script lives under `research/` or `scripts/audit/` (final location chosen at writing-plans time).
- **Pre-registered.** A YAML file at `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml` MUST exist and be committed before the diagnostic script runs. The YAML carries the full feature list, weight schema, thresholds, and literature extracts.
- **Holdout sealed.** Script raises `RuntimeError` if any retrieved row has `trading_day >= 2026-01-01`. Test for this in unit tests.
- **Single-pass.** No re-running with different feature lists, different weight schemas, different thresholds, or expanded K. Reopen requires a new pre-reg with new mechanism citation.
- **Mechanism citations required at pre-reg, literal extracts only.** Per CLAUDE.md "Local Academic / Project-Source Grounding Rule":
  - **Carver Ch. 7** (per-rule forecast scaling: target abs forecast = 10, cap at ±20, scalar = 10 / natural-mean-abs-forecast). The corrected citation — Ch. 8 is *combined-forecast diversification multiplier*, not per-rule scaling. Both Ch. 7 and Ch. 8 to be quoted.
  - **Carver Ch. 7, footnote 78** on forecast-stability ("forecasts should have well defined and relatively stable standard deviations") — grounds the §5.2 step 10 forecast-stability gate.
  - **López de Prado, *ML for Asset Managers* §1 / footnote 7** for the sign-vs-size decoupling principle. The full bet-sizing recipe (sigmoid of meta-label probability) lives in *Advances in Financial Machine Learning* (AFML, 2018a) Ch. 19, which is **NOT in our `resources/`**. The pre-reg either (a) acquires AFML and quotes Ch. 19 literally, or (b) explicitly states the sigmoid functional form is UNSUPPORTED in our local canon and pre-registers a simpler rank-based sizer with Carver-only justification. Default path: (b), defer sigmoid to Stage 2 with optional AFML acquisition.
  - **Bailey/López de Prado, *Pseudo-mathematics and Financial Charlatanism*** (`resources/Pseudo-mathematics-and-financial-charlatanism.pdf`) and *Deflated Sharpe Ratio* (`resources/deflated-sharpe.pdf`) for the selection-bias frame, with the explicit caveat (see §6 Selection-bias compounding below) that K=48 controls only the new diagnostic family and does NOT deflate for prior lane-discovery multiplicity.
  - **Aronson Ch. 6 dropped.** Ch. 6 of *Evidence-Based Technical Analysis* is "Data-Mining Bias: The Fool's Gold of Objective TA" — does not address quantization loss. The continuous-vs-binary doctrinal claim is grounded directly in **Carver Ch. 7**: *"forecasts shouldn't be binary... it's better to see forecasts changing continuously rather than jumping around."*
  - **Quintile-lift / decile-lift as a feature-evaluation method:** UNSUPPORTED in our local `resources/`. Justified on first principles (monotonicity of conditional mean E[Y|f∈Qk]). Pre-reg states this explicitly.
- **MinBTL accounting.** K=48 is well under the 300-trial Bailey ceiling (`docs/institutional/HANDOFF.md` Phase 0 grounding). No additional Stage-1 cells will be added.
- **Selection-bias compounding (silence acknowledged).** The 6 deployed lanes are themselves survivors of an earlier, larger trial space. K=48 BH-FDR controls *only* the new diagnostic family; it does NOT re-deflate for prior lane-discovery multiplicity. Any sizing edge layered on these lanes inherits their selection risk. This is a known limitation; Stage 2 deployment must apply DSR with the cumulative trial count (not just K=48).
- **Drift checks must pass** before commit of result artifacts (`pipeline/check_drift.py`).

## 7. Outputs

1. **Result markdown** at `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`. Sections: pre-reg link, environment (commit SHA, db SHA, rebalance date), per-cell table (lane, feature, ρ, p, q, Q1..Q5 means, monotonic flag, sized-vs-flat delta + CI, split-half signs, pass/fail), per-lane summary, global decision.
2. **Result JSON twin** at `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.json`. Same content, machine-readable. Field schema documented in the markdown.
3. **No** strategy-registry writes, no `validated_setups` rows, no allocator updates.

## 8. Success and failure criteria for the diagnostic itself

The diagnostic is *successful as a diagnostic* (independent of which way the answer goes) iff:
- Pre-reg YAML committed before any run.
- Script raises on any 2026 row.
- All 48 cells produce numbers (no silent NaNs swept under the rug; NULLs are reported as drop counts).
- BH-FDR computed across the full pre-registered K (not a subset).
- Bootstrap CIs computed with the pre-registered B=10,000 (not changed mid-run).
- Result artifacts contain the commit SHA of the diagnostic script and the db SHA of `gold.db` at run time.
- Drift checks pass post-run.

The diagnostic is *unsuccessful* (regardless of substrate decision) if any of the above fail. In that case the run is voided, fix the violation, run again — but only once. A second void requires escalation rather than another rerun.

## 9. What this design does NOT decide

- The exact Carver-style sizing function (linear, sigmoid, capped, vol-scaled). That is Stage 2, after substrate is confirmed and after Carver Ch. 8 + Lopez Ch. 19 are extracted and the function is pre-registered against them.
- Whether sizing improves *portfolio* SR after vol-targeting and correlation accounting. That requires a portfolio-level simulation in Stage 2.
- Whether the deployed binary filters should be replaced (no — they are evidence of edge; sizing complements, not replaces).
- Whether other lanes (paused, candidate, future) should be re-examined under the same lens. Out of scope.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Substrate-feature definition for `ORB_G5`/`COST_LT12`/`ATR_P50` is wrong | Verified at design time against `trading_app/config.py` (OrbSizeFilter, CostRatioFilter, OwnATRPercentileFilter); final formula re-derived from filter class implementations at pre-reg time and line numbers cited in YAML (§5.1). |
| Holdout leakage via lookahead in feature construction | All features used are already in canonical `daily_features`, computed by pipeline at end-of-trading-day for that day's outcomes. The pipeline already enforces no-lookahead at build time. Spot-check at pre-reg: confirm `daily_features` for trading_day D uses only data up to D's session boundary. |
| Quintile binning unstable for small N | §5.2 step 4 enforces N ≥ 902 power gate; cells below this are UNDERPOWERED and cannot pass. Document per-cell N in result. |
| NULL-heavy feature contaminating ρ | §5.2 step 3 NULL-coverage guard: drop_fraction > 20% ⇒ INVALID, not pass/fail. §5.4a tier-level guardrail: ≥50% INVALID/UNDERPOWERED in a tier ⇒ tier INCONCLUSIVE. |
| Post-hoc direction-of-edge ("size into whichever way the data points") | §5.2 step 5 ex-ante prediction in pre-reg YAML; prediction-flipped cells flagged and barred from Stage-2 promotion without mechanism revision. |
| Forecast-feature scale drift across the IS window | §5.2 step 10 Carver Ch. 7 fn 78 stability gate: rolling 252-day SD variation > 50% ⇒ UNSTABLE; UNSTABLE cells cannot be promoted to Stage 2 without forecast-normalization pre-reg. |
| Selection-bias compounding (lanes are themselves survivors) | §6 explicit acknowledgment; Stage 2 must apply DSR with cumulative trial count, not just K=48. |
| Bootstrap underestimates serial dependence | Trades are largely independent across days. Same-day clustering exists between lanes but within-lane same-day events are rare (one entry per session per day). Acceptable for Stage 1; Stage 2 must consider block bootstrap if substrate confirmed. |
| Reproducibility of bootstrap CIs across re-runs | §5.2 step 11 fixes random seed = 42; result artifact stamps commit SHA, db SHA, seed value (§7). |
| Tier-A's 3 functional forms are correlated | §5.1 correlation note: BH-FDR remains conservative-valid under positive dependence (BY 2001 §2). We do not switch to BY's correction because it is excessively conservative for our case. Effective independent K is smaller than 18 in Tier-A; passing cells are if anything stronger evidence than the q value suggests. |
| ChatGPT recommendation drift in implementation | The diagnostic spec is locked. Implementation must follow this design and the pre-reg YAML. Any deviation is treated as a void run per §8. |
| BH-FDR misapplied to non-pre-registered K | The pre-reg YAML enumerates all 48 cells before running. K=48 is fixed at YAML-commit time. Any post-hoc cell addition voids the run. |

## 11. Process to execute (after this spec is approved by the user)

1. User reviews this spec; revisions if requested.
2. Invoke `superpowers:writing-plans` to produce the implementation plan (script structure, test structure, exact SQL, exact bootstrap, exact output schema).
3. The implementation plan's Stage 1 is *writing the pre-reg YAML* including Carver/Lopez literal extracts and substrate-formula citations.
4. The implementation plan's Stage 2 is *script + tests*.
5. The implementation plan's Stage 3 is *running the diagnostic* against `gold.db` and writing the result artifacts.
6. Verification per CLAUDE.md "2-Pass Implementation Method": tests pass + drift pass + behavioral audit + self-review.

## 12. References

- `docs/runtime/lane_allocation.json` — live lanes (verified 2026-04-27).
- `trading_app/prop_profiles.py` — `ACCOUNT_PROFILES["topstep_50k_mnq_auto"]` (sole authority for live config).
- `trading_app/live/session_orchestrator.py` — `qty=1` confirmation.
- `pipeline/check_drift.py` — drift checks that must pass.
- `docs/institutional/pre_registered_criteria.md` — Criterion 3 BH FDR q=0.05 (Pathway A).
- `docs/institutional/HANDOFF.md` — Phase 0 grounding, MinBTL=300.
- `resources/Robert Carver - Systematic Trading.pdf` — Ch. 7 per-rule forecast scaling (target=10, cap=±20) + Ch. 7 fn 78 stability + Ch. 8 combined-forecast diversification multiplier. Literal extracts required at pre-reg.
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` — §1 / footnote 7 sign-vs-size decoupling principle. Sigmoid bet-sizing recipe (AFML 2018a Ch. 19) is **NOT** in our local resources; pre-reg defaults to Carver-only justification, AFML acquisition optional for Stage 2.
- `resources/Pseudo-mathematics-and-financial-charlatanism.pdf` + `resources/deflated-sharpe.pdf` + `resources/false-strategy-lopez.pdf` — Bailey/López selection-bias frame. K=48 controls only the new diagnostic family; does NOT deflate prior lane-discovery multiplicity.
- Aronson Ch. 6 citation **dropped** — original spec had it, but Ch. 6 is data-mining bias, not quantization loss. The continuous-vs-binary doctrinal claim is grounded directly in Carver Ch. 7.
- `memory/MEMORY.md` — current state, NO-GO registry, validated signals.

---

## Spec self-review (2026-04-27, v0.1 — superseded)

**Placeholder scan:** No TBDs. All thresholds, weights, K, q, and B specified.
**Internal consistency:** Section 3 lane count (6) matches Section 5.1 cell math (6×8=48) matches Section 4 BH-FDR family size matches Section 6 K. Decision rule in §4 (≥3 lanes) matches global pass in §5.4. Sample window in §5.2 (≤2025-12-31) matches holdout discipline in §6.
**Scope check:** Single Stage-1 diagnostic; explicitly defers sizing-function definition, portfolio simulation, and lane changes to Stage 2 or out-of-scope. Decomposed correctly.
**Ambiguity check:** Sized-weight direction in §5.2 step 7 specifies sign convention (always size UP into the edge). §5.3 specifies "ALL hold" for cell pass. §6 specifies pre-reg-before-run. No two-way reading found.

**v0.1 caveat (discovered post-write during /resources grounding):** Carver chapter attribution was wrong (forecast scaling is Ch. 7, not Ch. 8). Lopez Ch. 19 citation pointed to a chapter not in our local resources (AFML, not ML4AM). Aronson Ch. 6 was wrong topic. Direction-of-edge in step 7 was post-hoc on realized data — a look-ahead. Power analysis was missing. NULL handling was unguarded. No reproducibility seed. Forecast stability ungated. Selection-bias compounding unacknowledged. All addressed in v0.2 below.

## Spec self-review (2026-04-27, v0.2 — current)

**Placeholder scan:** No TBDs. All thresholds (ρ≥0.10, Q5−Q1≥0.20R, q=0.05, drop_fraction≤0.20, N≥902, SD-variation≤50%, B=10000, seed=42), all weights ({0.6, 0.8, 1.0, 1.2, 1.4}), all K (48), all sample windows (2010-06-07 → 2025-12-31), all decision thresholds (≥3 lanes for substrate-confirmed) explicit.

**Internal consistency:** §3 lanes (6) × §5.1 features (8) = §4 K (48) = §5.3 BH-FDR family = §6 K. §4 decision rule (≥3 lanes for substrate-confirmed) matches §5.4 global pass. §5.2 step 1 sample window (≤2025-12-31) matches §6 holdout seal. §5.2 step 4 power threshold (N≥902, t≥3.00) matches `pre_registered_criteria.md` Criterion 3 with-theory. §5.2 step 9 weight-direction sourced from §5.2 step 5 ex-ante prediction (no post-hoc lookahead).

**Scope check:** Single Stage-1 diagnostic. §9 explicitly defers: Carver actual sigmoid/scalar function (Stage 2), portfolio-SR simulation (Stage 2), lane changes (out-of-scope), AFML Ch. 19 acquisition (optional Stage-2 prerequisite). Decomposition is correct.

**Ambiguity check:**
- Direction-of-edge: §5.2 step 5 binds to pre-registered prediction; step 9 uses that prediction; no realized-data post-hoc loophole.
- Pass criteria §5.3: enumerated with "ALL hold"; mentions every gate (INVALID, UNDERPOWERED, ρ, monotonicity, sized-flat CI, BH-FDR, split-half, prediction-match, stability).
- §5.4a guardrail prevents the failure-to-confirm-as-evidence-of-null trap (≥50% INVALID/UNDERPOWERED ⇒ INCONCLUSIVE not H0-confirmed).
- §6 mechanism citations enumerate exact chapter/section/footnote and explicitly mark the AFML gap as UNSUPPORTED with two named alternatives.
- §6 selection-bias compounding limitation written, not silenced.

**Truth grounding:** §3 ground truth re-verified at design time against `docs/runtime/lane_allocation.json`, `trading_app/prop_profiles.py`, `trading_app/live/session_orchestrator.py:qty=1`, `trading_app/config.py` filter classes (lines 2980, 3072, 3274). Carver/Lopez/Bailey citations re-verified against `resources/` PDFs by literature-extraction pass; corrections applied.

**Bias check:** Selection-bias compounding flagged (§6 + §10). Confirmation-bias on direction handled by ex-ante prediction (§5.2 step 5). Survivorship-bias on lane set acknowledged. Outcome-on-the-same-data circularity in sized-vs-flat handled by ex-ante direction + bootstrap CI.

**Hardening summary applied:** ex-ante direction-of-edge prediction; forecast-stability gate (Carver fn 78); power floor (Harvey-Liu-Zhu t≥3.00); NULL-coverage guard with tier guardrail; reproducibility seed; correct Carver/Lopez chapter attribution; AFML gap explicitly stated; Aronson citation dropped; quintile-lift first-principles justification; selection-bias compounding written.

**Unresolved silences (intentionally deferred to Stage 2, documented):**
- Block-bootstrap for serial dependence (Stage 2 if substrate confirmed).
- Carver portfolio-vol target Ch. 9 (Stage 2 — per-trade weights here are dollar-vol-matched but not portfolio-vol-targeted).
- Carver Ch. 8 combined-forecast diversification multiplier (Stage 2 if multiple lanes promoted).
- AFML Ch. 19 sigmoid-of-meta-label functional form (Stage 2 if AFML acquired).
- Per-regime decomposition beyond split-half (Stage 2 if substrate confirmed and Stage-1 reveals heterogeneity).
