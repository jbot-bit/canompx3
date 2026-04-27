# Sizing-Substrate Diagnostic — Design Spec

**Date:** 2026-04-27
**Status:** APPROVED design (pre-implementation)
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

Substrate identification (canonical, verified before pre-reg lock):
- `ORB_G5` (lanes 1, 3, 6) → substrate is the ORB-size-class feature; the continuous original is `orb_<SESSION>_size` divided by the day's `atr_20`. Verify the formula in `trading_app/eligibility/builder.py` before locking.
- `COST_LT12` (lanes 4, 5) → substrate is the per-trade implied cost in points; continuous original derived from `pipeline/cost_model.py` cost computation against the lane's `avg_orb_pts`. Verify formula at pre-reg time.
- `ATR_P50` (lane 2) → substrate is `atr_20_pct`; binary filter splits at the 50th percentile.

**Tier-B — orthogonal continuous features (5 features per lane = 30 cells).**
The same five lane-conditional features applied to all 6 lanes:
- `rel_vol_<lane_session>` (the lane-specific relative-volume column).
- `overnight_range_pct`.
- `atr_vel_ratio`.
- `garch_forecast_vol_pct`.
- `pit_range_atr`.

Tier-B exists so the diagnostic does not collapse to "test only the variable you already trust." A passing Tier-B cell would mean the lane has continuous structure orthogonal to its deployed filter — a stronger sizing case than Tier-A alone.

### 5.2 Per-cell procedure

For one cell (lane L, feature f):

1. Pull the lane's IS trade tape from `orb_outcomes` filtered by L's (instrument, orb_label, orb_minutes, rr_target, confirm_bars, entry_model). Sample window: 2010-06-07 to 2025-12-31 inclusive. **Raise** if any returned row has `trading_day >= 2026-01-01`.
2. Join `daily_features` to attach feature `f` per (trading_day, symbol).
3. Drop rows with NULL `f`. Record drop count.
4. Compute Spearman ρ and two-sided p-value against `pnl_r`.
5. Bin trades into quintiles by `f`. Compute mean `pnl_r` per quintile (Q1 = lowest f, Q5 = highest f). Record monotonicity flag (Q1≤Q2≤Q3≤Q4≤Q5 OR Q1≥Q2≥Q3≥Q4≥Q5).
6. Compute baseline ExpR (mean `pnl_r`, qty=1).
7. Compute sized ExpR using linear-rank weights `w = {0.6, 0.8, 1.0, 1.2, 1.4}` indexed by quintile, with the weight assigned by direction-of-edge: if mean Q5−Q1 is positive, Q5 gets 1.4; if negative, Q1 gets 1.4 (so the diagnostic always sizes UP into the edge). The mean weight is 1.0 by construction → dollar-vol matched.
8. Bootstrap (10,000 resamples, with replacement, preserving trade rows) the sized-ExpR minus baseline-ExpR delta. Record 95% CI.
9. Split-half stability: split the IS trade rows by median trade date for this lane. Recompute ρ and sized-vs-flat delta on each half. Record sign of each.

### 5.3 Cell pass criteria

A cell passes only if ALL hold:
- `|ρ| ≥ 0.10`.
- Quintile lift is monotonic AND `|Q5_mean_R − Q1_mean_R| ≥ 0.20`.
- Sized-vs-flat ExpR delta 95% bootstrap CI is strictly positive.
- Two-sided p-value for ρ survives BH-FDR at q=0.05 over the full K=48 family.
- Split-half: sign of ρ matches in both halves AND sign of sized-vs-flat delta matches in both halves.

Any single failure ⇒ cell fails. No partial credit.

### 5.4 Lane and global pass

- Lane has substrate ⇔ ≥1 of its 8 cells passes.
- Substrate confirmed globally ⇔ ≥3 lanes have substrate.

## 6. Boundary discipline

- **Read-only.** No DB writes. No edits to `pipeline/` or `trading_app/`. The diagnostic script lives under `research/` or `scripts/audit/` (final location chosen at writing-plans time).
- **Pre-registered.** A YAML file at `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml` MUST exist and be committed before the diagnostic script runs. The YAML carries the full feature list, weight schema, thresholds, and literature extracts.
- **Holdout sealed.** Script raises `RuntimeError` if any retrieved row has `trading_day >= 2026-01-01`. Test for this in unit tests.
- **Single-pass.** No re-running with different feature lists, different weight schemas, different thresholds, or expanded K. Reopen requires a new pre-reg with new mechanism citation.
- **Mechanism citations required at pre-reg:** literal extracts from `resources/Robert Carver - Systematic Trading.pdf` (Ch. 8 forecast-scaling) and `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` (Ch. 19 bet-sizing). Pasted into the YAML, not paraphrased from training memory. Per CLAUDE.md "Local Academic / Project-Source Grounding Rule."
- **MinBTL accounting.** K=48 is well under the 300-trial Bailey ceiling (`docs/institutional/HANDOFF.md` Phase 0 grounding). No additional Stage-1 cells will be added.
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
| Substrate-feature definition for `ORB_G5`/`COST_LT12` is wrong | Verify formula in `trading_app/eligibility/builder.py` and `pipeline/cost_model.py` before pre-reg lock; cite line numbers in YAML. |
| Holdout leakage via lookahead in feature construction | All features used are already in canonical `daily_features`, computed by pipeline at end-of-trading-day for that day's outcomes. The pipeline already enforces no-lookahead at build time. Spot-check: confirm `daily_features` for trading_day D uses only data up to D's session boundary. |
| Quintile binning unstable for small N | Lane #2 (SINGAPORE_OPEN, N≈137 trailing 12mo, IS will be much larger). Historical IS per lane is on the order of 1000–3000 trades, so quintile bins of 200–600 are stable. Document per-cell N in result. |
| Bootstrap underestimates serial dependence | Trades are largely independent across days. Same-day clustering exists between lanes but within-lane same-day events are rare. Acceptable for Stage 1; if substrate is confirmed, Stage 2 must consider block bootstrap. |
| ChatGPT recommendation drift in implementation | The diagnostic spec is locked; implementation must follow this YAML. Any deviation is treated as a void run per §8. |
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
- `resources/Robert Carver - Systematic Trading.pdf` — forecast scaling (literal extract required at pre-reg).
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` — bet sizing (literal extract required at pre-reg).
- `resources/false-strategy-lopez.pdf` — Bailey selection-bias frame.
- `memory/MEMORY.md` — current state, NO-GO registry, validated signals.

---

## Spec self-review (2026-04-27, post-write)

**Placeholder scan:** No TBDs. All thresholds, weights, K, q, and B specified.
**Internal consistency:** Section 3 lane count (6) matches Section 5.1 cell math (6×8=48) matches Section 4 BH-FDR family size matches Section 6 K. Decision rule in §4 (≥3 lanes) matches global pass in §5.4. Sample window in §5.2 (≤2025-12-31) matches holdout discipline in §6.
**Scope check:** Single Stage-1 diagnostic; explicitly defers sizing-function definition, portfolio simulation, and lane changes to Stage 2 or out-of-scope. Decomposed correctly.
**Ambiguity check:** Sized-weight direction in §5.2 step 7 specifies sign convention (always size UP into the edge). §5.3 specifies "ALL hold" for cell pass. §6 specifies pre-reg-before-run. No two-way reading found.
