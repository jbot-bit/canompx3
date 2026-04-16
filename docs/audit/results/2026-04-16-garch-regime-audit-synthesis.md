# Garch Regime Audit Synthesis

**Date:** 2026-04-16
**Purpose:** Re-audit the `garch_forecast_vol_pct` work under the repo's institutional framework after broad reruns showed strong local patches but weak global BH survival.

---

## 1. Methodology audit

### 1.1 Look-ahead / timing

- `garch_forecast_vol` is computed from prior daily closes only in [pipeline/build_daily_features.py](/mnt/c/Users/joshd/canompx3/pipeline/build_daily_features.py:581).
- `garch_forecast_vol_pct` is a prior-only rolling percentile rank in [pipeline/build_daily_features.py](/mnt/c/Users/joshd/canompx3/pipeline/build_daily_features.py:1217).
- The helper `_prior_rank_pct()` explicitly uses `rows[max(0, current_index - lookback):current_index]`, i.e. excludes the current row from the ranking window.
- Conclusion: the feature construction is **look-ahead clean**.

### 1.2 Holdout discipline

- The repo's canonical holdout boundary remains `2026-01-01` in [trading_app/holdout_policy.py](/mnt/c/Users/joshd/canompx3/trading_app/holdout_policy.py:1).
- The broad / validated `garch` scripts used `IS_END = "2026-01-01"` and reported OOS separately.
- Conclusion: the current `garch` research scripts are **holdout-disciplined at the script level**.

### 1.3 Statistical framework

- The `garch` scripts use standard BH step-up FDR, e.g. [research/garch_broad_exact_role_exhaustion.py](/mnt/c/Users/joshd/canompx3/research/garch_broad_exact_role_exhaustion.py:314).
- The repo's own broad-scan standard is multi-framing BH, not global-only, in [research/comprehensive_deployed_lane_scan.py](/mnt/c/Users/joshd/canompx3/research/comprehensive_deployed_lane_scan.py:615).
- The institutional literature layer distinguishes:
  - exploratory family testing via BH-FDR
  - single pre-registered mechanism claims via the individual pathway
- Conclusion: the `garch` work is **not failing because of incorrect BH math**. The open question is honest family definition.

---

## 2. Existing hard results

### 2.1 Validated-scope production question

Source: [research/garch_validated_role_exhaustion.py](/mnt/c/Users/joshd/canompx3/research/garch_validated_role_exhaustion.py:1)

- Strategies in scope: `45`
- Primary tests: `429`
- BH mean survivors: `0`
- BH sharpe survivors: `0`

Interpretation:
- On the actual validated strategy populations, `garch` does **not** currently clear the production-ready broad overlay question.

### 2.2 Broad exact role run

Source: [research/garch_broad_exact_role_exhaustion.py](/mnt/c/Users/joshd/canompx3/research/garch_broad_exact_role_exhaustion.py:1)

- Strategy rows in scope: `430`
- Primary tests: `2630`
- BH mean survivors: `1`
- BH sharpe survivors: `0`

Only BH mean survivor:

- `MES_EUROPE_FLOW_E2_RR1.0_CB1_ATR_P70 long low@30`
- `N_on=18`, `N_off=213`
- `lift=-0.775`
- `sr_lift=-1.384`
- `p_mean=0.000019`
- `p_sharpe=0.001998`

Interpretation:
- The only global survivor is **negative**, i.e. an avoid-low regime result, not a broad positive deployment result.

---

## 3. Raw surface asymmetry

From the completed broad exact run:

- High-side local cells (`p_sharpe <= 0.05`):
  - `96` cells
  - `91` positive
  - `5` negative
  - mean `sr_lift=+0.277`
  - mean `lift=+0.270`

- Low-side local cells (`p_sharpe <= 0.05`):
  - `91` cells
  - `16` positive
  - `75` negative
  - mean `sr_lift=-0.312`
  - mean `lift=-0.255`

Interpretation:
- The raw surface is strongly asymmetric.
- `high garch` behaves like a favorable regime on many cells.
- `low garch` behaves like a hostile regime on many cells.

---

## 4. Family-framed BH rerun

The missing question was whether the effect remains coherent under natural family boundaries rather than one global `K=2630` headline.

### 4.1 Session-side framing

- BH survivors at `K = session × side` families:
  - survivors: `9`
  - families with survivors: `3`

Families with survivors:

- `LONDON_METALS high`
  - family size `96`
  - survivor count `2`
  - best `p_sharpe=0.000999`
  - best `sr_lift=+0.654`
  - best `lift=+0.729`

- `SINGAPORE_OPEN high`
  - family size `84`
  - survivor count `2`
  - best `p_sharpe=0.000999`
  - best `sr_lift=+0.436`
  - best `lift=+0.415`

- `SINGAPORE_OPEN low`
  - family size `80`
  - survivor count `5`
  - best `p_sharpe=0.000999`
  - best `sr_lift=-0.626`
  - best `lift=-0.554`

Interpretation:
- Global BH says "not universal."
- Session-side BH says "some regime families are structured and non-random."

### 4.2 Instrument-session-side framing

- BH survivors:
  - `22`
  - families with survivors: `6`

Families with survivors:

- `MES LONDON_METALS high`
  - size `42`
  - survivors `1`
  - best `p_sharpe=0.000999`

- `MES SINGAPORE_OPEN high`
  - size `12`
  - survivors `4`
  - best `p_sharpe=0.000999`

- `MES SINGAPORE_OPEN low`
  - size `10`
  - survivors `6`
  - best `p_sharpe=0.000999`

- `MGC LONDON_METALS high`
  - size `6`
  - survivors `3`
  - best `p_sharpe=0.000999`

- `MES EUROPE_FLOW high`
  - size `24`
  - survivors `4`
  - best `p_sharpe=0.001998`

- `MES EUROPE_FLOW low`
  - size `22`
  - survivors `4`
  - best `p_sharpe=0.001998`

Interpretation:
- Once the family matches the market-state question more honestly, the surface stops looking like pure noise.
- The effect is still not universal, but it is also not confined to one lane.

### 4.3 Filter-side framing

- BH survivors:
  - `11`
  - families with survivors: `2`

Families with survivors:

- `ATR_P70 low`
  - family size `118`
  - survivors `8`
  - best `p_sharpe=0.000999`

- `PDR_R080 high`
  - family size `6`
  - survivors `3`
  - best `p_sharpe=0.000999`

Interpretation:
- The strongest repeating structure in the broad set is concentrated in specific filter families, not everywhere.

---

## 5. Full-surface sign consistency by session-side

All `2630` tests, not just local p-value shortlist:

- `COMEX_SETTLE high`
  - `n=180`
  - `169` positive / `11` negative
  - mean `sr_lift=+0.166`
  - mean `lift=+0.170`
  - min `p_sharpe=0.003`

- `COMEX_SETTLE low`
  - `n=152`
  - `29` positive / `123` negative
  - mean `sr_lift=-0.126`
  - mean `lift=-0.125`
  - min `p_sharpe=0.004`

- `EUROPE_FLOW high`
  - `n=126`
  - `111` positive / `15` negative
  - mean `sr_lift=+0.074`
  - min `p_sharpe=0.002`

- `EUROPE_FLOW low`
  - `n=124`
  - `34` positive / `90` negative
  - mean `sr_lift=-0.102`
  - min `p_sharpe=0.002`

- `CME_PRECLOSE high`
  - `n=138`
  - `96` positive / `42` negative
  - mean `sr_lift=+0.094`
  - min `p_sharpe=0.005`

- `CME_PRECLOSE low`
  - `n=111`
  - `34` positive / `77` negative
  - mean `sr_lift=-0.071`
  - min `p_sharpe=0.004`

- `SINGAPORE_OPEN high`
  - `n=84`
  - `63` positive / `21` negative
  - mean `sr_lift=+0.070`
  - min `p_sharpe=0.001`

- `SINGAPORE_OPEN low`
  - `n=80`
  - `19` positive / `61` negative
  - mean `sr_lift=-0.137`
  - min `p_sharpe=0.001`

- `LONDON_METALS high`
  - `n=96`
  - `63` positive / `33` negative
  - mean `sr_lift=+0.039`
  - min `p_sharpe=0.001`

- `LONDON_METALS low`
  - `n=79`
  - `19` positive / `60` negative
  - mean `sr_lift=-0.120`
  - min `p_sharpe=0.003`

- `NYSE_OPEN high`
  - `n=180`
  - `41` positive / `139` negative
  - mean `sr_lift=-0.053`
  - min `p_sharpe=0.022`

- `NYSE_OPEN low`
  - `n=178`
  - `77` positive / `101` negative
  - mean `sr_lift=-0.023`
  - min `p_sharpe=0.014`

Interpretation:
- Several sessions show a broad **high-good / low-bad** sign pattern.
- `NYSE_OPEN` is the main counterexample; its high side is net negative in the full broad set.

---

## 6. Institutional verdict

### 6.1 What is supported

- `garch_forecast_vol_pct` is **look-ahead safe**.
- The broad raw surface shows **non-random regime asymmetry**.
- The asymmetry is not just one lane:
  - `COMEX_SETTLE`
  - `CME_PRECLOSE`
  - `EUROPE_FLOW`
  - `SINGAPORE_OPEN`
  - `LONDON_METALS`
  all show favorable high-side vs hostile low-side structure.
- Family-framed BH indicates the effect can survive under **natural regime-family questions**, even though it fails the global universal-overlay question.

### 6.2 What is not supported

- `garch` is **not** validated as a universal production binary filter.
- `garch` is **not** validated as a broad overlay on the current validated strategy population.
- The current evidence does **not** justify weakening BH or changing thresholds post hoc.

### 6.3 Most likely correct role

The data is more consistent with:

- `R3` position-size modifier
- `R7` confluence / regime score

than with:

- universal `R1` skip/take filter

per the role map in [docs/institutional/mechanism_priors.md](/mnt/c/Users/joshd/canompx3/docs/institutional/mechanism_priors.md:72).

---

## 7. Missing work before any production decision

1. Pre-register natural `garch` regime families instead of testing it as one giant mixed global claim.
2. Re-run under holdout-clean discovery populations, not just current research-provisional validated rows.
3. Add continuous-role tests:
   - rank monotonicity
   - quantile slope
   - forecast-style scaling instead of only hard thresholds
4. Add destruction controls:
   - shifted / lag-broken `garch`
   - permuted `garch`
   - placebo rank feature
5. Add forward shadow monitoring before live use.

---

## Bottom line

The right conclusion is not:

- "`garch` failed"
- or "`garch` is universal"

The right conclusion is:

- global BH kills the universal claim
- family-framed BH and sign consistency support a real **regime-family effect**
- the effect looks most usable as a **market-state / sizing / danger-regime variable**
- production promotion still requires a pre-registered, holdout-clean, forward-judged regime program
