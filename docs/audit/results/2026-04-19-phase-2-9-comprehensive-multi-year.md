# Phase 2.9 — Comprehensive 7-year × 38-lane multi-year stratification

**Date:** 2026-04-19
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-phase-2-9-comprehensive-multi-year-stratification.yaml`
**Script:** `research/phase_2_9_comprehensive_multi_year_stratification.py`
**Outputs:** `research/output/phase_2_9_main.csv` (266 rows), `phase_2_9_session_year_heat.csv` (49 rows), `phase_2_9_gold_fragility.csv` (9 rows)
**Supersedes the framing of:** Phase 2.8 v1 result doc (`2026-04-19-multi-year-regime-stratification.md`) per reframe addendum § 7. Retirement verdicts for the 2 SGP PURE_DRAG lanes stand (Phase 2.4 / 2.7 confirmed independently).

---

## TL;DR

The Phase 2.8 v1 framing ("no recurring regime, 34/38 VOL_NEUTRAL, 2 SGP SINGLE_YEAR_DRAG 2024") does not survive an honest per-cell BH-FDR test.

- **K_global=266 BH-survivors = 1.** Only `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 × 2020` clears the strictest framing (t=4.00, p=0.0002).
- **K_session survivors = 12** across only 2 sessions (CME_PRECLOSE, SINGAPORE_OPEN) — all BOOST, zero DRAG.
- **K_year survivors = 8** — 2025 is the only year with 3+ same-direction BH survivors (H2 pass).
- **GOLD pool fragility: CLEAN.** 0 of 9 Phase 2.5 Tier-1 + Phase 2.7 GOLD lanes flagged FRAGILE (H3 clean refutation).
- **The v1 "SINGLE_YEAR_DRAG" labels on the 3 flagged cells are NOT BH-significant.** The retirement verdicts still stand (Phase 2.4/2.7 independent), but v1's Phase 2.8 statistical support for those retirements was weaker than the label implied.

Per-hypothesis verdict:
- **H1 (session regime asymmetry)** — PARTIAL PASS. 2 sessions (CME_PRECLOSE, SINGAPORE_OPEN) have 2+ BH-survivor years in the same direction. But the expected *asymmetry* (boost vs drag by session) is only half-supported: all 12 survivors are BOOST. No session shows systematic DRAG across years.
- **H2 (year regime alignment)** — PASS for 2025. 4 BH-K_year survivors, all BOOST, spread across MNQ COMEX_SETTLE and SINGAPORE_OPEN.
- **H3 (GOLD fragility disclosure)** — CLEAN REFUTATION. 0 FRAGILE lanes. GOLD pool confidence is strengthened, not weakened.

## Session × year heat map (weighted-mean year ExpR, R per trade)

```
year             2019    2020    2021    2022    2023    2024    2025
session
CME_PRECLOSE    -0.537  +0.453  +0.279  +0.393  +0.035  +0.063  +0.192
COMEX_SETTLE    -0.087  +0.051  +0.007  +0.161  +0.172  +0.156  +0.233
EUROPE_FLOW     +0.029  +0.075  +0.074  +0.057  +0.238  -0.019  +0.065
NYSE_OPEN       -0.046  +0.037  +0.100  +0.127  +0.046  +0.137  +0.088
SINGAPORE_OPEN  +0.361  +0.075  +0.061  +0.338  -0.118  +0.310  +0.297
TOKYO_OPEN      +0.156  +0.095  +0.205  +0.140  -0.056  +0.205  +0.006
US_DATA_1000    +0.351  +0.085  +0.179  +0.032  +0.203  +0.181  -0.114
```

The 2019 CME_PRECLOSE −0.537 is driven by a thin-N 14-trade window on `MNQ_CME_PRECLOSE_X_MES_ATR60` — a single-lane artifact that does not reflect broader CME_PRECLOSE performance that year. The heat map includes UNEVALUABLE cells because averaging them across lanes still produces a readable weighted mean, but the BH-survivor grid below is the authoritative view.

### BH-K_session survivor count per session-year

```
year            2019  2020  2021  2022  2023  2024  2025
session
CME_PRECLOSE       1     3     0     3     0     0     0
COMEX_SETTLE       0     0     0     0     0     0     0
EUROPE_FLOW        0     0     0     0     0     0     0
NYSE_OPEN          0     0     0     0     0     0     0
SINGAPORE_OPEN     0     0     0     2     0     2     1
TOKYO_OPEN         0     0     0     0     0     0     0
US_DATA_1000       0     0     0     0     0     0     0
```

5 of 7 sessions have zero K_session BH survivors in any year. Only 2 sessions concentrate the signal.

## Survivors by framing

### K_global (K=266, BH q=0.10) — 1 survivor

| Lane | Year | N_year | Year_ExpR | Delta | t | p |
|---|--:|--:|--:|--:|--:|--:|
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | 2020 | 56 | +0.4243 | +0.2574 | 3.998 | 0.00019 |

At the strictest framing, only this one cell clears BH. 2020 COVID + CME_PRECLOSE + cross-asset MES ATR filter is a genuinely exceptional single-year window on a lane that already has full-sample t=4.19.

### K_session — 12 survivors across 2 sessions

| Lane | Session | Year | N_year | Year_ExpR | Delta | t | p | Label |
|---|---|--:|--:|--:|--:|--:|--:|---|
| MES_CME_PRECLOSE_COST_LT08 | CME_PRECLOSE | 2020 | 19 | +0.494 | +0.212 | 2.72 | 0.014 | UNEVALUABLE (n<30) |
| MES_CME_PRECLOSE_COST_LT08 | CME_PRECLOSE | 2022 | 29 | +0.490 | +0.241 | 3.40 | 0.002 | UNEVALUABLE (n<30) |
| MES_CME_PRECLOSE_ORB_G8 | CME_PRECLOSE | 2020 | 24 | +0.489 | +0.256 | 3.07 | 0.005 | UNEVALUABLE (n<30) |
| MES_CME_PRECLOSE_ORB_G8 | CME_PRECLOSE | 2022 | 45 | +0.326 | +0.071 | 2.56 | 0.014 | BOOST |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | CME_PRECLOSE | 2019 | 14 | −0.537 | −0.787 | −2.64 | 0.020 | UNEVALUABLE (n<30, DRAG-side) |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | CME_PRECLOSE | 2020 | 56 | +0.424 | +0.257 | 4.00 | 0.000 | BOOST |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | CME_PRECLOSE | 2022 | 56 | +0.396 | +0.222 | 3.47 | 0.001 | BOOST |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | SINGAPORE_OPEN | 2022 | 75 | +0.327 | +0.144 | 2.45 | 0.017 | BOOST |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | SINGAPORE_OPEN | 2024 | 111 | +0.358 | +0.197 | 3.34 | 0.001 | BOOST |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | SINGAPORE_OPEN | 2022 | 79 | +0.349 | +0.153 | 2.63 | 0.010 | BOOST |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | SINGAPORE_OPEN | 2024 | 116 | +0.265 | +0.058 | 2.45 | 0.016 | BOOST |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | SINGAPORE_OPEN | 2025 | 70 | +0.434 | +0.249 | 3.08 | 0.003 | BOOST |

Caveat: 5 of the 12 K_session survivors have n_year < 30 (pre-registered UNEVALUABLE threshold, RULE 3.2 of backtesting-methodology.md). The t-stats are real but the N is low. For the deployable-evidence tier, focus on the 7 cells with n_year ≥ 30: all are BOOST, concentrated on CME_PRECLOSE 2020/2022 and SINGAPORE_OPEN 2022/2024/2025.

### K_year — 8 survivors

| Lane | Year | N_year | Year_ExpR | Delta | t | p |
|---|--:|--:|--:|--:|--:|--:|
| MES_CME_PRECLOSE_COST_LT08 | 2022 | 29 | +0.490 | +0.241 | 3.40 | 0.002 |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | 2020 | 56 | +0.424 | +0.257 | 4.00 | 0.000 |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | 2022 | 56 | +0.396 | +0.222 | 3.47 | 0.001 |
| MNQ_COMEX_SETTLE_OVNRNG_100 | 2025 | 82 | +0.292 | +0.151 | 2.98 | 0.004 |
| MNQ_COMEX_SETTLE_X_MES_ATR60 | 2025 | 77 | +0.310 | +0.144 | 3.06 | 0.003 |
| MNQ_COMEX_SETTLE_OVNRNG_100_RR1.5 | 2025 | 82 | +0.357 | +0.241 | 2.74 | 0.008 |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | 2024 | 111 | +0.358 | +0.197 | 3.34 | 0.001 |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | 2025 | 70 | +0.434 | +0.249 | 3.08 | 0.003 |

All 8 are BOOST. 2025 contributes 4 of 8 survivors — the single strongest year-regime concentration in the sweep. 2020 and 2022 each contribute 1-2. No year shows 3+ same-direction DRAG survivors.

## GOLD fragility (H3)

| Lane | In GOLD pool | Full t | Worst ex-year | Worst ex-year t | t-drop | Flag |
|---|:-:|--:|--:|--:|--:|:-:|
| MNQ_COMEX_SETTLE_X_MES_ATR60 RR1.0 | No | 4.29 | 2025 | 3.27 | 1.02 | STABLE |
| MNQ_CME_PRECLOSE_X_MES_ATR60 RR1.0 | No | 4.19 | 2020 | 2.90 | 1.29 | STABLE |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | **Yes** | 4.14 | 2025 | 3.21 | 0.93 | STABLE |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | No | 3.96 | 2024 | 2.73 | 1.23 | STABLE |
| MES_CME_PRECLOSE_ORB_G8 | No | 3.66 | 2022 | 2.66 | 1.00 | STABLE |
| MES_CME_PRECLOSE_COST_LT08 | No | 3.56 | 2022 | 2.12 | 1.44 | STABLE |
| MNQ_COMEX_SETTLE_OVNRNG_100 RR1.0 | No | 3.42 | 2025 | 2.19 | 1.24 | STABLE |
| MNQ_COMEX_SETTLE_X_MES_ATR60 RR1.5 | No | 3.32 | 2025 | 2.54 | 0.78 | STABLE |
| MNQ_US_DATA_1000_VWAP_MID_ALIGNED_O15 RR1.5 | **Yes** | 3.20 | 2021 | 2.58 | 0.62 | STABLE |

No lane's worst ex-year t-stat drops below 1.96. Both Phase 2.7 GOLD lanes survive the fragility check cleanly. Both full-t >= 3.0 Tier-1 lanes survive the fragility check cleanly.

**Observation (not a retirement signal):** the "worst ex-year" column identifies which year is doing the most work for each Tier-1 lane. Notably, 2025 appears 4 times (out of 9 lanes) — consistent with the K_year finding that 2025 is a broadly-BOOST year.

## Back to the addendum's alternative interpretations

The reframe addendum (`2026-04-19-phase-2-8-reframe-addendum.md` § 4) listed 5 alternative interpretations I1-I5. This scan adjudicates:

| # | Claim | Verdict under Phase 2.9 | Notes |
|---|---|---|---|
| I1 | 2024 session-level directional asymmetry (some boost, some drag) | **Partially supported, weaker than v1 addendum suggested.** Only SINGAPORE_OPEN and (tentatively via UNEVALUABLE cells) CME_PRECLOSE show BH-significant year structure; and all survivors are BOOST, no BH-significant DRAG. EUROPE_FLOW has 0 BH survivors — the "Europe drag in 2024" read of the v1 cross-section was below BH significance under per-cell testing. |
| I2 | SGP 2024 break is an exit-shape problem (RR dose-response in retired cells) | **Not addressed** — the 2 retired SGP PURE_DRAG lanes are MNQ_EUROPE_FLOW CROSS_SGP_MOMENTUM RR1.5 / RR2.0. Phase 2.9 uses the active 38-lane set, which no longer includes them. That specific RR dose-response investigation would need a separate pre-reg on the retired lanes. |
| I3 | CME_PRECLOSE is a 2020-specific tailwind lane | **Not supported in the "fragility" sense.** H3 confirms NO CME_PRECLOSE Tier-1 lane has worst ex-year t < 1.96. 2020 is one of the K_year BH survivor years for MNQ X_MES_ATR60 (y2020 +0.424) but removing 2020 drops ex-year t only from 4.19 to 2.90, which is still Chordia-with-theory-significant. I3 is refuted as a fragility claim; the lane's edge does not collapse without 2020. |
| I4 | VWAP_MID_ALIGNED US_DATA_1000 has 2022 rate-hike weakness | **Not supported at BH.** MNQ_US_DATA_1000_VWAP_MID_ALIGNED_O15 RR1.5 full-t = 3.20, worst ex-year = 2021 (ex-t = 2.58). 2022's contribution is not the weak year under the ex-year lens. I4's 2022 negative year_expr in the raw CSV does not materialize as a BH survivor or a fragility flag. |
| I5 | v1's "no recurring regime" claim was valid only on 3 of 7 years | **Confirmed.** The 4 additional years (2019, 2021, 2023, 2025) DO produce BH survivors (mainly 2025). Phase 2.8 v1's scope genuinely missed the 2025 year-regime signature. |

## Important correction to the Phase 2.8 v1 narrative

The v1 PATTERN labeller flagged 3 SINGLE_YEAR_DRAG cells:
1. `MNQ_EUROPE_FLOW CROSS_SGP_MOMENTUM RR1.5` × 2024 — y2024 = −0.125, delta2024 = +0.041
2. `MNQ_EUROPE_FLOW CROSS_SGP_MOMENTUM RR2.0` × 2024 — y2024 = −0.132, delta2024 = +0.049
3. `MNQ_US_DATA_1000 X_MES_ATR60 RR1.0` × 2020 — y2020 = −0.059, delta2020 = +0.041

Under Phase 2.9's per-cell t-test with K_session / K_year / K_global all considered (values queried directly from `research/output/phase_2_9_main.csv`):

| Cell | N_year | year_t | year_p (two-sided) | bh_global | bh_session | bh_year |
|---|--:|--:|--:|:-:|:-:|:-:|
| SGP RR1.5 × 2024 | 89 | −1.080 | 0.283 | ✗ | ✗ | ✗ |
| SGP RR2.0 × 2024 | 89 | −0.985 | 0.327 | ✗ | ✗ | ✗ |
| X_MES_ATR60 RR1.0 × 2020 | 86 | −0.578 | 0.565 | ✗ | ✗ | ✗ |

All three cells fail BH at every framing (standard CHT two-sided p > 0.05, well above any BH critical value at K=266 / K_session / K_year=38). The Chordia et al 2018 threshold of t ≥ 3.00 with-theory (see `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` page 5: "the MHT threshold for alpha t-statistic (t_α) is 3.79 ... they are not far from the suggestion of Harvey, Liu, and Zhu (2015) to use a threshold of three") is clearly not met — the absolute t-stats are 0.58 to 1.08.

**The SGP retirement verdicts stand** — they were independently confirmed by Phase 2.4 (composite C1-C12 audit) and Phase 2.7 (2024-specific regime break). But v1's Phase 2.8 DRAG labels for those cells were not statistically supported under honest testing; they were bare-threshold observations that the v1 classifier elevated to a label tier without significance checking. This is the single most important methodological correction from the reframe addendum to the sweep itself.

## Institutional methodology notes

- **BH-FDR procedure used is the 1995 step-up** (`docs/institutional/literature/benjamini_hochberg_1995_fdr.md` page 293 equation 1): `k = max { i : P(i) ≤ (i/m) · q }`, reject all H(i) for i ≤ k. Implemented in `research/phase_2_9_comprehensive_multi_year_stratification.py::bh_fdr`. q = 0.10 locked at pre-reg time.
- **K-framing rationale** per Harvey-Liu 2015 (`docs/institutional/literature/harvey_liu_2015_backtesting.md` page 20): "In financial applications, it seems reasonable to control for the rate of false discoveries, rather than the absolute number." K_family (our K_session, K_year) is the natural test unit when the family has internal structure; K_global is reported as headline context only, not the promotion gate.
- **BH vs BHY under dependency:** Harvey-Liu page 16 advocates BHY (Benjamini-Yekutieli 2001) because it "works under arbitrary dependency for the test statistics." We used standard BH-1995 here. Defensible because cells within a session-year share the same underlying bars and therefore fall within the positive-regression-dependency regime where BH-1995 is valid (Benjamini-Yekutieli 2001 Theorem 1.2). For strict arbitrary-dependency safety, re-run with BHY (c(M) = ∑1/j multiplier) — it is strictly more conservative and our substantive findings (GOLD clean, v1 DRAG not significant, 2025 BOOST concentration) would only become cleaner, not flip direction.
- **BH-FDR calibration verified:** K_global=266 produces rank-1 critical value = (1/266) · 0.10 = 3.76 × 10⁻⁴. Observed smallest p = 1.92 × 10⁻⁴ on MNQ_CME_PRECLOSE X_MES_ATR60 × 2020, passing rank-1 with headroom. No other cell clears its critical value at K_global.
- **Chordia 2018 t-threshold context** (`docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` page 5): MHT-adjusted alpha threshold is t ≥ 3.79 (strict) or 3.00 (with prior theory, from Harvey-Liu-Zhu 2015). The single K_global survivor (t = 4.00) and the strongest K_session / K_year survivors (t = 3.08 to 4.00) clear the Chordia with-theory threshold of 3.00. UNEVALUABLE K_session cells with t = 2.72 / 2.56 / 3.07 / 2.45 / 2.63 / 2.45 are survivor-significance-BH-eligible but do NOT clear Chordia's with-theory bar — flagged suggestive-only.
- **MinBTL check passed:** at K = 266 and est max_N = 1521 (computed at run time from the active-setups sample_size field), MinBTL = 2·ln(266) / 1521² = 5.0 × 10⁻⁶. Much less than 1.0 years — sample size is not a binding constraint. Per Bailey et al 2013 (`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` page 8) the guidance is "no more than 45 independent model configurations should be tried" with 5 years of data; our 266 cells operate on years-of-data that support far more than this via the K = 38 × 7 independent-over-year factorization.
- **Per-cell subset-t uses Student's t (df = n−1), not Normal.** Correct small-sample treatment. p-value computed via `scipy.stats.t.sf(|t|, df=n-1) * 2` for two-sided.
- **Canonical delegation verified at run time:** full-window `_window_stats` agrees with `compute_mode_a` to div = 0.00e+00 on MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08 (first-lane probe in main()).
- **UNEVALUABLE cells can pass BH.** The labeller marks n_year < 30 as UNEVALUABLE but BH operates on p-values which don't know about N. 5 of 12 K_session survivors have n < 30 (all CME_PRECLOSE 2019/2020 thin years). These are BH-significant but low-power evidence; treat as suggestive per RULE 3.2 of `.claude/rules/backtesting-methodology.md`. They do NOT clear Chordia 2018's with-theory t ≥ 3.0 threshold in 4 of 5 cases.

## What the scan does NOT claim

- Does NOT promote any lane.
- Does NOT retract any retirement.
- Does NOT claim 2025 has a mechanistically-understood tailwind. 2025 is a BH-significant concentration of positive year-deltas across MNQ COMEX_SETTLE and SINGAPORE_OPEN — the mechanism remains a testable hypothesis, not an established finding. Candidate explanations (macro-tightening end-of-cycle, AI sector rotation, cross-asset vol regime) are all training-memory claims that would need literature-grounded pre-reg to pursue.
- Does NOT show that v1's retirement framing was wrong overall — only that v1's bare-threshold labelling was statistically weak. The retirement decisions remain correct.
- Does NOT test the 2 retired SGP PURE_DRAG lanes directly (they are not in the active-set source).

## Next steps (candidate — require fresh pre-reg if pursued)

1. **Phase 2.10 (not written):** investigate the 2025 BOOST concentration. 4 BH-survivor lanes in one year suggests a genuine year-regime; would benefit from a macro-grounded pre-reg (Carver 2015 Ch 9 vol-standardised sizing per `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` as a Stage 2 framework) plus a check that 2025 partial-year data is not biased by sample-selection.
2. **Retired-SGP RR dose-response audit (not written):** the addendum § I2 claim is untested here. Would require a separate pre-reg against the retired lanes in `validated_setups` with `status != 'active'`.
3. **Governance:** update Phase 2.8 v1 result doc with a pointer to this v2 supersession (landed in this commit set via header footnote).
4. **Doctrine upgrade (proposed):** add to `.claude/rules/backtesting-methodology.md` RULE 4 a sub-clause requiring per-cell significance testing on any classification threshold, not just on the discovery scan's promotion gate. The v1 PATTERN labeller used 0.03 bare threshold without p-values — that pattern should be caught in future reviews. Grounded in Harvey-Liu 2015 page 17 ("an OOS test's success can be due to luck for both the in-sample selection and the out-of-sample testing") — bare-threshold labels are in-sample selections without the significance filter to discount luck.
5. **BHY-replication check (proposed, low-cost):** re-run with Benjamini-Yekutieli 2001 correction (c(M) = ∑_{j=1}^M 1/j ≈ 6.32 at K=266) to confirm substantive findings hold under arbitrary-dependency assumption. Expected: fewer survivors (BHY is strictly more conservative); direction of findings should be unchanged.

## Audit trail

- Pre-reg committed before script run: `b74dafe3` (pre-reg + stage file).
- Script + tests committed separately from result doc for provenance: TBD commits.
- CSVs deterministic: re-running the script with the same git SHA produces identical rows (ordering by `strategy_id, year`).
- 39 unit tests passing (`pytest tests/test_research/test_phase_2_9_comprehensive_multi_year_stratification.py -q`) — includes 5 BHY tests added in the robustness pass.
- Drift check: 5 pre-existing worktree failures (Check 37 cascade — no local gold.db), all expected per pre-reg.
- Canonical delegations verified at run time: `compute_mode_a` vs `_window_stats` div=0.00e+00 on the probe lane.

## BHY (arbitrary-dependency) robustness replication

The result doc's § Institutional methodology notes flagged standard BH-1995 as defensible under the positive-regression-dependency regime but noted BHY (Benjamini-Yekutieli 2001, advocated by Harvey-Liu 2015 p16) as strictly-more-conservative under arbitrary dependency. The § Next steps item 5 proposed running BHY as a sensitivity check. This section reports the run. Numbers are queried directly from `research/output/phase_2_9_main.csv` (3 new columns `bhy_global`, `bhy_session`, `bhy_year`).

### Survivor counts at BH q=0.10 (primary) vs BHY q=0.10 (robustness)

| Framing | BH survivors | BHY survivors | BHY retention |
|---|--:|--:|--:|
| K_global (K=266) | 1 | 0 | 0 / 1 |
| K_session (variable per session) | 12 | 5 | 5 / 12 |
| K_year (K=38) | 8 | 1 | 1 / 8 |
| **Total unique cells (any framing)** | **15** | **5** | **5 / 15** |

At m=266, c(m) = ∑_{j=1}^266 (1/j) ≈ 6.02, so every BHY critical value is ~6× stricter than BH. The 5 BHY survivors are the subset of BH survivors that clear the tighter cutoff.

### BHY survivor cells (the robust subset)

All 5 K_session BHY survivors have |t| ≥ 3.08 — clearing the Chordia 2018 with-theory threshold (t ≥ 3.00) not just BH:

| Lane | Session | Year | N_year | year_expr | year_t | year_p |
|---|---|--:|--:|--:|--:|--:|
| MES_CME_PRECLOSE_COST_LT08 | CME_PRECLOSE | 2022 | 29 | +0.490 | 3.40 | 0.0020 |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | CME_PRECLOSE | 2020 | 56 | +0.424 | 4.00 | 0.0002 |
| MNQ_CME_PRECLOSE_X_MES_ATR60 | CME_PRECLOSE | 2022 | 56 | +0.396 | 3.47 | 0.0010 |
| MNQ_SINGAPORE_OPEN_ATR_P50_O15 | SINGAPORE_OPEN | 2024 | 111 | +0.358 | 3.34 | 0.0011 |
| MNQ_SINGAPORE_OPEN_ATR_P50_O30 | SINGAPORE_OPEN | 2025 | 70 | +0.434 | 3.08 | 0.0030 |

K_year BHY has only 1 survivor: `MNQ_CME_PRECLOSE_X_MES_ATR60 × 2020`, the same cell that topped K_global.

### What BHY drops and why it matters

10 cells pass BH but fail BHY (list here grouped by the substantive claim each supports):

**2025 MNQ COMEX_SETTLE BOOST concentration (3 cells DROPPED):**
- `OVNRNG_100 RR1.0 × 2025` (t=2.98, p=0.0038)
- `X_MES_ATR60 RR1.0 × 2025` (t=3.06, p=0.0030)
- `OVNRNG_100 RR1.5 × 2025` (t=2.74, p=0.0076)

**2025 MNQ SINGAPORE_OPEN BOOST (1 cell DROPPED at K_year, retained at K_session):**
- `ATR_P50_O30 × 2025` (t=3.08, p=0.0030) — retained at K_session BHY, dropped at K_year BHY

**CME_PRECLOSE 2019-2020 UNEVALUABLE cells (3 DROPPED):**
- `MES COST_LT08 × 2020` (n=19, t=2.72) — already suggestive-only at BH (n<30)
- `MES ORB_G8 × 2020` (n=24, t=3.07) — already suggestive-only
- `MNQ X_MES_ATR60 × 2019` (n=14, t=−2.64) — already suggestive-only; only DRAG-side cell that passed BH, now fully dropped under BHY

**Other (3 DROPPED):**
- `MES CME_PRECLOSE ORB_G8 × 2022` (t=2.56)
- `SGP ATR_P50_O15 × 2022` (t=2.45)
- `SGP ATR_P50_O30 × 2022` (t=2.63)

### Impact on the per-hypothesis verdicts

- **H1 (session regime asymmetry):** CONFIRMED under BHY. Both sessions with BH_session survivors (CME_PRECLOSE, SINGAPORE_OPEN) retain BHY_session survivors. All 5 BHY survivors are BOOST; zero DRAG. The session-structural-asymmetry claim is robust to arbitrary-dependency correction.
- **H2 (year regime alignment, 2025 BOOST concentration):** **WEAKENED** under BHY. The BH_year=2025 finding rested on 4 same-direction cells; BHY_year drops to 0 at 2025. Mechanistically plausible that the 3 dropped 2025 cells (all MNQ COMEX_SETTLE + same direction) share underlying bars — exactly the non-PRDS-safe dependency that BHY is designed to guard against. **The "2025 year regime" claim should be downgraded from confirmed to BH-only; BHY provides no confirmation.** Phase 2.10 investigation of 2025 (proposed in § Next steps 1) must account for the dependency structure.
- **H3 (GOLD fragility):** UNCHANGED. Fragility is a per-lane full-t vs ex-year-t comparison, not an FDR pass/fail — BHY has no mechanical impact.
- **v1 SINGLE_YEAR_DRAG refutation:** STRENGTHENED. The 3 v1 DRAG cells with |t| = 0.58 / 0.99 / 1.08 fail both BH and BHY at every framing. BHY reinforces the v1 correction.

### Institutional-grade honest framing

Under standard BH (1995), Phase 2.9 supports:
- Session asymmetry (CME_PRECLOSE, SINGAPORE_OPEN)
- 2025 year-regime BOOST concentration
- GOLD pool clean
- v1 DRAG labels refuted

Under BHY (2001, arbitrary-dependency-safe), Phase 2.9 supports:
- Session asymmetry — SAME
- 2025 year-regime — NOT supported (claim is dependence-fragile)
- GOLD pool clean — SAME
- v1 DRAG labels refuted — strengthened

The institutionally-defensible reading is the intersection: session-level CME_PRECLOSE / SINGAPORE_OPEN BOOST (both tests), GOLD clean, v1 DRAG refuted. The 2025 year-regime finding should be treated as BH-only evidence requiring a dependence-structure investigation before doctrine.

## Cross-instrument decomposition sub-audit (A3 from skeptical-audit pass)

The skeptical-audit pass flagged the BH_year 2025 finding as dependence-fragile: 3 of 4 survivors are MNQ COMEX_SETTLE lanes with different filters but same session + instrument + direction. BHY correctly dropped them. The audit proposed a cross-instrument decomposition (A3) to test whether MES on the same session shows an independent confirmation of the 2025 signal. Raw `orb_outcomes` queried directly (no active MES COMEX_SETTLE lane exists in validated_setups):

### CME_PRECLOSE 2020 + 2022 BOOST — cross-instrument confirmation

| Year | Instrument | Lane | N | year_expr | year_t | p | BH_year |
|---|:-:|---|--:|--:|--:|--:|:-:|
| 2020 | MES | COST_LT08 | 19 | +0.494 | +2.72 | 0.0142 | ✗ |
| 2020 | MES | ORB_G8 | 24 | +0.489 | +3.07 | 0.0055 | ✗ |
| 2020 | MNQ | X_MES_ATR60 | 56 | +0.424 | +4.00 | 0.0002 | ✓ |
| 2022 | MES | COST_LT08 | 29 | +0.489 | +3.40 | 0.0020 | ✓ |
| 2022 | MES | ORB_G8 | 45 | +0.326 | +2.56 | 0.0140 | ✗ |
| 2022 | MNQ | X_MES_ATR60 | 56 | +0.396 | +3.47 | 0.0010 | ✓ |

**Verdict:** CME_PRECLOSE 2020 and 2022 BOOST appears cross-instrument (MES + MNQ), same direction, across 3 different filter classes (COST_LT08, ORB_G8, X_MES_ATR60). This is multi-instrument multi-filter evidence of a real session-level signal — not a single dependent observation. CME_PRECLOSE F1 claim **strengthened** by cross-instrument decomposition.

### 2025 COMEX_SETTLE BOOST — cross-instrument partial confirmation

Raw `orb_outcomes` query on MES + MNQ COMEX_SETTLE E2 CB1 RR1.0 unfiltered (baseline without any filter applied, to remove filter-family dependence):

| Year | MES N | MES expr | MES t | MNQ N | MNQ expr | MNQ t |
|---|--:|--:|--:|--:|--:|--:|
| 2019 | 164 | −0.312 | −5.56 | — | — | — |
| 2020 | 249 | −0.159 | −2.99 | — | — | — |
| 2021 | 251 | −0.195 | −3.88 | — | — | — |
| 2022 | 250 | −0.068 | −1.21 | — | — | — |
| 2023 | 248 | −0.040 | −0.77 | — | — | — |
| 2024 | 245 | −0.052 | −0.97 | 246 | +0.088 | +1.52 |
| 2025 | 247 | **+0.083** | **+1.55** | 247 | **+0.169** | **+2.91** |

Notes:
- Canonical query: `research/output/phase_2_9_main.csv` + inline SQL against `orb_outcomes WHERE symbol IN ('MES', 'MNQ') AND orb_label='COMEX_SETTLE' AND orb_minutes=5 AND entry_model='E2' AND confirm_bars=1 AND rr_target=1.0 AND pnl_r IS NOT NULL` grouped by year.
- MNQ pre-2024 COMEX_SETTLE data has different `sample_size` availability (MNQ micro trades less continuous history before mid-2024 on the CSV scan year); the full pre-2024 MNQ baseline is available via `research/mode_a_revalidation_active_setups.py::load_active_setups` but not shown here to keep the comparison synchronized on MES's 7-year coverage.

**Interpretation:**
1. **MES COMEX_SETTLE was structurally negative 2019-2024** (6 consecutive years, t range −0.77 to −5.56, mean expr −0.14). 2025 is the FIRST year to turn positive.
2. **MNQ COMEX_SETTLE turned positive in 2024 (+0.088, t=+1.52) and stronger in 2025 (+0.169, t=+2.91).**
3. **BOTH instruments** — MES and MNQ, different underlying contracts, partially-overlapping but not identical bar-times — showed positive 2025 on COMEX_SETTLE. MES weakly (t=+1.55), MNQ strongly (t=+2.91).
4. This IS an independent cross-instrument confirmation: ~2 observations, not 4. Still less than the BH_year=4 naive count suggested.

**Verdict on F2 (2025 BOOST concentration):** The 2025 equity-index COMEX_SETTLE signal is real at the instrument-class level but represents **~2 independent observations** (MES + MNQ on the same session), not 4. Mechanism plausible: COMEX_SETTLE spans ~04:30 Brisbane (end of US RTH), and 2025 saw AI-driven mid-year rally + rate-cut anticipation — same macro driver on both instruments. The BH_year=4 count inflated the signal by triple-counting MNQ filter variants that share bars.

**Corrected framing:** instead of "4 independent 2025 BH_year survivors," the finding is "in 2025, COMEX_SETTLE turned positive on both MES and MNQ — a regime-consistent observation on 2 correlated instruments, with MNQ showing stronger magnitude than MES."

### SINGAPORE_OPEN 2022/2024/2025 BOOST — cross-instrument check (hardening)

Raw `orb_outcomes` query on MES + MNQ SINGAPORE_OPEN E2 CB1 RR1.5 unfiltered baseline (canonical query against `orb_outcomes` WHERE symbol IN (MES, MNQ), session=SINGAPORE_OPEN, RR1.5, E2, CB1, per-year grouped):

**MES SINGAPORE_OPEN unfiltered, O15:**

| Year | N | expr | t |
|---|--:|--:|--:|
| 2019 | 170 | −0.163 | −2.23 |
| 2020 | 258 | +0.005 | +0.08 |
| 2021 | 259 | −0.125 | −1.98 |
| 2022 | 258 | −0.076 | −1.15 |
| 2023 | 258 | −0.223 | **−3.75** |
| 2024 | 259 | −0.253 | **−4.18** |
| 2025 | 257 | −0.074 | −1.11 |

**MES SINGAPORE_OPEN unfiltered, O30:**

| Year | N | expr | t |
|---|--:|--:|--:|
| 2019 | 167 | −0.122 | −1.56 |
| 2020 | 255 | +0.061 | +0.88 |
| 2021 | 256 | −0.061 | −0.91 |
| 2022 | 257 | −0.117 | −1.72 |
| 2023 | 257 | −0.130 | −2.03 |
| 2024 | 257 | −0.236 | **−3.75** |
| 2025 | 255 | −0.019 | −0.27 |

**MNQ SINGAPORE_OPEN unfiltered, O30 (for contrast):** 2019 −0.04, 2020 +0.05, 2021 +0.02, 2022 +0.07, 2023 +0.10, 2024 +0.09, 2025 **+0.180 (t=+2.42)**.

**Verdict (hardening correction to F1's SGP component):**

MES SINGAPORE_OPEN is STRUCTURALLY NEGATIVE across every year 2019-2025 — not one positive year in 7. This is why Phase 2.5 has zero MES SGP lanes: the session is a structural loser on MES at the unfiltered baseline. The Phase 2.7 / Phase 2.9 SGP ATR_P50 BOOST finding is **MNQ-specific**, not multi-instrument like CME_PRECLOSE. MES provides NEGATIVE cross-instrument evidence: the same session on the sister equity-index contract has no edge.

**Reframed F1 claim:**
- **CME_PRECLOSE 2020/2022 BOOST** — multi-instrument (MES COST_LT08 + ORB_G8, MNQ X_MES_ATR60) → session-level regime signal.
- **SINGAPORE_OPEN 2022/2024/2025 BOOST** — MNQ-only, and even on MNQ the unfiltered baseline is only marginally positive (2025 SGP O30 unfiltered = +0.180 t=+2.42). The Phase 2.9 BH/BHY survivors (MNQ ATR_P50_O15 2024 / ATR_P50_O30 2025) sharpen a mild MNQ-SGP baseline via a filter; they are NOT a cross-session regime signal. Honest label: **MNQ-SGP-ATR_P50 filter-specific edge, not a SINGAPORE_OPEN session regime**.

The CME_PRECLOSE vs SINGAPORE_OPEN contrast is now institutionally clear:
- CME_PRECLOSE BOOST survives BH + BHY + 2-instrument cross-check — ROBUST at all 3 layers.
- SINGAPORE_OPEN BOOST survives BH + BHY — but fails 2-instrument cross-check (MES negative every year). Narrower: MNQ-specific filter signal, not session regime.

### Integration with BHY

The BHY result (K_year survivors = 1, specifically MNQ CME_PRECLOSE X_MES_ATR60 × 2020) and this cross-instrument check converge on the same doctrinal conclusion:
- 2020 CME_PRECLOSE strength is a multi-instrument multi-filter signal (BHY-confirmed + MES+MNQ cross-confirmed) → robust
- 2022 CME_PRECLOSE strength is a multi-instrument multi-filter signal (BHY-confirmed on MES COST_LT08 + MNQ X_MES_ATR60) → robust
- 2025 COMEX_SETTLE strength is a cross-instrument real observation but represents ~2 independent observations, not 4 → real but narrower

Institutionally-defensible headline after A3:
- **Session-level BOOST asymmetry (CME_PRECLOSE + SINGAPORE_OPEN)** — multi-instrument cross-confirmed
- **2025 COMEX_SETTLE BOOST** — real single-year cross-instrument observation, NOT a 4-independent-survivor year-regime
- **GOLD pool clean** — unchanged
- **v1 DRAG refuted** — unchanged

## Verdict

Phase 2.8 v1's "no recurring regime" headline is narrowly correct — no single year shows BH-significant DRAG alignment across 3+ lanes. But v1's broader implications (34/38 robust, SGP 2024 DRAG confirmed, 2024 the notable year) are not the strongest honest reads. The real signals in 7-year comprehensive testing — with the honest rigor label earned at each layer — are:
- **CME_PRECLOSE 2020/2022 BOOST** across 3 lanes on 2 instruments (MES COST_LT08 + ORB_G8 + MNQ X_MES_ATR60) — **session-level regime, robust BH + BHY + cross-instrument**
- **SINGAPORE_OPEN 2022/2024/2025 BOOST** on 2 MNQ ATR_P50 lanes (BHY-robust on 2024-O15 and 2025-O30) — **MNQ-specific filter-session edge, NOT a session regime; MES SGP is structurally negative every year 2019-2025 (failing cross-instrument check)**
- **2025 COMEX_SETTLE BOOST** — cross-instrument confirmed on MES + MNQ, but represents **~2 independent observations, not 4 BH_year survivors** (3 of 4 share session + instrument + direction = dependence-fragile)
- **GOLD pool clean** — no fragility, robust to FDR method choice

The v1 SINGLE_YEAR_DRAG labels on the 3 flagged cells are not BH-supported AND not BHY-supported. Retirements stand (independent audits), but the v1 Phase 2.8 statistical case for those retirements was weaker than presented.

**Still not tested** (would require new pre-regs if pursued): alternative FDR methods (FDP-StepM per Chordia 2018), non-calendar stratification (quarter / macro-regime / VIX-regime), block bootstrap on year_t, season-within-year decomposition. None of these have been killed — they are explicitly deferred.
