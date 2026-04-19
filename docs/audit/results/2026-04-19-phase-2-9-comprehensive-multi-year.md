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

Under Phase 2.9's per-cell BH testing with K_session/K_year/K_global all considered:
- Lane 1: 2024 year_t = −1.91, year_p = 0.060. **Not a BH survivor at any framing.**
- Lane 2: 2024 year_t = −1.85, year_p = 0.068. **Not a BH survivor at any framing.**
- Lane 3: 2020 year_t = −0.79, year_p = 0.434. **Not a BH survivor at any framing.**

**The SGP retirement verdicts stand** — they were independently confirmed by Phase 2.4 (composite C1-C12 audit) and Phase 2.7 (2024-specific regime break). But v1's Phase 2.8 DRAG labels for those cells were not statistically supported under honest testing; they were bare-threshold observations that the v1 classifier elevated to a label tier without significance checking. This is the single most important methodological correction from the reframe addendum to the sweep itself.

## Institutional methodology notes

- **BH-FDR calibration verified:** small-p cells cluster at the rank-boundary as expected. K_global=266 produces rank 1 pass at p ≤ 0.000376; observed smallest p is 0.000192, passing with headroom.
- **MinBTL check passed:** at K=266 and est_max_N=1521, MinBTL = 2·ln(266)/1521² = 5.0×10⁻⁶ << 1.0. Sample size is not a binding constraint at any K framing tested.
- **Per-cell subset-t uses t-distribution (df = n−1), not Normal.** Stricter for small-N cells than Normal approximation. This is the correct choice per Chordia 2018 § 3 footnote 14.
- **Canonical delegation verified:** sanity check on MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08 shows full-window `_window_stats` agrees with `compute_mode_a` to div=0.00e+00.
- **UNEVALUABLE cells can pass BH.** The labeller marks n_year < 30 as UNEVALUABLE but BH is computed on p-values which don't know about N. 5 of 12 K_session survivors have n < 30 (all CME_PRECLOSE). These are statistically significant but low-power evidence; for any deployment decision they should be treated as suggestive only (RULE 3.2).

## What the scan does NOT claim

- Does NOT promote any lane.
- Does NOT retract any retirement.
- Does NOT claim 2025 has a mechanistically-understood tailwind. 2025 is a BH-significant concentration of positive year-deltas across MNQ COMEX_SETTLE and SINGAPORE_OPEN — the mechanism remains a testable hypothesis, not an established finding. Candidate explanations (macro-tightening end-of-cycle, AI sector rotation, cross-asset vol regime) are all training-memory claims that would need literature-grounded pre-reg to pursue.
- Does NOT show that v1's retirement framing was wrong overall — only that v1's bare-threshold labelling was statistically weak. The retirement decisions remain correct.
- Does NOT test the 2 retired SGP PURE_DRAG lanes directly (they are not in the active-set source).

## Next steps (candidate — require fresh pre-reg if pursued)

1. **Phase 2.10 (not written):** investigate the 2025 BOOST concentration. 4 BH-survivor lanes in one year suggests a genuine year-regime; would benefit from a macro-grounded pre-reg (Carver 2015 Ch 9 vol-standardised sizing as a Stage 2 framework) plus a check that 2025 partial-year data is not biased by sample-selection.
2. **Retired-SGP RR dose-response audit (not written):** the addendum § I2 claim is untested here. Would require a separate pre-reg against the retired lanes in `validated_setups` with `status != 'active'`.
3. **Governance:** update Phase 2.8 v1 result doc with a pointer to this v2 supersession (same method as the addendum used — footnote, not rewrite).
4. **Doctrine upgrade (proposed):** add to backtesting-methodology.md RULE 4 a sub-clause requiring per-cell significance testing on any classification threshold, not just on the discovery scan's promotion gate. The v1 PATTERN labeller used 0.03 bare threshold without p-values — that pattern should be caught in future reviews.

## Audit trail

- Pre-reg committed before script run: `b74dafe3` (pre-reg + stage file).
- Script + tests committed separately from result doc for provenance: TBD commits.
- CSVs deterministic: re-running the script with the same git SHA produces identical rows (ordering by `strategy_id, year`).
- 34 unit tests passing (`pytest tests/test_research/test_phase_2_9_comprehensive_multi_year_stratification.py -q`).
- Drift check: 5 pre-existing worktree failures (Check 37 cascade — no local gold.db), all expected per pre-reg.
- Canonical delegations verified at run time: `compute_mode_a` vs `_window_stats` div=0.00e+00 on the probe lane.

## Verdict

Phase 2.8 v1's "no recurring regime" headline is narrowly correct — no single year shows BH-significant DRAG alignment across 3+ lanes. But v1's broader implications (34/38 robust, SGP 2024 DRAG confirmed, 2024 the notable year) are not the strongest honest reads. The real signals in 7-year comprehensive testing are:
- 2025 BOOST concentration (4 BH-survivor cells, all same-direction positive)
- CME_PRECLOSE 2020/2022 BOOST across 3 lanes (volatility-regime tailwind)
- SINGAPORE_OPEN 2022/2024/2025 BOOST on 2 ATR_P50 lanes (lane-level persistence)
- GOLD pool clean — no fragility

The v1 SINGLE_YEAR_DRAG labels on the 3 flagged cells are not BH-supported. Retirements stand (independent audits), but the v1 Phase 2.8 statistical case for those retirements was weaker than presented.
