---
pooled_finding: false
prereg: docs/audit/hypotheses/drafts/2026-05-17-mnq-deployed-lanes-regime-stratified-audit-v1.draft.yaml
---

# MNQ Deployed-Lane Regime-Stratified Era-Stability Audit (v1)

Per-lane regime-stability audit on the 4 currently-deployed MNQ lanes from `docs/runtime/lane_allocation.json`. H1 = chi-square + logistic GLM on per-eligible-session fire-rate across R2/R3/R4/R5. H2 = one-way ANOVA on per-trade `pnl_r_effective` across R2/R3/R4/R5. R0 and R6 are runtime-asserted excluded from hypothesis-test inputs. R6 is reported in a separate forward-monitoring section.

## K=8 multiplicity-escalation pre-amble

Primary verdict is at K=2 (family of H1 omnibus + H2 omnibus). The K=8 sensitivity table below decomposes per-lane x per-hypothesis (4 lanes x 2 hypotheses = 8 cells). Per the prereg's multiplicity_escalation_rule and per-lane_kill_escalation_rule: if ANY downstream consumer treats per-lane p-values as selection evidence (allocator-pause, capital reallocation, manual deploy/pause decision), the K=8 verdict MUST be examined and the more conservative verdict (K=8 if it differs from K=2) WINS. This pre-amble forecloses the 'we only used K=2 in the headline, so K=8 doesn't apply' loophole.

## Execution-integrity gate

- `null_pnl_non_scratch: 0` (PASS - must be 0; if >0, audit FAILS execution-integrity gate per Criterion 13).
- `scratch_policy: realized-eod` (Criterion 13 BINDING).

## dropped_regimes_per_lane (per-regime power floor)

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K`: H1_dropped=[], H2_dropped=[]
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`: H1_dropped=[], H2_dropped=[]
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`: H1_dropped=[], H2_dropped=[]
- `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25`: H1_dropped=[], H2_dropped=[]

## Per-lane verdicts (K=2 primary)

| lane | H1_p | H2_p | R5_ExpR | R5_N | verdict | rationale |
|---|---|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` | 0.00000 | 0.06319 | 0.2494 | 247 | **PARK** | H1 omnibus p=0.0000 < 0.01 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 0.75861 | 0.42594 | -0.0019 | 257 | **PARK** | R5 ExpR=-0.0019 <= 0 at N=257 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 0.23366 | 0.55235 | 0.1282 | 257 | **CONTINUE** | H1 p=0.2337, H2 p=0.5523, R5 ExpR=0.1282 (N=257). |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` | 0.00608 | 0.39092 | 0.0792 | 257 | **PARK** | H1 omnibus p=0.0061 < 0.01 |

## H1: fire-rate stability (chi-square 4x2 per lane)

### `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` -- H1

Regimes used: ['R2', 'R3', 'R4', 'R5']
Contingency table (regime x [fired, not_fired]): [[445, 55], [250, 0], [495, 2], [247, 0]]
Expected cells: [[480.9236947791165, 19.076305220883533], [240.46184738955824, 9.538152610441767], [478.0381526104418, 18.961847389558233], [237.57630522088354, 9.423694779116467]]
chi2 = 105.8220, dof = 3, raw_p = 0.00000
Logistic GLM status: FIT; coefficients: {'Intercept': 2.0907410979346026, 'C(regime)[T.R3]': 20.22142975876864, 'C(regime)[T.R4]': 3.4206694840741623, 'C(regime)[T.R5]': 26.019027262899666}

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` -- H1

Regimes used: ['R2', 'R3', 'R4', 'R5']
Contingency table (regime x [fired, not_fired]): [[274, 242], [138, 120], [259, 257], [135, 122]]
Expected cells: [[268.84033613445376, 247.1596638655462], [134.42016806722688, 123.5798319327731], [268.84033613445376, 247.1596638655462], [133.89915966386553, 123.10084033613445]]
chi2 = 1.1766, dof = 3, raw_p = 0.75861
Logistic GLM status: FIT; coefficients: {'Intercept': 0.1241903802313841, 'C(regime)[T.R3]': 0.015571562143774587, 'C(regime)[T.R4]': -0.11643840342706631, 'C(regime)[T.R5]': -0.022936646526211375}

### `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` -- H1

Regimes used: ['R2', 'R3', 'R4', 'R5']
Contingency table (regime x [fired, not_fired]): [[509, 8], [256, 2], [510, 6], [257, 0]]
Expected cells: [[511.656330749354, 5.343669250645995], [255.33333333333334, 2.6666666666666665], [510.6666666666667, 5.333333333333333], [254.343669250646, 2.656330749354005]]
chi2 = 4.2709, dof = 3, raw_p = 0.23366
Small-cell fallback fired (CHI2_YATES); fallback_p = 0.23366
Logistic GLM status: FIT; coefficients: {'Intercept': 4.153006386277246, 'C(regime)[T.R3]': 0.6990238776423742, 'C(regime)[T.R4]': 0.28964487021306345, 'C(regime)[T.R5]': 21.333995235549096}

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` -- H1

Regimes used: ['R2', 'R3', 'R4', 'R5']
Contingency table (regime x [fired, not_fired]): [[505, 11], [258, 0], [503, 13], [257, 0]]
Expected cells: [[507.994828700711, 8.005171299288946], [253.9974143503555, 4.002585649644473], [507.994828700711, 8.005171299288946], [253.01292824822238, 3.987071751777634]]
chi2 = 12.4193, dof = 3, raw_p = 0.00608
Small-cell fallback fired (CHI2_YATES); fallback_p = 0.00608
Logistic GLM status: FIT; coefficients: {'Intercept': 3.826663201590391, 'C(regime)[T.R3]': 18.30421978004199, 'C(regime)[T.R4]': -0.1710223889521852, 'C(regime)[T.R5]': 14.959140420640892}

## H2: ExpR stability (one-way ANOVA per lane)

### `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` -- H2

Regimes used: ['R2', 'R3', 'R4', 'R5']
N per regime: {'R2': 500, 'R3': 250, 'R4': 497, 'R5': 247}
Mean ExpR per regime: {'R2': 0.0551, 'R3': 0.002, 'R4': 0.1365, 'R5': 0.2494}
F = 2.4353, raw_p = 0.06319

#### H2 Sensitivity Diagnostics

DIAGNOSTIC ONLY - primary verdict is prereg ANOVA per Criterion 1 (pre-registered hypothesis). Sensitivity tests do NOT flip verdict; they flag prereg misspecification for the NEXT prereg cycle.
- Welch's ANOVA p = 0.07080
- Kruskal-Wallis p = 0.00018
- Levene p = 0.28286, shapiro_flag = True

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` -- H2

Regimes used: ['R2', 'R3', 'R4', 'R5']
N per regime: {'R2': 516, 'R3': 258, 'R4': 516, 'R5': 257}
Mean ExpR per regime: {'R2': 0.1031, 'R3': 0.1106, 'R4': 0.1024, 'R5': -0.0019}
F = 0.9290, raw_p = 0.42594

#### H2 Sensitivity Diagnostics

DIAGNOSTIC ONLY - primary verdict is prereg ANOVA per Criterion 1 (pre-registered hypothesis). Sensitivity tests do NOT flip verdict; they flag prereg misspecification for the NEXT prereg cycle.
- Welch's ANOVA p = 0.43578
- Kruskal-Wallis p = 0.00125
- Levene p = 0.51691, shapiro_flag = True

### `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` -- H2

Regimes used: ['R2', 'R3', 'R4', 'R5']
N per regime: {'R2': 517, 'R3': 258, 'R4': 516, 'R5': 257}
Mean ExpR per regime: {'R2': 0.0377, 'R3': 0.1066, 'R4': 0.1025, 'R5': 0.1282}
F = 0.6995, raw_p = 0.55235

#### H2 Sensitivity Diagnostics

DIAGNOSTIC ONLY - primary verdict is prereg ANOVA per Criterion 1 (pre-registered hypothesis). Sensitivity tests do NOT flip verdict; they flag prereg misspecification for the NEXT prereg cycle.
- Welch's ANOVA p = 0.55265
- Kruskal-Wallis p = 0.00000
- Levene p = 0.71722, shapiro_flag = True

### `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` -- H2

Regimes used: ['R2', 'R3', 'R4', 'R5']
N per regime: {'R2': 516, 'R3': 258, 'R4': 516, 'R5': 257}
Mean ExpR per regime: {'R2': 0.0422, 'R3': 0.1482, 'R4': 0.1242, 'R5': 0.0792}
F = 1.0022, raw_p = 0.39092

#### H2 Sensitivity Diagnostics

DIAGNOSTIC ONLY - primary verdict is prereg ANOVA per Criterion 1 (pre-registered hypothesis). Sensitivity tests do NOT flip verdict; they flag prereg misspecification for the NEXT prereg cycle.
- Welch's ANOVA p = 0.39044
- Kruskal-Wallis p = 0.00000
- Levene p = 0.54223, shapiro_flag = True

## K=8 sensitivity table (per-lane x per-hypothesis Bonferroni)

| lane | hypothesis | raw_p | bonferroni_k8_p | k8_verdict_tier |
|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` | H1 | 0.00000 | 0.00000 | FAIL |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` | H2 | 0.06319 | 0.50553 | PASS |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | H1 | 0.75861 | 1.00000 | PASS |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | H2 | 0.42594 | 1.00000 | PASS |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | H1 | 0.23366 | 1.00000 | PASS |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | H2 | 0.55235 | 1.00000 | PASS |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` | H1 | 0.00608 | 0.04861 | WATCH |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` | H2 | 0.39092 | 1.00000 | PASS |

## Forward monitoring (R6 sacred holdout)

NOT SELECTION EVIDENCE per pre_registered_criteria.md Amendment 2.7 sacred holdout

| lane | R6_N | R6_ExpR | R6_Sharpe | R6_fire_rate |
|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` | 77 | 0.0291 | 0.0250 | 1.0000 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 72 | 0.2052 | 0.2219 | 0.6528 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 80 | 0.0817 | 0.0830 | 1.0000 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` | 79 | 0.0338 | 0.0350 | 1.0000 |

Forbidden uses of this section: ranking lanes; killing or pausing a lane; rescuing a lane that failed IS R5 kill clause; re-running the audit with different thresholds to rescue an R6 outcome.

## R0 (pre-2020 micro-launch) -- INFORMATIONAL_EXCLUDED

R0 is excluded from both H1 and H2 test inputs at runtime via `HoldoutContaminationError`. The window is reported here for completeness; it does NOT feed any verdict.

| lane | R0_N_trades | R0_ExpR |
|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K` | 164 | -0.2585 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 170 | 0.1411 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 171 | 0.0041 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25` | 171 | 0.0229 |

