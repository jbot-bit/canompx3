# MES low-ATR (~ATR_P70) inverse-filter Pathway A family — 5 sessions × 2 directions × RR=1.5

**Generated:** 2026-04-19T00:02:37+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mes-low-atr-inverse-filter-v1.yaml` (LOCKED, commit_sha=b9f38e33)
**Script:** `research/mes_low_atr_inverse_filter_v1_scan.py`
**IS:** `trading_day < 2026-01-01`

## ERRATUM — pre-reg arithmetic error (K=20 → K=10 actual)

The LOCKED pre-reg declares `k_family: 20` with the structure "5 sessions × 2 directions × RR=1.5 × (~ATR_P70) = 20 cells". The correct multiplication is `5 × 2 × 1 × 1 = 10`. The "× 2" factor in the pre-reg's narrative was an arithmetic error — this family has 10 cells, not 20.

**Impact on verdict:** NONE. All 10 cells fail the positive-ExpR gate (`ExpR_on_IS > 0`) — every cell has strongly NEGATIVE ExpR (t from −2.04 to −7.33). The K1 kill criterion (0 cells pass all 6 gate clauses) fires at BOTH K=10 and K=20 framings.

**Impact on q-values:** q-values reported below are computed at the actual K=10. At the declared K=20, q-values approximately double. The strongest cell (H07 LONDON_METALS long) moves from q=0.042 to q≈0.084 — borderline fail at K=20. All other cells remain q < 0.001 even at K=20. Substantive verdict unchanged.

**Action:** pre-reg is NOT being modified (the LOCKED commit SHA is preserved). This erratum block is appended to the result doc per the spirit of `backtesting-methodology.md` RULE 11 (audit trail: supersede via new docs, never delete).

## Substantive finding — MES unfiltered baseline is NEGATIVE across ATR distribution

The hypothesis from `2026-04-19-correction-cycle-multi-angle-synthesis.md` § Angle 1 was: "since 10/10 MES ATR_P70 cells (top-30% ATR) show negative ExpR, the inverse (~ATR_P70 = bottom-70% ATR) should be the profitable regime." This scan REFUTES that hypothesis.

**Observed:** every cell's unfiltered baseline (`ExpR_b` column) is ALREADY negative, ranging from −0.084 (LONDON_METALS long) to −0.232 (SINGAPORE_OPEN short). The ~ATR_P70 subset's ExpR is approximately equal to the unfiltered baseline — the filter barely moves expectancy. So neither top-30% nor bottom-70% ATR is a profitable MES regime.

**Correct interpretation:** MES unfiltered E2 breakouts are structurally unprofitable across the entire ATR distribution. The ATR_P70 pattern observed in the K=40 scan (top-30% ATR cells all negative) is NOT a diagnostic for a low-ATR edge — it's just a reflection of the NEGATIVE UNFILTERED BASELINE. Every subset of MES unfiltered E2 is negative, on average, because the unfiltered mean is negative.

**What this means for the original K=40 finding:**
- The K=40 ATR_P70 cells being 10/10 negative does NOT validate "high ATR is bad for MES". It reflects the instrument-wide unfiltered negative baseline.
- The K=40's single validated MES lane (`MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`) remains the exception — ORB_G8 (top octile orb size) on CME_PRECLOSE long is the ONE known profitable MES subset.
- The correction-cycle-synthesis Angle 1 framing was a PREMATURE hypothesis. Now falsified.

**Canonical framing going forward:** MES requires an AFFIRMATIVE edge signal (size/vol/cross-asset confluence), not an inverse-ATR SKIP. Future MES pre-regs should test positive-selection hypotheses, not negation of already-negative patterns.

## Summary: 10 cells | CONTINUE: 0 | KILL: 10

**K2 baseline sanity smoke-test:** PASS

| Cell | Session | Dir | RR | Filter | N_base | N_on | Fire% | ExpR_b | ExpR_on | Δ_IS | t | raw_p | boot_p | q | yrs+ |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H01_TOK_LOWATR_L_RR15 | TOKYO_OPEN | long | 1.5 | ATR_P70 | 845 | 582 | 0.689 | -0.1302 | -0.1776 | -0.0474 | -4.435 | 0.0000 | 0.0001 | 0.0000 | 0 |
| H02_TOK_LOWATR_S_RR15 | TOKYO_OPEN | short | 1.5 | ATR_P70 | 876 | 587 | 0.670 | -0.1361 | -0.1731 | -0.0370 | -4.324 | 0.0000 | 0.0001 | 0.0000 | 0 |
| H03_SGP_LOWATR_L_RR15 | SINGAPORE_OPEN | long | 1.5 | ATR_P70 | 894 | 606 | 0.678 | -0.1922 | -0.2591 | -0.0669 | -7.048 | 0.0000 | 0.0001 | 0.0000 | 0 |
| H04_SGP_LOWATR_S_RR15 | SINGAPORE_OPEN | short | 1.5 | ATR_P70 | 827 | 563 | 0.681 | -0.2322 | -0.2793 | -0.0471 | -7.335 | 0.0000 | 0.0001 | 0.0000 | 0 |
| H05_EUR_LOWATR_L_RR15 | EUROPE_FLOW | long | 1.5 | ATR_P70 | 850 | 570 | 0.671 | -0.1573 | -0.1821 | -0.0248 | -4.506 | 0.0000 | 0.0001 | 0.0000 | 1 |
| H06_EUR_LOWATR_S_RR15 | EUROPE_FLOW | short | 1.5 | ATR_P70 | 868 | 598 | 0.689 | -0.2233 | -0.2384 | -0.0151 | -6.099 | 0.0000 | 0.0001 | 0.0000 | 1 |
| H07_LDM_LOWATR_L_RR15 | LONDON_METALS | long | 1.5 | ATR_P70 | 875 | 595 | 0.680 | -0.0838 | -0.0851 | -0.0013 | -2.038 | 0.0420 | 0.0403 | 0.0420 | 1 |
| H08_LDM_LOWATR_S_RR15 | LONDON_METALS | short | 1.5 | ATR_P70 | 842 | 574 | 0.682 | -0.1446 | -0.1799 | -0.0353 | -4.292 | 0.0000 | 0.0002 | 0.0000 | 0 |
| H09_CMX_LOWATR_L_RR15 | COMEX_SETTLE | long | 1.5 | ATR_P70 | 887 | 596 | 0.672 | -0.0993 | -0.1309 | -0.0316 | -3.176 | 0.0016 | 0.0019 | 0.0017 | 2 |
| H10_CMX_LOWATR_S_RR15 | COMEX_SETTLE | short | 1.5 | ATR_P70 | 755 | 518 | 0.686 | -0.1456 | -0.1620 | -0.0164 | -3.631 | 0.0003 | 0.0004 | 0.0004 | 1 |

## Gate breakdown

| Cell | bh_pass | abs_t_ge_3 | N_on_ge_100 | years_pos_ge_4 | boot_p_lt_0.10 | ExpR_gt_0 | not_taut | not_ext_fire | not_arith | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H01_TOK_LOWATR_L_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H02_TOK_LOWATR_S_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H03_SGP_LOWATR_L_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H04_SGP_LOWATR_S_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H05_EUR_LOWATR_L_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H06_EUR_LOWATR_S_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H07_LDM_LOWATR_L_RR15 | Y | N | Y | N | Y | N | Y | Y | Y | KILL |
| H08_LDM_LOWATR_S_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H09_CMX_LOWATR_L_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |
| H10_CMX_LOWATR_S_RR15 | Y | Y | Y | N | Y | N | Y | Y | Y | KILL |

## T0 / flags

| Cell | corr_orbsize (expected ~1 for ORB_G) | corr_atr | tautology | extreme_fire | arith_only |
|---|---:|---:|---|---|---|
| H01_TOK_LOWATR_L_RR15 | -0.258 | -0.581 | N | N | N |
| H02_TOK_LOWATR_S_RR15 | -0.278 | -0.555 | N | N | N |
| H03_SGP_LOWATR_L_RR15 | -0.246 | -0.554 | N | N | N |
| H04_SGP_LOWATR_S_RR15 | -0.323 | -0.583 | N | N | N |
| H05_EUR_LOWATR_L_RR15 | -0.271 | -0.584 | N | N | N |
| H06_EUR_LOWATR_S_RR15 | -0.297 | -0.549 | N | N | N |
| H07_LDM_LOWATR_L_RR15 | -0.348 | -0.575 | N | N | N |
| H08_LDM_LOWATR_S_RR15 | -0.215 | -0.560 | N | N | N |
| H09_CMX_LOWATR_L_RR15 | -0.267 | -0.572 | N | N | N |
| H10_CMX_LOWATR_S_RR15 | -0.282 | -0.576 | N | N | N |

## OOS descriptive

| Cell | N_OOS_on | ExpR_OOS | Δ_OOS | dir_match |
|---|---:|---:|---:|---|
| H01_TOK_LOWATR_L_RR15 | 14 | 0.3709 | 0.2262 | N |
| H02_TOK_LOWATR_S_RR15 | 15 | -0.4342 | -0.2728 | Y |
| H03_SGP_LOWATR_L_RR15 | 16 | 0.0221 | -0.0231 | Y |
| H04_SGP_LOWATR_S_RR15 | 13 | -0.5202 | -0.3864 | Y |
| H05_EUR_LOWATR_L_RR15 | 15 | -0.2927 | -0.1703 | Y |
| H06_EUR_LOWATR_S_RR15 | 14 | -0.2789 | -0.0246 | Y |
| H07_LDM_LOWATR_L_RR15 | 15 | -0.0072 | 0.0985 | N |
| H08_LDM_LOWATR_S_RR15 | 14 | 0.3883 | 0.3900 | N |
| H09_CMX_LOWATR_L_RR15 | 18 | -0.0381 | 0.1649 | N |
| H10_CMX_LOWATR_S_RR15 | 10 | -0.1259 | -0.1267 | Y |

## Per-year IS

| Cell | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H01_TOK_LOWATR_L_RR15 | --0.263(N=79) | --0.167(N=78) | --0.043(N=110) | --0.240(N=70) | --0.181(N=121) | --0.281(N=55) | --0.155(N=69) |
| H02_TOK_LOWATR_S_RR15 | --0.224(N=67) | --0.073(N=71) | --0.055(N=99) | --0.081(N=69) | --0.251(N=136) | --0.316(N=74) | --0.182(N=71) |
| H03_SGP_LOWATR_L_RR15 | --0.322(N=79) | --0.192(N=85) | --0.294(N=95) | --0.172(N=72) | --0.237(N=132) | --0.497(N=72) | --0.110(N=71) |
| H04_SGP_LOWATR_S_RR15 | --0.370(N=67) | --0.086(N=64) | --0.192(N=114) | --0.249(N=67) | --0.364(N=125) | --0.389(N=57) | --0.300(N=69) |
| H05_EUR_LOWATR_L_RR15 | --0.316(N=68) | +0.107(N=71) | --0.246(N=99) | --0.201(N=65) | --0.125(N=124) | --0.311(N=64) | --0.217(N=79) |
| H06_EUR_LOWATR_S_RR15 | --0.512(N=78) | --0.061(N=77) | --0.270(N=110) | --0.196(N=74) | --0.288(N=133) | --0.307(N=65) | +0.075(N=61) |
| H07_LDM_LOWATR_L_RR15 | --0.174(N=74) | +0.323(N=70) | --0.056(N=109) | --0.154(N=72) | --0.153(N=140) | --0.320(N=55) | --0.055(N=75) |
| H08_LDM_LOWATR_S_RR15 | --0.285(N=72) | --0.368(N=79) | --0.033(N=100) | --0.213(N=67) | --0.148(N=117) | --0.171(N=74) | --0.096(N=65) |
| H09_CMX_LOWATR_L_RR15 | --0.335(N=83) | --0.203(N=79) | --0.352(N=110) | --0.092(N=62) | +0.084(N=121) | +0.006(N=69) | --0.005(N=72) |
| H10_CMX_LOWATR_S_RR15 | --0.624(N=55) | --0.184(N=64) | --0.161(N=90) | --0.042(N=72) | --0.186(N=126) | --0.114(N=52) | +0.155(N=59) |

## Decision

**Verdict: KILL per K1.**

## Reproduction
```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mes_broader_mode_a_rediscovery_v1_scan.py
```
