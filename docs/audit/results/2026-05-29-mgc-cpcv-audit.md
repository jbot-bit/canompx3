# MGC CPCV Audit — PASS 2 (run, methodology-correct)

**Mandate:** CPCV (AFML 2018 Sec 12.4) multi-path re-test of 6 underpowered-but-promising MGC cells. NOT a threshold rescue — same gates, better estimator. No DB writes. Read-only canonical query.

**Design (LOCKED):** N=6 temporal groups, k=2 -> C(6,4)=15 splits -> phi=5 combinatorial paths. Embargo h=ceil(0.01*T). Purge per AFML Ch 7 Sec 7.4.

**METHODOLOGICAL FINDING (load-bearing — read before the numbers):** the phi=5 full-coverage CPCV paths are MATHEMATICALLY IDENTICAL for this audit, and that is structural, not a bug. AFML Sec 12.4 builds each path as one tested instance of every group threaded across splits; each path is therefore a union of ALL N groups = the full trade series. A trade's `pnl_r` is a FIXED historical outcome that does not change with which split tested it (there is no model to refit, so purging only removes TRAIN observations and cannot alter a test group's own pnl). Hence all phi paths reconstruct the identical full series with identical ExpR/t — CPCV's path-dispersion innovation requires a per-split model refit, which a fixed-outcome backtest does not have. **The genuine multi-path object reported below is the per-split TEST-FOLD distribution (15 folds, each the k=2 held-out groups of one split), whose means DO vary across temporal windows. The full-coverage-path t (== pooled t) is used for the K4 t-gate; the fold distribution drives K1/K2 dispersion.**

**Costs:** MGC total_friction=5.74 (canonical `COST_SPECS['MGC']`); `orb_outcomes.pnl_r` already net — scored directly.

**Selection budget (honest):** K=1992 (the 6 were selected from a 1992-cell wide scan). With a ~3yr horizon this is well over the Bailey 2013 MinBTL bound, so the EXPECTED outcome is UNVERIFIED even under CPCV. This audit tests whether the multi-path estimator changes that.

**Kill criteria (LOCKED, no post-hoc change):** K1 median fold ExpR<=0 -> UNVERIFIED; K2 worst fold ExpR<-0.05 AND >40% folds neg -> WRONG; K3 PBO>0.50 -> WRONG; K4 full-path t<3.79 AND pooled power<0.50 -> UNVERIFIED. VALID only if median fold ExpR>0 AND full-path t>=3.79 AND PBO<0.50 AND worst fold>=-0.05 AND pooled power>=0.50.


## Aggregate verdict table

| # | candidate | N | folds | median fold ExpR | full-path t | worst fold | fold IQR | %folds+ | pooled power | tier | PBO | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MGC US_DATA_830 O30 RR2 long [day_of_week==1] | 100 | 15 | +0.3273 | 2.64 | +0.1314 | 0.1249 | 100% | 0.74 | DIRECTIONAL_ONLY | 0.000 | **CONDITIONAL** |
| 2 | MGC NYSE_OPEN O30 RR2 long [day_of_week==3] | 103 | 15 | +0.2000 | 2.56 | -0.2152 | 0.3246 | 93% | 0.72 | DIRECTIONAL_ONLY | 0.067 | **CONDITIONAL** |
| 3 | MGC NYSE_OPEN O30 RR1 long [day_of_week==3] | 103 | 15 | +0.1673 | 2.50 | +0.0134 | 0.1348 | 100% | 0.70 | DIRECTIONAL_ONLY | 0.000 | **CONDITIONAL** |
| 4 | MGC SINGAPORE_OPEN O30 RR2 long [day_of_week==4] | 98 | 15 | +0.2906 | 2.40 | -0.0832 | 0.2896 | 93% | 0.66 | DIRECTIONAL_ONLY | 0.067 | **CONDITIONAL** |
| 5 | MGC EUROPE_FLOW O30 RR2 long [atr_20_pct>=60] | 296 | 15 | +0.1573 | 2.27 | +0.0486 | 0.1303 | 100% | 0.62 | DIRECTIONAL_ONLY | 0.000 | **CONDITIONAL** |
| 6 | MGC LONDON_METALS O30 RR1.5 long [overnight_range_pct>=80] | 124 | 15 | +0.2059 | 2.12 | +0.0039 | 0.1360 | 100% | 0.56 | DIRECTIONAL_ONLY | 0.000 | **CONDITIONAL** |

## Per-candidate detail

### #1 — MGC US_DATA_830 O30 RR2 long [day_of_week==1]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.3273>0 and not killed, but VALID gate unmet (full-path t=2.64, power=0.74)
- N_total=100, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.000 (logit=None)
- full-path t=2.64 (N=100), pooled power=0.74 (DIRECTIONAL_ONLY); median fold t=1.52 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 32 | +0.2655 | +0.2392 | 1.35 |
  | 1 | {0,2} | 32 | +0.4453 | +0.3865 | 2.19 |
  | 2 | {0,3} | 32 | +0.3464 | +0.2837 | 1.60 |
  | 3 | {0,4} | 32 | +0.3183 | +0.2610 | 1.48 |
  | 4 | {0,5} | 36 | +0.4935 | +0.3817 | 2.29 |
  | 5 | {1,2} | 32 | +0.2584 | +0.2265 | 1.28 |
  | 6 | {1,3} | 32 | +0.1594 | +0.1335 | 0.76 |
  | 7 | {1,4} | 32 | +0.1314 | +0.1106 | 0.63 |
  | 8 | {1,5} | 36 | +0.3273 | +0.2534 | 1.52 |
  | 9 | {2,3} | 32 | +0.3393 | +0.2715 | 1.54 |
  | 10 | {2,4} | 32 | +0.3112 | +0.2493 | 1.41 |
  | 11 | {2,5} | 36 | +0.4872 | +0.3698 | 2.22 |
  | 12 | {3,4} | 32 | +0.2123 | +0.1631 | 0.92 |
  | 13 | {3,5} | 36 | +0.3992 | +0.2904 | 1.74 |
  | 14 | {4,5} | 36 | +0.3743 | +0.2723 | 1.63 |

### #2 — MGC NYSE_OPEN O30 RR2 long [day_of_week==3]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.2000>0 and not killed, but VALID gate unmet (full-path t=2.56, power=0.72)
- N_total=103, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.067 (logit=-2.6391)
- full-path t=2.56 (N=103), pooled power=0.72 (DIRECTIONAL_ONLY); median fold t=1.22 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 34 | +0.0683 | +0.0791 | 0.46 |
  | 1 | {0,2} | 34 | +0.4431 | +0.4605 | 2.69 |
  | 2 | {0,3} | 34 | +0.1270 | +0.1527 | 0.89 |
  | 3 | {0,4} | 34 | +0.5421 | +0.5681 | 3.31 |
  | 4 | {0,5} | 35 | +0.3494 | +0.3528 | 2.09 |
  | 5 | {1,2} | 34 | +0.1009 | +0.1088 | 0.63 |
  | 6 | {1,3} | 34 | -0.2152 | -0.3390 | -1.98 |
  | 7 | {1,4} | 34 | +0.2000 | +0.2091 | 1.22 |
  | 8 | {1,5} | 35 | +0.0170 | +0.0184 | 0.11 |
  | 9 | {2,3} | 34 | +0.1596 | +0.1782 | 1.04 |
  | 10 | {2,4} | 34 | +0.5747 | +0.5767 | 3.36 |
  | 11 | {2,5} | 35 | +0.3811 | +0.3678 | 2.18 |
  | 12 | {3,4} | 34 | +0.2586 | +0.2814 | 1.64 |
  | 13 | {3,5} | 35 | +0.0740 | +0.0822 | 0.49 |
  | 14 | {4,5} | 35 | +0.4773 | +0.4612 | 2.73 |

### #3 — MGC NYSE_OPEN O30 RR1 long [day_of_week==3]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.1673>0 and not killed, but VALID gate unmet (full-path t=2.50, power=0.70)
- N_total=103, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.000 (logit=None)
- full-path t=2.50 (N=103), pooled power=0.70 (DIRECTIONAL_ONLY); median fold t=1.40 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 34 | +0.1087 | +0.1473 | 0.86 |
  | 1 | {0,2} | 34 | +0.2291 | +0.3324 | 1.94 |
  | 2 | {0,3} | 34 | +0.1100 | +0.1556 | 0.91 |
  | 3 | {0,4} | 34 | +0.3400 | +0.5046 | 2.94 |
  | 4 | {0,5} | 35 | +0.1673 | +0.2366 | 1.40 |
  | 5 | {1,2} | 34 | +0.1325 | +0.1709 | 1.00 |
  | 6 | {1,3} | 34 | +0.0134 | +0.0173 | 0.10 |
  | 7 | {1,4} | 34 | +0.2435 | +0.3138 | 1.83 |
  | 8 | {1,5} | 35 | +0.0735 | +0.0940 | 0.56 |
  | 9 | {2,3} | 34 | +0.1338 | +0.1795 | 1.05 |
  | 10 | {2,4} | 34 | +0.3638 | +0.5152 | 3.00 |
  | 11 | {2,5} | 35 | +0.1904 | +0.2564 | 1.52 |
  | 12 | {3,4} | 34 | +0.2447 | +0.3281 | 1.91 |
  | 13 | {3,5} | 35 | +0.0747 | +0.0992 | 0.59 |
  | 14 | {4,5} | 35 | +0.2982 | +0.4058 | 2.40 |

### #4 — MGC SINGAPORE_OPEN O30 RR2 long [day_of_week==4]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.2906>0 and not killed, but VALID gate unmet (full-path t=2.40, power=0.66)
- N_total=98, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.067 (logit=-2.6391)
- full-path t=2.40 (N=98), pooled power=0.66 (DIRECTIONAL_ONLY); median fold t=1.34 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 32 | -0.0832 | -0.0690 | -0.39 |
  | 1 | {0,2} | 32 | +0.0547 | +0.0449 | 0.25 |
  | 2 | {0,3} | 32 | +0.2844 | +0.2165 | 1.22 |
  | 3 | {0,4} | 32 | +0.1653 | +0.1319 | 0.75 |
  | 4 | {0,5} | 34 | +0.3694 | +0.2718 | 1.58 |
  | 5 | {1,2} | 32 | +0.0421 | +0.0349 | 0.20 |
  | 6 | {1,3} | 32 | +0.2718 | +0.2086 | 1.18 |
  | 7 | {1,4} | 32 | +0.1527 | +0.1230 | 0.70 |
  | 8 | {1,5} | 34 | +0.3576 | +0.2647 | 1.54 |
  | 9 | {2,3} | 32 | +0.4098 | +0.3208 | 1.81 |
  | 10 | {2,4} | 32 | +0.2906 | +0.2368 | 1.34 |
  | 11 | {2,5} | 34 | +0.4874 | +0.3696 | 2.16 |
  | 12 | {3,4} | 32 | +0.5203 | +0.4070 | 2.30 |
  | 13 | {3,5} | 34 | +0.7036 | +0.5281 | 3.08 |
  | 14 | {4,5} | 34 | +0.5914 | +0.4501 | 2.62 |

### #5 — MGC EUROPE_FLOW O30 RR2 long [atr_20_pct>=60]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.1573>0 and not killed, but VALID gate unmet (full-path t=2.27, power=0.62)
- N_total=296, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.000 (logit=None)
- full-path t=2.27 (N=296), pooled power=0.62 (DIRECTIONAL_ONLY); median fold t=1.21 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 98 | +0.1190 | +0.0934 | 0.92 |
  | 1 | {0,2} | 98 | +0.1573 | +0.1218 | 1.21 |
  | 2 | {0,3} | 98 | +0.2696 | +0.2059 | 2.04 |
  | 3 | {0,4} | 98 | +0.1227 | +0.0930 | 0.92 |
  | 4 | {0,5} | 100 | +0.0486 | +0.0367 | 0.37 |
  | 5 | {1,2} | 98 | +0.1797 | +0.1393 | 1.38 |
  | 6 | {1,3} | 98 | +0.2920 | +0.2237 | 2.21 |
  | 7 | {1,4} | 98 | +0.1451 | +0.1101 | 1.09 |
  | 8 | {1,5} | 100 | +0.0706 | +0.0533 | 0.53 |
  | 9 | {2,3} | 98 | +0.3302 | +0.2508 | 2.48 |
  | 10 | {2,4} | 98 | +0.1834 | +0.1375 | 1.36 |
  | 11 | {2,5} | 100 | +0.1080 | +0.0806 | 0.81 |
  | 12 | {3,4} | 98 | +0.2957 | +0.2193 | 2.17 |
  | 13 | {3,5} | 100 | +0.2181 | +0.1603 | 1.60 |
  | 14 | {4,5} | 100 | +0.0742 | +0.0543 | 0.54 |

### #6 — MGC LONDON_METALS O30 RR1.5 long [overnight_range_pct>=80]

- **Verdict:** CONDITIONAL — median fold ExpR=+0.2059>0 and not killed, but VALID gate unmet (full-path t=2.12, power=0.56)
- N_total=124, splits/folds=15, phi=5 (degenerate — see finding), PBO=0.000 (logit=None)
- full-path t=2.12 (N=124), pooled power=0.56 (DIRECTIONAL_ONLY); median fold t=1.17 (per-fold ~N/3 trades, not the gate)

  CPCV test-fold distribution (15 folds, k=2 held-out groups each):

  | fold | test groups | N_test | ExpR | Sharpe | t |
  |---|---|---|---|---|---|
  | 0 | {0,1} | 40 | +0.2380 | +0.2202 | 1.39 |
  | 1 | {0,2} | 40 | +0.0445 | +0.0400 | 0.25 |
  | 2 | {0,3} | 40 | +0.2059 | +0.1843 | 1.17 |
  | 3 | {0,4} | 40 | +0.3143 | +0.2740 | 1.73 |
  | 4 | {0,5} | 44 | +0.0488 | +0.0418 | 0.28 |
  | 5 | {1,2} | 40 | +0.1887 | +0.1727 | 1.09 |
  | 6 | {1,3} | 40 | +0.3501 | +0.3260 | 2.06 |
  | 7 | {1,4} | 40 | +0.4584 | +0.4204 | 2.66 |
  | 8 | {1,5} | 44 | +0.1798 | +0.1567 | 1.04 |
  | 9 | {2,3} | 40 | +0.1565 | +0.1389 | 0.88 |
  | 10 | {2,4} | 40 | +0.2649 | +0.2282 | 1.44 |
  | 11 | {2,5} | 44 | +0.0039 | +0.0033 | 0.02 |
  | 12 | {3,4} | 40 | +0.4263 | +0.3763 | 2.38 |
  | 13 | {3,5} | 44 | +0.1506 | +0.1279 | 0.85 |
  | 14 | {4,5} | 44 | +0.2492 | +0.2062 | 1.37 |

## Summary

- VALID=0, CONDITIONAL=6, UNVERIFIED=0, WRONG=0 (of 6)
- CPCV is a confirmatory re-estimator on prior survivors; the K=1992 selection budget is carried for honest PBO/DSR accounting and is NOT re-spent here.
- No threshold was changed. No deployment claim is made. ZERO candidates reached VALID — the multi-path estimator did NOT rescue any cell (matches the pre-registered expectation under K=1992 / ~3yr horizon). CONDITIONAL means positive-and-not-overfit but unconfirmable at the locked t/power floors — NOT an edge, NOT dead.
- MGC path forward remains the SR-monitor signal-only shadow per pre_registered_criteria.md Criterion 12 (grounded pepelyshev_polunchenko_2015_cusum_sr) — NOT a calendar wait, NOT a threshold relaxation.
