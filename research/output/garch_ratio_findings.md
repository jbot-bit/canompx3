# GARCH/ATR Ratio Mispricing Filter Validation
**Date:** 2026-02-21
**Script:** research/research_garch_ratio.py

======================================================================
PART 0: DIRECT CLAIM CHECK -- MNQ 0900, garch_atr_ratio < 0.58
======================================================================

  MNQ 0900 E0 RR1.0:
    GARCH<0.58: N= 108, avgR=+0.213, WR=64.8%
    GARCH>=0.58: N= 430, avgR=-0.051, WR=52.1%
    Delta: +0.264R, WR delta: +12.7pp, p=0.0074, PRELIMINARY

  MNQ 0900 E0 RR1.5:
    GARCH<0.58: N= 108, avgR=+0.042, WR=44.4%
    GARCH>=0.58: N= 430, avgR=-0.054, WR=41.4%
    Delta: +0.096R, WR delta: +3.0pp, p=0.4434, PRELIMINARY

  MNQ 0900 E0 RR2.0:
    GARCH<0.58: N= 108, avgR=+0.174, WR=41.7%
    GARCH>=0.58: N= 430, avgR=-0.032, WR=35.3%
    Delta: +0.206R, WR delta: +6.3pp, p=0.1681, PRELIMINARY

  MNQ 0900 E1 RR1.0:
    GARCH<0.58: N= 278, avgR=-0.071, WR=48.9%
    GARCH>=0.58: N=1092, avgR=-0.173, WR=44.4%
    Delta: +0.102R, WR delta: +4.5pp, p=0.1097, CORE

  MNQ 0900 E1 RR1.5:
    GARCH<0.58: N= 278, avgR=-0.043, WR=40.3%
    GARCH>=0.58: N=1092, avgR=-0.132, WR=37.3%
    Delta: +0.089R, WR delta: +3.0pp, p=0.2518, CORE

  MNQ 0900 E1 RR2.0:
    GARCH<0.58: N= 278, avgR=-0.034, WR=33.8%
    GARCH>=0.58: N=1092, avgR=-0.130, WR=31.2%
    Delta: +0.096R, WR delta: +2.6pp, p=0.2884, CORE

======================================================================
PART 1: GARCH<0.58 vs >=0.58 (all instruments x sessions x entry models)
======================================================================

  MGC_0900_E0_RR2.0:
    GARCH<0.58: N= 678, avgR=-0.282, WR=36.3%
    GARCH>=0.58: N=2918, avgR=-0.442, WR=27.9%
    Delta: +0.160R, WR_delta: +8.4pp, p=0.0001, CORE

  MGC_0900_E1_RR2.0:
    GARCH<0.58: N=1977, avgR=-0.330, WR=32.6%
    GARCH>=0.58: N=9121, avgR=-0.455, WR=28.2%
    Delta: +0.125R, WR_delta: +4.4pp, p=0.0000, CORE

  MGC_1000_E0_RR2.0:
    GARCH<0.58: N= 687, avgR=-0.254, WR=38.9%
    GARCH>=0.58: N=3152, avgR=-0.334, WR=36.2%
    Delta: +0.080R, WR_delta: +2.7pp, p=0.0430, CORE

  MGC_1000_E1_RR2.0:
    GARCH<0.58: N=2081, avgR=-0.298, WR=33.8%
    GARCH>=0.58: N=9477, avgR=-0.359, WR=33.1%
    Delta: +0.060R, WR_delta: +0.7pp, p=0.0116, CORE

  MGC_1800_E0_RR2.0:
    GARCH<0.58: N= 734, avgR=-0.198, WR=40.1%
    GARCH>=0.58: N=3333, avgR=-0.247, WR=39.9%
    Delta: +0.049R, WR_delta: +0.2pp, p=0.2276, CORE

  MGC_1800_E1_RR2.0:
    GARCH<0.58: N=2051, avgR=-0.248, WR=34.8%
    GARCH>=0.58: N=9350, avgR=-0.309, WR=33.4%
    Delta: +0.061R, WR_delta: +1.4pp, p=0.0167, CORE

  MES_0900_E0_RR2.0:
    GARCH<0.58: N= 511, avgR=-0.110, WR=39.3%
    GARCH>=0.58: N=2217, avgR=-0.150, WR=38.8%
    Delta: +0.040R, WR_delta: +0.5pp, p=0.4688, CORE

  MES_0900_E1_RR2.0:
    GARCH<0.58: N=1378, avgR=-0.199, WR=32.6%
    GARCH>=0.58: N=6064, avgR=-0.237, WR=31.9%
    Delta: +0.038R, WR_delta: +0.7pp, p=0.2769, CORE

  MES_1000_E0_RR2.0:
    GARCH<0.58: N= 537, avgR=-0.034, WR=40.0%
    GARCH>=0.58: N=2284, avgR=-0.029, WR=41.7%
    Delta: -0.005R, WR_delta: -1.6pp, p=0.9322, CORE

  MES_1000_E1_RR2.0:
    GARCH<0.58: N=1473, avgR=-0.169, WR=32.6%
    GARCH>=0.58: N=6397, avgR=-0.151, WR=34.1%
    Delta: -0.018R, WR_delta: -1.5pp, p=0.5966, CORE

  MNQ_0900_E0_RR2.0:
    GARCH<0.58: N= 108, avgR=+0.174, WR=41.7%
    GARCH>=0.58: N= 430, avgR=-0.032, WR=35.3%
    Delta: +0.206R, WR_delta: +6.3pp, p=0.1681, PRELIMINARY

  MNQ_0900_E1_RR2.0:
    GARCH<0.58: N= 278, avgR=-0.034, WR=33.8%
    GARCH>=0.58: N=1092, avgR=-0.130, WR=31.2%
    Delta: +0.096R, WR_delta: +2.6pp, p=0.2884, CORE

  MNQ_1000_E0_RR2.0:
    GARCH<0.58: N= 111, avgR=-0.165, WR=30.6%
    GARCH>=0.58: N= 458, avgR=+0.138, WR=41.0%
    Delta: -0.303R, WR_delta: -10.4pp, p=0.0270, PRELIMINARY

  MNQ_1000_E1_RR2.0:
    GARCH<0.58: N= 296, avgR=-0.265, WR=26.0%
    GARCH>=0.58: N=1158, avgR=+0.048, WR=37.0%
    Delta: -0.313R, WR_delta: -11.0pp, p=0.0002, CORE

======================================================================
PART 2: THRESHOLD SENSITIVITY (MNQ 0900)
======================================================================

  MNQ 0900 E0 RR2.0:
    <0.50: N=  24, avgR=+0.037, WR=37.5% | >=0.50: N= 514, avgR=+0.008, WR=36.6% | delta=+0.029, p=0.9202
    <0.55: N=  76, avgR=+0.449, WR=51.3% | >=0.55: N= 462, avgR=-0.063, WR=34.2% | delta=+0.511, p=0.0041
    <0.58: N= 108, avgR=+0.174, WR=41.7% | >=0.58: N= 430, avgR=-0.032, WR=35.3% | delta=+0.206, p=0.1681
    <0.60: N= 120, avgR=+0.203, WR=42.5% | >=0.60: N= 418, avgR=-0.046, WR=34.9% | delta=+0.249, p=0.0848
    <0.65: N= 182, avgR=+0.133, WR=40.1% | >=0.65: N= 356, avgR=-0.053, WR=34.8% | delta=+0.186, p=0.1347
    <0.70: N= 254, avgR=+0.020, WR=36.2% | >=0.70: N= 284, avgR=+0.000, WR=37.0% | delta=+0.020, p=0.8643
    <0.75: N= 315, avgR=+0.016, WR=36.2% | >=0.75: N= 223, avgR=+0.001, WR=37.2% | delta=+0.015, p=0.9001
    <0.80: N= 387, avgR=+0.019, WR=36.7% | >=0.80: N= 151, avgR=-0.015, WR=36.4% | delta=+0.035, p=0.7839

  MNQ 0900 E1 RR2.0:
    <0.45: N=  15, avgR=-1.000, WR=0.0% | >=0.45: N=1355, avgR=-0.101, WR=32.1% | delta=-0.899, p=0.0000
    <0.50: N=  75, avgR=+0.137, WR=40.0% | >=0.50: N=1295, avgR=-0.125, WR=31.3% | delta=+0.262, p=0.1182
    <0.55: N= 196, avgR=+0.138, WR=39.8% | >=0.55: N=1174, avgR=-0.152, WR=30.4% | delta=+0.290, p=0.0073
    <0.58: N= 278, avgR=-0.034, WR=33.8% | >=0.58: N=1092, avgR=-0.130, WR=31.2% | delta=+0.096, p=0.2884
    <0.60: N= 313, avgR=-0.077, WR=32.3% | >=0.60: N=1057, avgR=-0.121, WR=31.6% | delta=+0.044, p=0.6066
    <0.65: N= 483, avgR=-0.020, WR=34.4% | >=0.65: N= 887, avgR=-0.160, WR=30.3% | delta=+0.140, p=0.0633
    <0.70: N= 684, avgR=-0.114, WR=31.1% | >=0.70: N= 686, avgR=-0.108, WR=32.4% | delta=-0.006, p=0.9314
    <0.75: N= 854, avgR=-0.133, WR=30.6% | >=0.75: N= 516, avgR=-0.073, WR=33.7% | delta=-0.060, p=0.4120
    <0.80: N=1009, avgR=-0.114, WR=31.4% | >=0.80: N= 361, avgR=-0.101, WR=32.7% | delta=-0.013, p=0.8696

======================================================================
PART 3: YEAR-BY-YEAR (MNQ 0900, GARCH<0.58)
======================================================================

  MNQ 0900 E0 RR1.0 GARCH<0.58 (N=108):
    2025: N= 108, avgR=+0.213, WR=64.8% [+]
    Years positive: 1/1

  MNQ 0900 E0 RR2.0 GARCH<0.58 (N=108):
    2025: N= 108, avgR=+0.174, WR=41.7% [+]
    Years positive: 1/1

  MNQ 0900 E1 RR1.0 GARCH<0.58 (N=278):
    2025: N= 278, avgR=-0.071, WR=48.9% [-]
    Years positive: 0/1

  MNQ 0900 E1 RR2.0 GARCH<0.58 (N=278):
    2025: N= 278, avgR=-0.034, WR=33.8% [-]
    Years positive: 0/1

======================================================================
PART 4: REDUNDANCY -- GARCH RATIO vs ATR VELOCITY REGIME
======================================================================

  MGC 0900 E0 RR2.0:
       Expanding x   GARCH<0.58: N=  89, avgR=-0.050, WR=39.3%
       Expanding x  GARCH>=0.58: N= 425, avgR=-0.171, WR=37.9%
          Stable x   GARCH<0.58: N= 252, avgR=-0.255, WR=38.9%
          Stable x  GARCH>=0.58: N=1118, avgR=-0.384, WR=32.7%
     Contracting x   GARCH<0.58: N=  99, avgR=-0.172, WR=41.4%
     Contracting x  GARCH>=0.58: N= 388, avgR=-0.504, WR=26.5%

  MES 0900 E0 RR2.0:
       Expanding x   GARCH<0.58: N=  50, avgR=+0.068, WR=44.0%
       Expanding x  GARCH>=0.58: N= 585, avgR=-0.088, WR=38.6%
          Stable x   GARCH<0.58: N= 247, avgR=-0.086, WR=38.9%
          Stable x  GARCH>=0.58: N=1119, avgR=-0.133, WR=39.9%
     Contracting x   GARCH<0.58: N= 207, avgR=-0.164, WR=39.6%
     Contracting x  GARCH>=0.58: N= 491, avgR=-0.262, WR=36.5%

  MNQ 0900 E0 RR2.0:
       Expanding x   GARCH<0.58: N=  13, avgR=+0.064, WR=38.5%
       Expanding x  GARCH>=0.58: N=  95, avgR=+0.290, WR=45.3%
          Stable x   GARCH<0.58: N=  54, avgR=+0.622, WR=57.4%
          Stable x  GARCH>=0.58: N= 233, avgR=-0.116, WR=32.6%
     Contracting x   GARCH<0.58: N=  41, avgR=-0.381, WR=22.0%
     Contracting x  GARCH>=0.58: N= 102, avgR=-0.138, WR=32.4%

  CORRELATION: garch_atr_ratio vs atr_vel_ratio:
    MGC: r=0.042, p=0.0795, N=1769
    MES: r=0.505, p=0.0000, N=1778
    MNQ: r=0.496, p=0.0000, N=332

======================================================================
BH FDR CORRECTION (Part 1 tests)
======================================================================

  Total tests: 14
  BH survivors (q=0.10): 7
    MGC_0900_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_below=1977, delta=+0.125R, LOW GARCH HELPS
    MGC_0900_E0_RR2.0: raw_p=0.0001, p_bh=0.0004, N_below=678, delta=+0.160R, LOW GARCH HELPS
    MNQ_1000_E1_RR2.0: raw_p=0.0002, p_bh=0.0008, N_below=296, delta=-0.313R, LOW GARCH HURTS
    MGC_1000_E1_RR2.0: raw_p=0.0116, p_bh=0.0405, N_below=2081, delta=+0.060R, LOW GARCH HELPS
    MGC_1800_E1_RR2.0: raw_p=0.0167, p_bh=0.0467, N_below=2051, delta=+0.061R, LOW GARCH HELPS
    MNQ_1000_E0_RR2.0: raw_p=0.0270, p_bh=0.0631, N_below=111, delta=-0.303R, LOW GARCH HURTS
    MGC_1000_E0_RR2.0: raw_p=0.0430, p_bh=0.0861, N_below=687, delta=+0.080R, LOW GARCH HELPS

======================================================================
HONEST SUMMARY
======================================================================

### MNQ 0900 GARCH<0.58 Claim Check
  Claim: WR=60.7%, avgR=+0.26R when garch_atr_ratio < 0.58
  [Results above show actual values]

### CAVEATS
  - MNQ has only ~332 rows with GARCH data (2+ year warm-up eats early data)
  - GARCH<0.58 produces N~66 for MNQ -- REGIME size at best
  - Previous research (research_garch_vs_atr.py) already found ATR wins 6/8 sessions
  - 0.58 threshold appears cherry-picked from a continuous distribution
  - 0900 session NOT DST-split