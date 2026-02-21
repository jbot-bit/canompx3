# Cross-Session & Cross-Instrument Research Findings
**Date:** 2026-02-21
**Script:** research/research_cross_session.py
**Total tests:** 31
**BH survivors (q=0.10):** 6

## BH Survivors
- **Q2: ORB size concordance at 1000: MGC_1000_all_wide**: raw_p=0.0002, p_bh=0.003
- **Q2: ORB size concordance at 1000: MGC_1000_all_narrow**: raw_p=0.0002, p_bh=0.003
- **Q2: ORB size concordance at 1000: MNQ_1000_all_narrow**: raw_p=0.009, p_bh=0.062
- **Q5: Friday × vol regime: MGC_1000_fri_high_vol**: raw_p=0.010, p_bh=0.062
- **Q5: Friday × vol regime: MNQ_1000_fri_high_vol**: raw_p=0.008, p_bh=0.062
- **Q2: ORB size concordance at 1000: MES_1000_all_wide**: raw_p=0.014, p_bh=0.074

## Top 5 by Raw P-Value
- **Q2: ORB size concordance at 1000: MGC_1000_all_wide**: raw_p=0.0002, p_bh=0.003
- **Q2: ORB size concordance at 1000: MGC_1000_all_narrow**: raw_p=0.0002, p_bh=0.003
- **Q5: Friday × vol regime: MNQ_1000_fri_high_vol**: raw_p=0.008, p_bh=0.062
- **Q2: ORB size concordance at 1000: MNQ_1000_all_narrow**: raw_p=0.009, p_bh=0.062
- **Q5: Friday × vol regime: MGC_1000_fri_high_vol**: raw_p=0.010, p_bh=0.062

## Q1: MGC 0900 -> MES/MNQ 1000
- MES_1000_mgc_align_aligned: N=736, avgR=+0.014, p=0.936, d=+0.004
- MES_1000_mgc_align_opposed: N=757, avgR=+0.009, p=0.936, d=-0.004
- MNQ_1000_mgc_align_aligned: N=218, avgR=+0.184, p=0.189, d=+0.120
- MNQ_1000_mgc_align_opposed: N=279, avgR=+0.034, p=0.189, d=-0.120

## Q2: ORB size concordance at 1000
- MGC_1000_all_wide: N=149, avgR=+0.172, p=0.0002, d=+0.395 **BH-SIG**
- MGC_1000_all_narrow: N=155, avgR=-0.363, p=0.0002, d=-0.347 **BH-SIG**
- MGC_1000_mixed: N=212, avgR=-0.177, p=0.371, d=-0.081
- MES_1000_all_wide: N=149, avgR=+0.199, p=0.014, d=+0.249 **BH-SIG**
- MES_1000_all_narrow: N=155, avgR=-0.152, p=0.033, d=-0.200
- MES_1000_mixed: N=212, avgR=-0.033, p=0.594, d=-0.048
- MNQ_1000_all_wide: N=149, avgR=+0.162, p=0.567, d=+0.057
- MNQ_1000_all_narrow: N=155, avgR=-0.100, p=0.009, d=-0.247 **BH-SIG**
- MNQ_1000_mixed: N=212, avgR=+0.227, p=0.078, d=+0.159

## Q3: Gap × break direction
- MGC_1000_big_gap_vs_small: N=612, avgR=-0.191, p=0.237, d=+0.063
- MES_1000_big_gap_vs_small: N=445, avgR=+0.084, p=0.098, d=+0.094
- MNQ_1000_big_gap_vs_small: N=216, avgR=+0.180, p=0.303, d=+0.093

## Q4: Same-instrument session cascading
- MGC_0900brk_to_1000: N=1494, avgR=-0.230, p=0.577, d=-0.066
- MGC_cascade_same: N=739, avgR=-0.237, p=0.782, d=-0.015
- MGC_cascade_flip: N=755, avgR=-0.223, p=0.782, d=+0.015
- MES_0900brk_to_1000: N=1505, avgR=-0.005, p=0.031, d=-0.263
- MES_cascade_same: N=729, avgR=+0.061, p=0.026, d=+0.116
- MES_cascade_flip: N=776, avgR=-0.067, p=0.026, d=-0.116
- MNQ_0900brk_to_1000: N=486, avgR=+0.121, p=0.435, d=+0.143
- MNQ_cascade_same: N=252, avgR=+0.159, p=0.493, d=+0.063
- MNQ_cascade_flip: N=234, avgR=+0.080, p=0.493, d=-0.063

## Q5: Friday × vol regime
- MES_1000_fri_high_vol: N=158, avgR=-0.032, p=0.307, d=-0.092
- MES_1000_fri_low_vol: N=153, avgR=-0.058, p=0.794, d=-0.024
- MGC_1000_fri_high_vol: N=159, avgR=-0.311, p=0.010, d=-0.228 **BH-SIG**
- MGC_1000_fri_low_vol: N=149, avgR=-0.423, p=0.092, d=-0.160
- MNQ_1000_fri_high_vol: N=52, avgR=-0.275, p=0.008, d=-0.416 **BH-SIG**
- MNQ_1000_fri_low_vol: N=50, avgR=+0.159, p=0.654, d=+0.072

## Methodology
- All tests: Welch's t-test (unequal variance)
- FDR: Benjamini-Hochberg at q=0.10
- Anchor: E0 CB1 RR2.0 at 1000 session (standardized comparison)
- Look-ahead: break_dir known at entry time; gap/ATR known at day start
- orb_minutes=5 filter on all daily_features joins
