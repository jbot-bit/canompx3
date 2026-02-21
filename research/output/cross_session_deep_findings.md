# Cross-Session Deep Dive Findings
**Date:** 2026-02-21
**Script:** research/research_cross_session_deep.py

======================================================================
PART 1: ORB SIZE CONCORDANCE -- YEAR-BY-YEAR HONESTY CHECK
======================================================================

  Total days with all 3 instruments: 516
    all_wide: 149 days (28.9%)
    all_narrow: 155 days (30.0%)
    mixed: 212 days (41.1%)

--- Year-by-year for each instrument x concordance x RR ---

  TOP 20 CONCORDANCE RESULTS (by raw p-value):
  Sym   EM   RR    Conc         N         avgR    base   delta        p  yrs+ label          
  ------------------------------------------------------------------------------------------
  MGC   E0   1.0   all_narrow   146     -0.366  -0.067  -0.299   0.0000 0/ 2 PRELIMINARY    
  MGC   E0   1.5   all_narrow   146     -0.370  -0.019  -0.351   0.0000 0/ 2 PRELIMINARY    
  MGC   E1   1.0   all_narrow   155     -0.390  -0.126  -0.264   0.0000 0/ 2 PRELIMINARY    
  MGC   E1   1.5   all_narrow   155     -0.452  -0.133  -0.318   0.0000 0/ 2 PRELIMINARY    
  MGC   E0   2.5   all_wide     148     +0.257  -0.250  +0.507   0.0001 2/ 3 PRELIMINARY    
  MGC   E0   1.5   all_wide     148     +0.133  -0.231  +0.364   0.0001 3/ 3 PRELIMINARY    
  MGC   E0   2.0   all_wide     148     +0.172  -0.255  +0.427   0.0002 2/ 3 PRELIMINARY    
  MGC   E0   2.0   all_narrow   146     -0.363  -0.030  -0.334   0.0002 0/ 2 PRELIMINARY    
  MGC   E0   1.0   all_wide     148     +0.037  -0.237  +0.275   0.0003 2/ 3 PRELIMINARY    
  MGC   E0   2.5   all_narrow   146     -0.349  +0.006  -0.355   0.0005 0/ 2 PRELIMINARY    
  MGC   E0   3.0   all_wide     148     +0.287  -0.208  +0.495   0.0007 3/ 3 PRELIMINARY    
  MGC   E1   1.5   all_wide     149     -0.003  -0.321  +0.318   0.0009 1/ 3 PRELIMINARY    
  MNQ   E0   2.5   all_narrow   153     -0.181  +0.216  -0.397    0.002 0/ 2 PRELIMINARY    
  MES   E1   3.0   all_narrow   155     -0.345  +0.021  -0.367    0.002 0/ 2 PRELIMINARY    
  MGC   E1   2.0   all_wide     149     +0.011  -0.326  +0.337    0.002 2/ 3 PRELIMINARY    
  MGC   E1   2.5   all_wide     149     +0.063  -0.314  +0.377    0.003 1/ 3 PRELIMINARY    
  MGC   E1   2.0   all_narrow   155     -0.410  -0.151  -0.260    0.003 0/ 2 PRELIMINARY    
  MGC   E0   3.0   all_narrow   146     -0.297  +0.038  -0.335    0.003 0/ 2 PRELIMINARY    
  MGC   E1   3.0   all_narrow   155     -0.392  -0.070  -0.321    0.003 0/ 2 PRELIMINARY    
  MES   E0   1.0   all_wide     146     +0.138  -0.077  +0.215    0.004 2/ 3 PRELIMINARY    

  YEAR-BY-YEAR DETAIL (top 5 by p-value):

  MGC E0 RR1.0 all_narrow (N=146, avgR=-0.366, p=0.0000):
    2024: N= 119, avgR= -0.426 [-]
    2025: N=  27, avgR= -0.102 [-]

  MGC E0 RR1.5 all_narrow (N=146, avgR=-0.370, p=0.0000):
    2024: N= 119, avgR= -0.449 [-]
    2025: N=  27, avgR= -0.022 [-]

  MGC E1 RR1.0 all_narrow (N=155, avgR=-0.390, p=0.0000):
    2024: N= 125, avgR= -0.456 [-]
    2025: N=  30, avgR= -0.116 [-]

  MGC E1 RR1.5 all_narrow (N=155, avgR=-0.452, p=0.0000):
    2024: N= 125, avgR= -0.490 [-]
    2025: N=  30, avgR= -0.294 [-]

  MGC E0 RR2.5 all_wide (N=148, avgR=+0.257, p=0.0001):
    2024: N=   9, avgR= +1.554 [+]
    2025: N= 124, avgR= +0.206 [+]
    2026: N=  15, avgR= -0.096 [-]

======================================================================
PART 1b: CONCORDANCE + G4 FILTER -- DO THEY STACK?
======================================================================
  MGC E0 RR2.0 G4+ all_wide: N=89, avgR=+0.218, p_vs_0=0.112, REGIME
  MGC E0 RR2.0 G4+ mixed: N=32, avgR=-0.109, p_vs_0=0.590, REGIME
  MES E0 RR2.0 G4+ all_wide: N=111, avgR=+0.217, p_vs_0=0.064, PRELIMINARY
  MES E0 RR2.0 G4+ mixed: N=54, avgR=+0.046, p_vs_0=0.772, REGIME
  MNQ E0 RR2.0 G4+ all_wide: N=147, avgR=+0.162, p_vs_0=0.135, PRELIMINARY
  MNQ E0 RR2.0 G4+ all_narrow: N=150, avgR=-0.082, p_vs_0=0.383, PRELIMINARY
  MNQ E0 RR2.0 G4+ mixed: N=212, avgR=+0.227, p_vs_0=0.010, CORE

======================================================================
PART 2: FRIDAY TOXICITY x VOL REGIME -- THRESHOLD SENSITIVITY
======================================================================

--- ATR percentile threshold sensitivity (Friday vs Mon-Thu) ---

  Sym   Pct   Vol        N_fri     Fri_R   M-T_R   Delta        p ATR_thresh
  --------------------------------------------------------------------------------
  MNQ   P50   high_vol   51       -0.275  +0.241  -0.516    0.008      304.6
  MNQ   P67   high_vol   35       -0.238  +0.268  -0.505    0.037      355.3
  MNQ   P75   high_vol   26       -0.300  +0.158  -0.459    0.092      396.0
  MGC   P75   high_vol   75       -0.254  +0.088  -0.343    0.011       35.9
  MGC   P67   high_vol   98       -0.243  +0.038  -0.281    0.013       31.4
  MNQ   P33   high_vol   69       -0.064  +0.187  -0.251    0.145      263.6
  MNQ   P25   low_vol    23       -0.079  +0.150  -0.229    0.411      246.3
  MGC   P50   high_vol   151      -0.311  -0.090  -0.221    0.010       26.6
  MNQ   P25   high_vol   78       -0.054  +0.157  -0.211    0.191      246.3
  MGC   P25   high_vol   226      -0.357  -0.160  -0.197    0.003       22.1
  MGC   P33   high_vol   196      -0.324  -0.151  -0.173    0.018       23.5
  MGC   P33   low_vol    91       -0.450  -0.281  -0.169    0.052       23.5
  MES   P75   high_vol   78       -0.048  +0.103  -0.151    0.307       75.8
  MNQ   P33   low_vol    32       -0.051  +0.091  -0.141    0.551      263.6
  MES   P33   high_vol   215      -0.073  +0.065  -0.139    0.111       48.7

--- Year-by-year for top toxic Friday combos ---

  MNQ P50 high_vol Friday (ATR>304.6):
    2024: N= 12, avgR= -0.018 [-]
    2025: N= 35, avgR= -0.361 [-]
    2026: N=  4, avgR= -0.286 [-]

  MNQ P67 high_vol Friday (ATR>355.3):
    2024: N=  5, avgR= +0.160 [+]
    2025: N= 29, avgR= -0.378 [-]

  MNQ P75 high_vol Friday (ATR>396.0):
    2024: N=  3, avgR= -0.034 [-]
    2025: N= 23, avgR= -0.335 [-]

--- Low-vol Friday: is there actually POSITIVE edge? ---
  MNQ P50 low_vol Friday: N=50, avgR=+0.159, p_vs_0=0.382, REGIME
  MES P33 low_vol Friday: N=88, avgR=+0.014, p_vs_0=0.901, REGIME
  MES P25 low_vol Friday: N=68, avgR=+0.004, p_vs_0=0.974, REGIME

======================================================================
PART 3: CONCORDANCE AT OTHER SESSIONS (0900, 1800)
======================================================================
  MGC 0900 all_wide: N=140, avgR=+0.142, baseline=-0.337, p=0.0000, PRELIMINARY
  MGC 0900 all_narrow: N=125, avgR=-0.388, baseline=-0.117, p=0.003, PRELIMINARY
  MES 0900 all_wide: N=141, avgR=-0.167, baseline=-0.131, p=0.727, PRELIMINARY
  MES 0900 all_narrow: N=133, avgR=-0.054, baseline=-0.176, p=0.237, PRELIMINARY
  MNQ 0900 all_wide: N=142, avgR=+0.018, baseline=+0.110, p=0.447, PRELIMINARY
  MNQ 0900 all_narrow: N=141, avgR=+0.242, baseline=+0.017, p=0.057, PRELIMINARY
  MGC 1800 all_wide: N=148, avgR=-0.008, baseline=-0.105, p=0.406, PRELIMINARY
  MGC 1800 all_narrow: N=152, avgR=-0.093, baseline=-0.070, p=0.825, PRELIMINARY

======================================================================
PART 4: IS CONCORDANCE JUST G4+ IN DISGUISE?
======================================================================
  MGC all_wide: 149 days, 60% pass G4, avg orb_size=6.2
  MGC all_narrow: 155 days, 0% pass G4, avg orb_size=1.1
  MGC mixed: 212 days, 16% pass G4, avg orb_size=2.6
  MGC WITHIN G4+: all_wide=89, not_wide=34
  MGC G4+ all_wide: N=89, avgR=+0.218 vs G4+ not_wide: N=32, avgR=-0.109, p=0.181, d=+0.270

  MES all_wide: 149 days, 77% pass G4, avg orb_size=6.6
  MES all_narrow: 155 days, 0% pass G4, avg orb_size=1.8
  MES mixed: 212 days, 25% pass G4, avg orb_size=3.5
  MES WITHIN G4+: all_wide=114, not_wide=54
  MES G4+ all_wide: N=111, avgR=+0.217 vs G4+ not_wide: N=54, avgR=+0.046, p=0.388, d=+0.143

  MNQ all_wide: 149 days, 100% pass G4, avg orb_size=30.8
  MNQ all_narrow: 155 days, 97% pass G4, avg orb_size=8.8
  MNQ mixed: 212 days, 100% pass G4, avg orb_size=16.8
  MNQ WITHIN G4+: all_wide=149, not_wide=363
  MNQ G4+ all_wide: N=147, avgR=+0.162 vs G4+ not_wide: N=362, avgR=+0.099, p=0.617, d=+0.050


======================================================================
BH FDR CORRECTION (Parts 1-4 pooled)
======================================================================

  Total tests: 90
  BH survivors (q=0.10): 48
    MGC_E0_RR1.0_all_narrow: raw_p=0.0000, p_bh=0.0001
    MGC_E0_RR1.5_all_narrow: raw_p=0.0000, p_bh=0.0002
    MGC_E1_RR1.0_all_narrow: raw_p=0.0000, p_bh=0.0005
    MGC_E1_RR1.5_all_narrow: raw_p=0.0000, p_bh=0.0005
    MGC_E0_RR2.5_all_wide: raw_p=0.0001, p_bh=0.002
    MGC_E0_RR1.5_all_wide: raw_p=0.0001, p_bh=0.002
    MGC_E0_RR2.0_all_wide: raw_p=0.0002, p_bh=0.002
    MGC_E0_RR2.0_all_narrow: raw_p=0.0002, p_bh=0.002
    MGC_E0_RR1.0_all_wide: raw_p=0.0003, p_bh=0.003
    MGC_E0_RR2.5_all_narrow: raw_p=0.0005, p_bh=0.004
    MGC_E0_RR3.0_all_wide: raw_p=0.0007, p_bh=0.006
    MGC_E1_RR1.5_all_wide: raw_p=0.0009, p_bh=0.007
    MNQ_E0_RR2.5_all_narrow: raw_p=0.002, p_bh=0.013
    MES_E1_RR3.0_all_narrow: raw_p=0.002, p_bh=0.013
    MGC_E1_RR2.0_all_wide: raw_p=0.002, p_bh=0.015
    MGC_E1_RR2.5_all_wide: raw_p=0.003, p_bh=0.015
    MGC_E1_RR2.0_all_narrow: raw_p=0.003, p_bh=0.015
    MGC_E0_RR3.0_all_narrow: raw_p=0.003, p_bh=0.015
    MGC_E1_RR3.0_all_narrow: raw_p=0.003, p_bh=0.015
    FRI_MGC_P25_high_vol: raw_p=0.003, p_bh=0.015

--- Concordance tests with year-consistency filter ---
  50%+ years positive AND raw p<0.05: 10 tests
    MGC E0 RR2.5 all_wide: N=148, avgR=+0.257, 2/3 yrs+, p=0.0001
    MGC E0 RR1.5 all_wide: N=148, avgR=+0.133, 3/3 yrs+, p=0.0001
    MGC E0 RR2.0 all_wide: N=148, avgR=+0.172, 2/3 yrs+, p=0.0002
    MGC E0 RR1.0 all_wide: N=148, avgR=+0.037, 2/3 yrs+, p=0.0003
    MGC E0 RR3.0 all_wide: N=148, avgR=+0.287, 3/3 yrs+, p=0.0007
  60%+ years positive AND raw p<0.05: 10 tests
    MGC E0 RR2.5 all_wide: N=148, avgR=+0.257, 2/3 yrs+, p=0.0001
    MGC E0 RR1.5 all_wide: N=148, avgR=+0.133, 3/3 yrs+, p=0.0001
    MGC E0 RR2.0 all_wide: N=148, avgR=+0.172, 2/3 yrs+, p=0.0002
    MGC E0 RR1.0 all_wide: N=148, avgR=+0.037, 2/3 yrs+, p=0.0003
    MGC E0 RR3.0 all_wide: N=148, avgR=+0.287, 3/3 yrs+, p=0.0007
  67%+ years positive AND raw p<0.05: 5 tests
    MGC E0 RR1.5 all_wide: N=148, avgR=+0.133, 3/3 yrs+, p=0.0001
    MGC E0 RR3.0 all_wide: N=148, avgR=+0.287, 3/3 yrs+, p=0.0007
    MGC E1 RR3.0 all_wide: N=149, avgR=+0.110, 3/3 yrs+, p=0.006
    MES E0 RR2.0 all_wide: N=146, avgR=+0.199, 3/3 yrs+, p=0.014
    MES E0 RR1.5 all_wide: N=146, avgR=+0.115, 3/3 yrs+, p=0.040
  75%+ years positive AND raw p<0.05: 5 tests
    MGC E0 RR1.5 all_wide: N=148, avgR=+0.133, 3/3 yrs+, p=0.0001
    MGC E0 RR3.0 all_wide: N=148, avgR=+0.287, 3/3 yrs+, p=0.0007
    MGC E1 RR3.0 all_wide: N=149, avgR=+0.110, 3/3 yrs+, p=0.006
    MES E0 RR2.0 all_wide: N=146, avgR=+0.199, 3/3 yrs+, p=0.014
    MES E0 RR1.5 all_wide: N=146, avgR=+0.115, 3/3 yrs+, p=0.040

## HONEST SUMMARY

### SURVIVED
- [List BH survivors with consistent year-by-year]

### DID NOT SURVIVE
- [List findings that failed year-by-year or BH]

### CAVEATS
- Concordance medians computed over full sample (survivorship in median calc)
- MNQ only has 2 years of data -- any MNQ finding is REGIME at best
- 0900 session tests NOT DST-split (applies to Part 3 only)

### NEXT STEPS
- [Based on what survives]
