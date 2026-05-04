# Compressed ORB Filter Validation
**Date:** 2026-02-21
**Script:** research/research_compressed_orb.py

======================================================================
PART 0: DATA COVERAGE
======================================================================

  MGC 0900: total=2922, has_tier=1563, Compressed=570, Neutral=634, Expanded=359
  MGC 1000: total=2922, has_tier=1565, Compressed=537, Neutral=643, Expanded=385
  MGC 1800: total=2922, has_tier=1564, Compressed=567, Neutral=599, Expanded=398
  MES 0900: total=2047, has_tier=1572, Compressed=598, Neutral=574, Expanded=400
  MES 1000: total=2047, has_tier=1574, Compressed=578, Neutral=556, Expanded=440
  MES 1800: total=2047, has_tier=1573, Compressed=584, Neutral=564, Expanded=425
  MNQ 0900: total=584, has_tier=510, Compressed=194, Neutral=208, Expanded=108
  MNQ 1000: total=584, has_tier=510, Compressed=186, Neutral=189, Expanded=135
  MNQ 1800: total=584, has_tier=509, Compressed=173, Neutral=194, Expanded=142

======================================================================
PART 1: COMPRESSION TIER vs OUTCOME (all instruments x sessions)
======================================================================

  MGC_0900_E1_RR2.0:
    Compressed: N=2776, avgR=-0.488, WR=28.0%
    Non-Comp:   N=4526, avgR=-0.299, WR=33.1%
    Delta: -0.188R, WR delta: -5.0pp, p=0.0000, CORE
        Compressed: N=2776, avgR=-0.488, WR=28.0%
           Neutral: N=2989, avgR=-0.342, WR=32.2%
          Expanded: N=1537, avgR=-0.215, WR=34.7%

  MGC_1000_E1_RR2.0:
    Compressed: N=2631, avgR=-0.346, WR=34.1%
    Non-Comp:   N=5043, avgR=-0.246, WR=34.9%
    Delta: -0.099R, WR delta: -0.8pp, p=0.0000, CORE
        Compressed: N=2631, avgR=-0.346, WR=34.1%
           Neutral: N=3165, avgR=-0.268, WR=35.4%
          Expanded: N=1878, avgR=-0.210, WR=34.1%

  MGC_1800_E1_RR2.0:
    Compressed: N=2736, avgR=-0.307, WR=33.0%
    Non-Comp:   N=4849, avgR=-0.186, WR=35.3%
    Delta: -0.121R, WR delta: -2.3pp, p=0.0000, CORE
        Compressed: N=2736, avgR=-0.307, WR=33.0%
           Neutral: N=2933, avgR=-0.202, WR=35.7%
          Expanded: N=1916, avgR=-0.161, WR=34.7%

  MES_0900_E1_RR2.0:
    Compressed: N=2904, avgR=-0.169, WR=37.0%
    Non-Comp:   N=4470, avgR=-0.266, WR=29.0%
    Delta: +0.097R, WR delta: +8.0pp, p=0.0003, CORE
        Compressed: N=2904, avgR=-0.169, WR=37.0%
           Neutral: N=2708, avgR=-0.336, WR=26.8%
          Expanded: N=1762, avgR=-0.158, WR=32.3%

  MES_1000_E1_RR2.0:
    Compressed: N=2870, avgR=-0.201, WR=33.7%
    Non-Comp:   N=4920, avgR=-0.124, WR=34.0%
    Delta: -0.077R, WR delta: -0.3pp, p=0.0051, CORE
        Compressed: N=2870, avgR=-0.201, WR=33.7%
           Neutral: N=2763, avgR=-0.162, WR=33.2%
          Expanded: N=2157, avgR=-0.076, WR=34.9%

  MNQ_0900_E1_RR2.0:
    Compressed: N= 925, avgR=-0.081, WR=34.4%
    Non-Comp:   N=1444, avgR=-0.071, WR=33.2%
    Delta: -0.010R, WR delta: +1.2pp, p=0.8547, CORE
        Compressed: N= 925, avgR=-0.081, WR=34.4%
           Neutral: N= 974, avgR=+0.005, WR=36.1%
          Expanded: N= 470, avgR=-0.230, WR=27.0%

  MNQ_1000_E1_RR2.0:
    Compressed: N= 921, avgR=-0.105, WR=33.0%
    Non-Comp:   N=1587, avgR=+0.039, WR=36.7%
    Delta: -0.144R, WR delta: -3.7pp, p=0.0081, CORE
        Compressed: N= 921, avgR=-0.105, WR=33.0%
           Neutral: N= 933, avgR=-0.002, WR=35.7%
          Expanded: N= 654, avgR=+0.097, WR=38.2%

======================================================================
PART 1b: SAME AT E0 (strongest entry model)
======================================================================

  MGC_0900_E0_RR2.0:
    Compressed: N= 866, avgR=-0.453, WR=28.4%
    Non-Comp:   N=1505, avgR=-0.260, WR=37.1%
    Delta: -0.193R, WR delta: -8.7pp, p=0.0000, CORE

  MGC_1000_E0_RR2.0:
    Compressed: N= 843, avgR=-0.326, WR=39.4%
    Non-Comp:   N=1681, avgR=-0.219, WR=39.1%
    Delta: -0.108R, WR delta: +0.3pp, p=0.0043, CORE

  MGC_1800_E0_RR2.0:
    Compressed: N= 972, avgR=-0.220, WR=42.7%
    Non-Comp:   N=1694, avgR=-0.116, WR=41.1%
    Delta: -0.104R, WR delta: +1.6pp, p=0.0084, CORE

  MES_0900_E0_RR2.0:
    Compressed: N=1054, avgR=-0.077, WR=46.2%
    Non-Comp:   N=1645, avgR=-0.182, WR=34.3%
    Delta: +0.105R, WR delta: +11.9pp, p=0.0140, CORE

  MES_1000_E0_RR2.0:
    Compressed: N=1018, avgR=-0.082, WR=42.8%
    Non-Comp:   N=1773, avgR=+0.001, WR=40.4%
    Delta: -0.082R, WR delta: +2.4pp, p=0.0648, CORE

  MNQ_0900_E0_RR2.0:
    Compressed: N= 336, avgR=-0.007, WR=39.0%
    Non-Comp:   N= 559, avgR=+0.060, WR=38.5%
    Delta: -0.067R, WR delta: +0.5pp, p=0.4537, CORE

  MNQ_1000_E0_RR2.0:
    Compressed: N= 369, avgR=+0.032, WR=39.6%
    Non-Comp:   N= 619, avgR=+0.111, WR=40.1%
    Delta: -0.079R, WR delta: -0.5pp, p=0.3579, CORE

======================================================================
PART 2: YEAR-BY-YEAR CONSISTENCY (MNQ 0900 focus)
======================================================================

  MNQ 0900 E0 RR1.0 Compressed (N=336):
    2024: N= 157, avgR=+0.100, WR=68.8% [+]
    2025: N= 169, avgR=+0.045, WR=59.2% [+]
    2026: N=  10, avgR=+0.095, WR=60.0% [+]
    Years positive: 3/3

  MNQ 0900 E0 RR1.5 Compressed (N=336):
    2024: N= 157, avgR=+0.082, WR=53.5% [+]
    2025: N= 169, avgR=-0.128, WR=39.6% [-]
    2026: N=  10, avgR=+0.368, WR=60.0% [+]
    Years positive: 2/3

  MNQ 0900 E0 RR2.0 Compressed (N=336):
    2024: N= 157, avgR=-0.018, WR=40.1% [-]
    2025: N= 169, avgR=-0.019, WR=37.3% [-]
    2026: N=  10, avgR=+0.367, WR=50.0% [+]
    Years positive: 1/3

  MNQ 0900 E1 RR1.0 Compressed (N=925):
    2024: N= 441, avgR=-0.081, WR=53.1% [-]
    2025: N= 464, avgR=-0.129, WR=47.8% [-]
    2026: N=  20, avgR=+0.113, WR=60.0% [+]
    Years positive: 1/3

  MNQ 0900 E1 RR1.5 Compressed (N=925):
    2024: N= 441, avgR=-0.166, WR=38.3% [-]
    2025: N= 464, avgR=-0.083, WR=40.3% [-]
    2026: N=  20, avgR=+0.274, WR=55.0% [+]
    Years positive: 1/3

  MNQ 0900 E1 RR2.0 Compressed (N=925):
    2024: N= 441, avgR=-0.162, WR=32.2% [-]
    2025: N= 464, avgR=-0.025, WR=35.8% [-]
    2026: N=  20, avgR=+0.390, WR=50.0% [+]
    Years positive: 1/3

======================================================================
PART 3: REDUNDANCY CHECK -- COMPRESSION vs ATR VELOCITY
======================================================================

  MGC 0900 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N= 144, avgR=-0.467, WR=25.7%
       Expanding x      Neutral: N= 222, avgR=-0.017, WR=46.8%
       Expanding x     Expanded: N= 148, avgR=-0.040, WR=37.2%
          Stable x   Compressed: N= 567, avgR=-0.422, WR=30.7%
          Stable x      Neutral: N= 548, avgR=-0.356, WR=34.7%
          Stable x     Expanded: N= 255, avgR=-0.230, WR=39.2%
     Contracting x   Compressed: N= 155, avgR=-0.551, WR=22.6%
     Contracting x      Neutral: N= 214, avgR=-0.496, WR=29.0%
     Contracting x     Expanded: N= 118, avgR=-0.178, WR=39.8%

  MGC 1000 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N= 139, avgR=-0.399, WR=33.1%
       Expanding x      Neutral: N= 209, avgR=-0.185, WR=40.2%
       Expanding x     Expanded: N= 150, avgR=+0.036, WR=43.3%
          Stable x   Compressed: N= 549, avgR=-0.285, WR=42.3%
          Stable x      Neutral: N= 600, avgR=-0.249, WR=40.2%
          Stable x     Expanded: N= 357, avgR=-0.264, WR=35.3%
     Contracting x   Compressed: N= 155, avgR=-0.407, WR=34.8%
     Contracting x      Neutral: N= 224, avgR=-0.265, WR=40.2%
     Contracting x     Expanded: N= 141, avgR=-0.220, WR=36.2%

  MES 0900 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N= 224, avgR=-0.008, WR=46.0%
       Expanding x      Neutral: N= 238, avgR=-0.176, WR=34.0%
       Expanding x     Expanded: N= 173, avgR=-0.026, WR=37.0%
          Stable x   Compressed: N= 547, avgR=-0.042, WR=47.3%
          Stable x      Neutral: N= 508, avgR=-0.236, WR=33.9%
          Stable x     Expanded: N= 311, avgR=-0.088, WR=36.0%
     Contracting x   Compressed: N= 283, avgR=-0.200, WR=44.2%
     Contracting x      Neutral: N= 241, avgR=-0.265, WR=33.6%
     Contracting x     Expanded: N= 174, avgR=-0.241, WR=31.6%

  MES 1000 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N= 244, avgR=+0.014, WR=45.1%
       Expanding x      Neutral: N= 235, avgR=-0.019, WR=40.0%
       Expanding x     Expanded: N= 214, avgR=+0.219, WR=46.3%
          Stable x   Compressed: N= 528, avgR=-0.052, WR=44.7%
          Stable x      Neutral: N= 506, avgR=+0.035, WR=43.5%
          Stable x     Expanded: N= 342, avgR=-0.078, WR=36.0%
     Contracting x   Compressed: N= 246, avgR=-0.240, WR=36.6%
     Contracting x      Neutral: N= 252, avgR=-0.170, WR=35.3%
     Contracting x     Expanded: N= 224, avgR=+0.048, WR=41.1%

  MNQ 0900 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N=  62, avgR=+0.168, WR=43.5%
       Expanding x      Neutral: N=  90, avgR=+0.171, WR=42.2%
       Expanding x     Expanded: N=  46, avgR=+0.360, WR=47.8%
          Stable x   Compressed: N= 183, avgR=+0.040, WR=40.4%
          Stable x      Neutral: N= 191, avgR=+0.192, WR=44.0%
          Stable x     Expanded: N=  94, avgR=-0.339, WR=23.4%
     Contracting x   Compressed: N=  91, avgR=-0.220, WR=33.0%
     Contracting x      Neutral: N=  95, avgR=-0.345, WR=24.2%
     Contracting x     Expanded: N=  43, avgR=+0.684, WR=60.5%

  MNQ 1000 E0 RR2.0 -- 2x2 matrix:
       Expanding x   Compressed: N=  85, avgR=+0.217, WR=45.9%
       Expanding x      Neutral: N=  76, avgR=+0.535, WR=55.3%
       Expanding x     Expanded: N=  59, avgR=+0.497, WR=52.5%
          Stable x   Compressed: N= 194, avgR=+0.065, WR=40.7%
          Stable x      Neutral: N= 199, avgR=+0.123, WR=41.2%
          Stable x     Expanded: N= 118, avgR=+0.269, WR=44.9%
     Contracting x   Compressed: N=  90, avgR=-0.214, WR=31.1%
     Contracting x      Neutral: N= 109, avgR=-0.403, WR=22.0%
     Contracting x     Expanded: N=  58, avgR=-0.232, WR=27.6%

======================================================================
PART 4: Z-SCORE THRESHOLD SENSITIVITY (MNQ 0900)
======================================================================

  MNQ 0900 E0 RR2.0 -- z-score thresholds:
    z<=-1.00: N=  98, avgR=-0.020, WR=39.8% | z>-1.00: N= 797, avgR=+0.041, WR=38.5% | delta=-0.061, p=0.6458
    z<=-0.50: N= 336, avgR=-0.007, WR=39.0% | z>-0.50: N= 559, avgR=+0.060, WR=38.5% | delta=-0.067, p=0.4537
    z<=-0.25: N= 437, avgR=+0.015, WR=39.4% | z>-0.25: N= 458, avgR=+0.053, WR=38.0% | delta=-0.038, p=0.6615
    z<=+0.00: N= 552, avgR=+0.040, WR=39.9% | z>+0.00: N= 343, avgR=+0.026, WR=36.7% | delta=+0.015, p=0.8730
    z<=+0.25: N= 641, avgR=+0.010, WR=38.5% | z>+0.25: N= 254, avgR=+0.098, WR=39.0% | delta=-0.088, p=0.3815
    z<=+0.50: N= 712, avgR=+0.024, WR=38.8% | z>+0.50: N= 183, avgR=+0.077, WR=38.3% | delta=-0.053, p=0.6365
    z<=+1.00: N= 783, avgR=-0.008, WR=37.4% | z>+1.00: N= 112, avgR=+0.334, WR=47.3% | delta=-0.342, p=0.0169

  MNQ 0900 E1 RR2.0 -- z-score thresholds:
    z<=-1.50: N=  35, avgR=-0.098, WR=37.1% | z>-1.50: N=2334, avgR=-0.075, WR=33.6% | delta=-0.023, p=0.9102
    z<=-1.00: N= 271, avgR=-0.078, WR=35.1% | z>-1.00: N=2098, avgR=-0.075, WR=33.5% | delta=-0.003, p=0.9686
    z<=-0.50: N= 925, avgR=-0.081, WR=34.4% | z>-0.50: N=1444, avgR=-0.071, WR=33.2% | delta=-0.010, p=0.8547
    z<=-0.25: N=1197, avgR=-0.004, WR=36.9% | z>-0.25: N=1172, avgR=-0.148, WR=30.3% | delta=+0.144, p=0.0069
    z<=+0.00: N=1479, avgR=+0.020, WR=37.6% | z>+0.00: N= 890, avgR=-0.234, WR=27.1% | delta=+0.255, p=0.0000
    z<=+0.25: N=1709, avgR=-0.017, WR=36.2% | z>+0.25: N= 660, avgR=-0.227, WR=27.1% | delta=+0.210, p=0.0004
    z<=+0.50: N=1899, avgR=-0.037, WR=35.3% | z>+0.50: N= 470, avgR=-0.230, WR=27.0% | delta=+0.193, p=0.0035
    z<=+1.00: N=2067, avgR=-0.070, WR=34.0% | z>+1.00: N= 302, avgR=-0.112, WR=31.1% | delta=+0.042, p=0.6056

======================================================================
BH FDR CORRECTION (all tests pooled)
======================================================================

  Total tests: 14
  BH survivors (q=0.10): 11
    MGC_0900_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_comp=2776, delta=-0.188R, WR_comp=28.0%, WR_other=33.1%
    MGC_0900_E0_RR2.0: raw_p=0.0000, p_bh=0.0000, N_comp=866, delta=-0.193R, WR_comp=28.4%, WR_other=37.1%
    MGC_1800_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_comp=2736, delta=-0.121R, WR_comp=33.0%, WR_other=35.3%
    MGC_1000_E1_RR2.0: raw_p=0.0000, p_bh=0.0001, N_comp=2631, delta=-0.099R, WR_comp=34.1%, WR_other=34.9%
    MES_0900_E1_RR2.0: raw_p=0.0003, p_bh=0.0009, N_comp=2904, delta=+0.097R, WR_comp=37.0%, WR_other=29.0%
    MGC_1000_E0_RR2.0: raw_p=0.0043, p_bh=0.0100, N_comp=843, delta=-0.108R, WR_comp=39.4%, WR_other=39.1%
    MES_1000_E1_RR2.0: raw_p=0.0051, p_bh=0.0103, N_comp=2870, delta=-0.077R, WR_comp=33.7%, WR_other=34.0%
    MNQ_1000_E1_RR2.0: raw_p=0.0081, p_bh=0.0130, N_comp=921, delta=-0.144R, WR_comp=33.0%, WR_other=36.7%
    MGC_1800_E0_RR2.0: raw_p=0.0084, p_bh=0.0130, N_comp=972, delta=-0.104R, WR_comp=42.7%, WR_other=41.1%
    MES_0900_E0_RR2.0: raw_p=0.0140, p_bh=0.0196, N_comp=1054, delta=+0.105R, WR_comp=46.2%, WR_other=34.3%
    MES_1000_E0_RR2.0: raw_p=0.0648, p_bh=0.0825, N_comp=1018, delta=-0.082R, WR_comp=42.8%, WR_other=40.4%

======================================================================
HONEST SUMMARY
======================================================================

### MNQ 0900 Compressed ORB Claim Check
  MNQ_0900_E1_RR2.0: WR_comp=34.4%, WR_other=33.2%, WR_delta=+1.2pp, p=0.8547
  MNQ_0900_E0_RR2.0: WR_comp=39.0%, WR_other=38.5%, WR_delta=+0.5pp, p=0.4537

### VERDICT
  BH-significant compression effects found:
    MGC_0900_E1_RR2.0: NEGATIVE (compressed hurts), delta=-0.188R, p_bh=0.0000
    MGC_0900_E0_RR2.0: NEGATIVE (compressed hurts), delta=-0.193R, p_bh=0.0000
    MGC_1800_E1_RR2.0: NEGATIVE (compressed hurts), delta=-0.121R, p_bh=0.0000
    MGC_1000_E1_RR2.0: NEGATIVE (compressed hurts), delta=-0.099R, p_bh=0.0001
    MES_0900_E1_RR2.0: POSITIVE (compressed helps), delta=+0.097R, p_bh=0.0009
    MGC_1000_E0_RR2.0: NEGATIVE (compressed hurts), delta=-0.108R, p_bh=0.0100
    MES_1000_E1_RR2.0: NEGATIVE (compressed hurts), delta=-0.077R, p_bh=0.0103
    MNQ_1000_E1_RR2.0: NEGATIVE (compressed hurts), delta=-0.144R, p_bh=0.0130
    MGC_1800_E0_RR2.0: NEGATIVE (compressed hurts), delta=-0.104R, p_bh=0.0130
    MES_0900_E0_RR2.0: POSITIVE (compressed helps), delta=+0.105R, p_bh=0.0196
    MES_1000_E0_RR2.0: NEGATIVE (compressed hurts), delta=-0.082R, p_bh=0.0825

### CAVEATS
  - MNQ only has ~2 years of data -- any MNQ finding is PRELIMINARY at best
  - Compression tier z=-0.5 threshold is a design choice, not optimized
  - 0900 session NOT DST-split in this analysis
  - orb_0900_compression_tier compares vs prior 20 days of SAME instrument only
  - This is NOT the cross-instrument 'all_narrow' concordance from deep research