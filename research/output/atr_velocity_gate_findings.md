# ATR Velocity Contraction Gate — Quantified Improvement
**Date:** 2026-02-22
**Script:** research/research_atr_velocity_gate.py

======================================================================
PART 0: GATE REMOVAL RATES
======================================================================

  Gate logic: skip when atr_vel_regime='Contracting' AND
  orb_{session}_compression_tier IN ('Neutral','Compressed')

  MNQ 0900: 186/895 days skipped (20.8%), contracting total=229
  MNQ 1000: 199/988 days skipped (20.1%), contracting total=257
  MGC 0900: 369/2371 days skipped (15.6%), contracting total=487
  MGC 1000: 379/2524 days skipped (15.0%), contracting total=520
  MES 0900: 524/2699 days skipped (19.4%), contracting total=698
  MES 1000: 498/2791 days skipped (17.8%), contracting total=722

======================================================================
PART 1: SHARPE LIFT FROM ATR VELOCITY GATE
======================================================================

  Comparing: ALL trades vs GATED trades (skipping Contracting+Neutral/Compressed)
  Sharpe = annualized (mean/std * sqrt(252)) on daily R returns

  MNQ_0900_E0_RR2.0:
    ALL:     N=  895, avgR=+0.035, WR=38.7%, Sharpe=+0.42
    GATED:   N=  709, avgR=+0.118, WR=41.3%, Sharpe=+1.40
    SKIPPED: N=  186, avgR=-0.284, WR=28.5%
    Sharpe lift: +0.981
    Skipped vs Gated: delta=-0.402R, p=0.0001

  MNQ_0900_E1_RR2.0:
    ALL:     N= 2369, avgR=-0.075, WR=33.6%, Sharpe=-0.92
    GATED:   N= 1917, avgR=-0.050, WR=34.3%, Sharpe=-0.61
    SKIPPED: N=  452, avgR=-0.181, WR=31.0%
    Sharpe lift: +0.310
    Skipped vs Gated: delta=-0.130R, p=0.0463

  MNQ_1000_E0_RR2.0:
    ALL:     N=  988, avgR=+0.081, WR=39.9%, Sharpe=+0.97
    GATED:   N=  789, avgR=+0.182, WR=43.3%, Sharpe=+2.13
    SKIPPED: N=  199, avgR=-0.318, WR=26.1%
    Sharpe lift: +1.162
    Skipped vs Gated: delta=-0.500R, p=0.0000

  MNQ_1000_E1_RR2.0:
    ALL:     N= 2508, avgR=-0.014, WR=35.4%, Sharpe=-0.17
    GATED:   N= 2030, avgR=+0.072, WR=38.3%, Sharpe=+0.84
    SKIPPED: N=  478, avgR=-0.381, WR=22.8%
    Sharpe lift: +1.011
    Skipped vs Gated: delta=-0.453R, p=0.0000

  MGC_0900_E0_RR2.0:
    ALL:     N= 2371, avgR=-0.330, WR=33.9%, Sharpe=-5.71
    GATED:   N= 2002, avgR=-0.295, WR=35.3%, Sharpe=-5.00
    SKIPPED: N=  369, avgR=-0.519, WR=26.3%
    Sharpe lift: +0.712
    Skipped vs Gated: delta=-0.224R, p=0.0000

  MGC_0900_E1_RR2.0:
    ALL:     N= 7302, avgR=-0.371, WR=31.2%, Sharpe=-6.18
    GATED:   N= 6164, avgR=-0.340, WR=32.4%, Sharpe=-5.55
    SKIPPED: N= 1138, avgR=-0.538, WR=24.3%
    Sharpe lift: +0.632
    Skipped vs Gated: delta=-0.198R, p=0.0000

  MGC_1000_E0_RR2.0:
    ALL:     N= 2524, avgR=-0.255, WR=39.2%, Sharpe=-4.25
    GATED:   N= 2145, avgR=-0.243, WR=39.4%, Sharpe=-4.00
    SKIPPED: N=  379, avgR=-0.323, WR=38.0%
    Sharpe lift: +0.255
    Skipped vs Gated: delta=-0.081R, p=0.1042

  MGC_1000_E1_RR2.0:
    ALL:     N= 7674, avgR=-0.281, WR=34.6%, Sharpe=-4.41
    GATED:   N= 6479, avgR=-0.261, WR=35.2%, Sharpe=-4.05
    SKIPPED: N= 1195, avgR=-0.388, WR=31.6%
    Sharpe lift: +0.369
    Skipped vs Gated: delta=-0.127R, p=0.0000

  MES_0900_E0_RR2.0:
    ALL:     N= 2699, avgR=-0.141, WR=39.0%, Sharpe=-2.03
    GATED:   N= 2175, avgR=-0.120, WR=38.9%, Sharpe=-1.68
    SKIPPED: N=  524, avgR=-0.230, WR=39.3%
    Sharpe lift: +0.345
    Skipped vs Gated: delta=-0.110R, p=0.0267

  MES_0900_E1_RR2.0:
    ALL:     N= 7374, avgR=-0.227, WR=32.1%, Sharpe=-3.17
    GATED:   N= 5921, avgR=-0.203, WR=32.5%, Sharpe=-2.78
    SKIPPED: N= 1453, avgR=-0.326, WR=30.5%
    Sharpe lift: +0.391
    Skipped vs Gated: delta=-0.122R, p=0.0001

  MES_1000_E0_RR2.0:
    ALL:     N= 2791, avgR=-0.029, WR=41.3%, Sharpe=-0.40
    GATED:   N= 2293, avgR=+0.009, WR=42.5%, Sharpe=+0.12
    SKIPPED: N=  498, avgR=-0.205, WR=35.9%
    Sharpe lift: +0.515
    Skipped vs Gated: delta=-0.214R, p=0.0001

  MES_1000_E1_RR2.0:
    ALL:     N= 7790, avgR=-0.152, WR=33.9%, Sharpe=-2.03
    GATED:   N= 6371, avgR=-0.133, WR=34.4%, Sharpe=-1.75
    SKIPPED: N= 1419, avgR=-0.241, WR=31.4%
    Sharpe lift: +0.282
    Skipped vs Gated: delta=-0.108R, p=0.0013

======================================================================
PART 2: YEAR-BY-YEAR SKIPPED-DAY TOXICITY
======================================================================

  MNQ_0900_E0_RR2.0 skipped-day R by year (3/3 negative):
    2024: N= 69, avgR=-0.136, WR=36.2% [-]
    2025: N=114, avgR=-0.354, WR=24.6% [-]
    2026: N=  3, avgR=-1.000, WR=0.0% [-]

  MNQ_0900_E1_RR2.0 skipped-day R by year (3/3 negative):
    2024: N=175, avgR=-0.190, WR=32.0% [-]
    2025: N=272, avgR=-0.160, WR=30.9% [-]
    2026: N=  5, avgR=-1.000, WR=0.0% [-]

  MNQ_1000_E0_RR2.0 skipped-day R by year (2/2 negative):
    2024: N= 89, avgR=-0.272, WR=28.1% [-]
    2025: N=110, avgR=-0.355, WR=24.5% [-]

  MNQ_1000_E1_RR2.0 skipped-day R by year (2/2 negative):
    2024: N=213, avgR=-0.283, WR=26.8% [-]
    2025: N=265, avgR=-0.461, WR=19.6% [-]

  MGC_0900_E0_RR2.0 skipped-day R by year (6/6 negative):
    2020: N= 61, avgR=-0.420, WR=29.5% [-]
    2021: N= 74, avgR=-0.804, WR=12.2% [-]
    2022: N= 49, avgR=-0.415, WR=36.7% [-]
    2023: N= 66, avgR=-0.564, WR=25.8% [-]
    2024: N= 51, avgR=-0.699, WR=15.7% [-]
    2025: N= 68, avgR=-0.196, WR=39.7% [-]

  MGC_0900_E1_RR2.0 skipped-day R by year (6/6 negative):
    2020: N=216, avgR=-0.486, WR=25.9% [-]
    2021: N=214, avgR=-0.751, WR=13.6% [-]
    2022: N=162, avgR=-0.542, WR=29.0% [-]
    2023: N=173, avgR=-0.509, WR=27.2% [-]
    2024: N=158, avgR=-0.741, WR=15.2% [-]
    2025: N=215, avgR=-0.250, WR=34.0% [-]

  MGC_1000_E0_RR2.0 skipped-day R by year (6/6 negative):
    2020: N= 78, avgR=-0.353, WR=35.9% [-]
    2021: N= 54, avgR=-0.257, WR=46.3% [-]
    2022: N= 48, avgR=-0.473, WR=31.2% [-]
    2023: N= 57, avgR=-0.375, WR=38.6% [-]
    2024: N= 66, avgR=-0.417, WR=36.4% [-]
    2025: N= 76, avgR=-0.125, WR=39.5% [-]

  MGC_1000_E1_RR2.0 skipped-day R by year (6/6 negative):
    2020: N=231, avgR=-0.526, WR=25.5% [-]
    2021: N=185, avgR=-0.331, WR=37.3% [-]
    2022: N=164, avgR=-0.669, WR=17.1% [-]
    2023: N=183, avgR=-0.280, WR=42.1% [-]
    2024: N=215, avgR=-0.386, WR=32.1% [-]
    2025: N=217, avgR=-0.168, WR=35.0% [-]

  MES_0900_E0_RR2.0 skipped-day R by year (5/7 negative):
    2020: N=111, avgR=+0.094, WR=54.1% [+]
    2021: N= 91, avgR=-0.108, WR=48.4% [-]
    2022: N= 77, avgR=-0.216, WR=37.7% [-]
    2023: N= 79, avgR=-0.542, WR=27.8% [-]
    2024: N= 81, avgR=-0.344, WR=34.6% [-]
    2025: N= 80, avgR=-0.455, WR=25.0% [-]
    2026: N=  5, avgR=+0.521, WR=60.0% [+]

  MES_0900_E1_RR2.0 skipped-day R by year (6/7 negative):
    2020: N=302, avgR=-0.036, WR=42.1% [-]
    2021: N=265, avgR=-0.230, WR=35.5% [-]
    2022: N=213, avgR=-0.489, WR=22.1% [-]
    2023: N=220, avgR=-0.568, WR=22.3% [-]
    2024: N=223, avgR=-0.380, WR=29.6% [-]
    2025: N=215, avgR=-0.461, WR=23.3% [-]
    2026: N= 15, avgR=+0.774, WR=66.7% [+]

  MES_1000_E0_RR2.0 skipped-day R by year (7/7 negative):
    2020: N=113, avgR=-0.041, WR=39.8% [-]
    2021: N= 84, avgR=-0.144, WR=39.3% [-]
    2022: N= 62, avgR=-0.054, WR=40.3% [-]
    2023: N= 58, avgR=-0.087, WR=48.3% [-]
    2024: N= 88, avgR=-0.353, WR=31.8% [-]
    2025: N= 90, avgR=-0.503, WR=21.1% [-]
    2026: N=  3, avgR=-0.200, WR=33.3% [-]

  MES_1000_E1_RR2.0 skipped-day R by year (7/7 negative):
    2020: N=315, avgR=-0.008, WR=39.4% [-]
    2021: N=235, avgR=-0.206, WR=33.6% [-]
    2022: N=187, avgR=-0.154, WR=33.2% [-]
    2023: N=200, avgR=-0.145, WR=39.0% [-]
    2024: N=235, avgR=-0.430, WR=24.7% [-]
    2025: N=237, avgR=-0.517, WR=19.0% [-]
    2026: N= 10, avgR=-1.000, WR=0.0% [-]

======================================================================
PART 3: SIMPLER GATE — JUST 'Contracting' (no compression check)
======================================================================

  Tests whether the compression tier check adds value
  or if 'Contracting' alone is sufficient.

  MNQ_0900_E0:
    ALL Contracting:  N= 229, avgR=-0.102
    Contr+Neut/Comp:  N= 186, avgR=-0.284
    Contr+Expanded:   N=  43, avgR=+0.684
    Simple gate Sharpe: +0.98

  MNQ_0900_E1:
    ALL Contracting:  N= 577, avgR=-0.129
    Contr+Neut/Comp:  N= 452, avgR=-0.181
    Contr+Expanded:   N= 125, avgR=+0.055
    Simple gate Sharpe: -0.70

  MNQ_1000_E0:
    ALL Contracting:  N= 257, avgR=-0.298
    Contr+Neut/Comp:  N= 199, avgR=-0.318
    Contr+Expanded:   N=  58, avgR=-0.232
    Simple gate Sharpe: +2.51

  MNQ_1000_E1:
    ALL Contracting:  N= 617, avgR=-0.271
    Contr+Neut/Comp:  N= 478, avgR=-0.381
    Contr+Expanded:   N= 139, avgR=+0.108
    Simple gate Sharpe: +0.81

  MGC_0900_E0:
    ALL Contracting:  N= 487, avgR=-0.437
    Contr+Neut/Comp:  N= 369, avgR=-0.519
    Contr+Expanded:   N= 118, avgR=-0.178
    Simple gate Sharpe: -5.17

  MGC_0900_E1:
    ALL Contracting:  N=1530, avgR=-0.474
    Contr+Neut/Comp:  N=1138, avgR=-0.538
    Contr+Expanded:   N= 392, avgR=-0.288
    Simple gate Sharpe: -5.63

  MGC_1000_E0:
    ALL Contracting:  N= 520, avgR=-0.295
    Contr+Neut/Comp:  N= 379, avgR=-0.323
    Contr+Expanded:   N= 141, avgR=-0.220
    Simple gate Sharpe: -4.05

  MGC_1000_E1:
    ALL Contracting:  N=1613, avgR=-0.380
    Contr+Neut/Comp:  N=1195, avgR=-0.388
    Contr+Expanded:   N= 418, avgR=-0.359
    Simple gate Sharpe: -3.94

  MES_0900_E0:
    ALL Contracting:  N= 698, avgR=-0.233
    Contr+Neut/Comp:  N= 524, avgR=-0.230
    Contr+Expanded:   N= 174, avgR=-0.241
    Simple gate Sharpe: -1.53

  MES_0900_E1:
    ALL Contracting:  N=1919, avgR=-0.310
    Contr+Neut/Comp:  N=1453, avgR=-0.326
    Contr+Expanded:   N= 466, avgR=-0.262
    Simple gate Sharpe: -2.71

  MES_1000_E0:
    ALL Contracting:  N= 722, avgR=-0.126
    Contr+Neut/Comp:  N= 498, avgR=-0.205
    Contr+Expanded:   N= 224, avgR=+0.048
    Simple gate Sharpe: +0.06

  MES_1000_E1:
    ALL Contracting:  N=2023, avgR=-0.151
    Contr+Neut/Comp:  N=1419, avgR=-0.241
    Contr+Expanded:   N= 604, avgR=+0.060
    Simple gate Sharpe: -2.03

======================================================================
PART 4: RR SENSITIVITY (does the gate work at RR1.0 too?)
======================================================================

  MNQ_0900_E0_RR1: skip N=186, avgR_skip=-0.194, Sharpe_all=+0.55, Sharpe_gated=+1.61, lift=+1.058
  MNQ_1000_E0_RR1: skip N=199, avgR_skip=-0.179, Sharpe_all=+1.43, Sharpe_gated=+2.63, lift=+1.196
  MGC_0900_E0_RR1: skip N=369, avgR_skip=-0.437, Sharpe_all=-7.65, Sharpe_gated=-6.91, lift=+0.739
  MGC_1000_E0_RR1: skip N=379, avgR_skip=-0.332, Sharpe_all=-5.94, Sharpe_gated=-5.46, lift=+0.481
  MES_0900_E0_RR1: skip N=524, avgR_skip=-0.169, Sharpe_all=-2.14, Sharpe_gated=-1.76, lift=+0.384
  MES_1000_E0_RR1: skip N=498, avgR_skip=-0.141, Sharpe_all=-0.79, Sharpe_gated=-0.33, lift=+0.454
  MNQ_0900_E0_RR2: skip N=186, avgR_skip=-0.284, Sharpe_all=+0.42, Sharpe_gated=+1.40, lift=+0.981
  MNQ_1000_E0_RR2: skip N=199, avgR_skip=-0.318, Sharpe_all=+0.97, Sharpe_gated=+2.13, lift=+1.162
  MGC_0900_E0_RR2: skip N=369, avgR_skip=-0.519, Sharpe_all=-5.71, Sharpe_gated=-5.00, lift=+0.712
  MGC_1000_E0_RR2: skip N=379, avgR_skip=-0.323, Sharpe_all=-4.25, Sharpe_gated=-4.00, lift=+0.255
  MES_0900_E0_RR2: skip N=524, avgR_skip=-0.230, Sharpe_all=-2.03, Sharpe_gated=-1.68, lift=+0.345
  MES_1000_E0_RR2: skip N=498, avgR_skip=-0.205, Sharpe_all=-0.40, Sharpe_gated=+0.12, lift=+0.515

======================================================================
BH FDR CORRECTION
======================================================================

  Total tests: 12
  BH survivors (q=0.10): 11
    MNQ_1000_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_skip=478, delta=-0.453R, WR_skip=22.8% vs WR_gate=38.3%, Sharpe_lift=+1.011, TOXIC (skip helps)
    MGC_0900_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_skip=1138, delta=-0.198R, WR_skip=24.3% vs WR_gate=32.4%, Sharpe_lift=+0.632, TOXIC (skip helps)
    MNQ_1000_E0_RR2.0: raw_p=0.0000, p_bh=0.0000, N_skip=199, delta=-0.500R, WR_skip=26.1% vs WR_gate=43.3%, Sharpe_lift=+1.162, TOXIC (skip helps)
    MGC_0900_E0_RR2.0: raw_p=0.0000, p_bh=0.0000, N_skip=369, delta=-0.224R, WR_skip=26.3% vs WR_gate=35.3%, Sharpe_lift=+0.712, TOXIC (skip helps)
    MGC_1000_E1_RR2.0: raw_p=0.0000, p_bh=0.0000, N_skip=1195, delta=-0.127R, WR_skip=31.6% vs WR_gate=35.2%, Sharpe_lift=+0.369, TOXIC (skip helps)
    MNQ_0900_E0_RR2.0: raw_p=0.0001, p_bh=0.0001, N_skip=186, delta=-0.402R, WR_skip=28.5% vs WR_gate=41.3%, Sharpe_lift=+0.981, TOXIC (skip helps)
    MES_0900_E1_RR2.0: raw_p=0.0001, p_bh=0.0001, N_skip=1453, delta=-0.122R, WR_skip=30.5% vs WR_gate=32.5%, Sharpe_lift=+0.391, TOXIC (skip helps)
    MES_1000_E0_RR2.0: raw_p=0.0001, p_bh=0.0001, N_skip=498, delta=-0.214R, WR_skip=35.9% vs WR_gate=42.5%, Sharpe_lift=+0.515, TOXIC (skip helps)
    MES_1000_E1_RR2.0: raw_p=0.0013, p_bh=0.0017, N_skip=1419, delta=-0.108R, WR_skip=31.4% vs WR_gate=34.4%, Sharpe_lift=+0.282, TOXIC (skip helps)
    MES_0900_E0_RR2.0: raw_p=0.0267, p_bh=0.0320, N_skip=524, delta=-0.110R, WR_skip=39.3% vs WR_gate=38.9%, Sharpe_lift=+0.345, TOXIC (skip helps)
    MNQ_0900_E1_RR2.0: raw_p=0.0463, p_bh=0.0505, N_skip=452, delta=-0.130R, WR_skip=31.0% vs WR_gate=34.3%, Sharpe_lift=+0.310, TOXIC (skip helps)

======================================================================
HONEST SUMMARY
======================================================================

### Claim: 'ATR velocity gate improves Sharpe by 0.2-0.3'
  See Part 1 for actual Sharpe lifts per instrument/session.

### Key questions answered:
  1. How many days does the gate remove? -> Part 0
  2. Actual Sharpe improvement? -> Part 1
  3. Year-by-year consistency of skipped toxicity? -> Part 2
  4. Does compression check add value? -> Part 3
  5. Does it work at RR1.0? -> Part 4
  6. BH FDR survived? -> BH section

### CAVEATS
  - ATR velocity gate is already live in paper_trader.py
  - MNQ only ~2 years of data
  - Gate was designed from same data (IS/OOS split not available)
  - Sharpe lift depends on base strategy performance