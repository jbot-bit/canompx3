# Overnight PDH/PDL Take Signal Validation
**Date:** 2026-02-21
**Script:** research/research_pdh_pdl_signal.py

======================================================================
PART 0: DATA AVAILABILITY CHECK
======================================================================

  daily_features prev columns: ['prev_day_close', 'prev_day_direction', 'prev_day_high', 'prev_day_low', 'prev_day_range']
  daily_features daily columns: ['daily_close', 'daily_high', 'daily_low', 'daily_open']
  bars_1m row counts: [('M6E', 1745492), ('M2K', 1715959), ('SIL', 691315), ('MNQ', 706272), ('MCL', 1591965), ('MGC', 3537841), ('MES', 2467504)]

======================================================================
PART 1: PDH/PDL TAKE VIA GAP (gap_open vs prev_day_high/low)
======================================================================

  Method: Use daily_features columns + LAG() for prev_day_high/low.
  PDH_taken: today's daily_open > prev_day_high (gapped above PDH)
  PDL_taken: today's daily_open < prev_day_low (gapped below PDL)
  Neither: daily_open between prev_day_low and prev_day_high
  Note: This only captures GAP-based PDH/PDL take, not intra-overnight take.

  MNQ_0900_E0_RR2.0:
       PDH_taken: N=   27, avgR=+0.914, WR=70.4%
       PDL_taken: N=    9, avgR=+1.308, WR=77.8%
         Neither: N=  865, avgR=-0.004, WR=37.3%
    PDH vs rest: delta=+0.905R, WR_delta=+32.6pp, p=0.0011

  MNQ_0900_E1_RR2.0:
       PDH_taken: N=   65, avgR=+0.289, WR=46.2%
       PDL_taken: N=   15, avgR=+0.979, WR=66.7%
         Neither: N= 2309, avgR=-0.092, WR=33.1%
    PDH vs rest: delta=+0.374R, WR_delta=+12.8pp, p=0.0377
    PDL vs rest: delta=+1.061R, WR_delta=+33.2pp, p=0.0133

  MNQ_1000_E0_RR2.0:
       PDH_taken: N=   27, avgR=+0.316, WR=48.1%
       PDL_taken: N=    7, avgR=+1.013, WR=71.4%
         Neither: N=  962, avgR=+0.072, WR=39.6%
    PDH vs rest: delta=+0.237R, WR_delta=+8.3pp, p=0.3904

  MNQ_1000_E1_RR2.0:
       PDH_taken: N=   68, avgR=-0.070, WR=33.8%
       PDL_taken: N=   25, avgR=+0.718, WR=60.0%
         Neither: N= 2440, avgR=-0.015, WR=35.4%
    PDH vs rest: delta=-0.062R, WR_delta=-1.8pp, p=0.7039
    PDL vs rest: delta=+0.735R, WR_delta=+24.7pp, p=0.0173

  MGC_0900_E0_RR2.0:
       PDH_taken: N=   50, avgR=+0.188, WR=54.0%
       PDL_taken: N=    6, avgR=-1.000, WR=0.0%
         Neither: N= 3831, avgR=-0.423, WR=28.9%
    PDH vs rest: delta=+0.612R, WR_delta=+25.2pp, p=0.0005

  MGC_0900_E1_RR2.0:
       PDH_taken: N=  156, avgR=-0.234, WR=34.0%
       PDL_taken: N=   20, avgR=-1.000, WR=0.0%
         Neither: N=11962, avgR=-0.442, WR=28.8%
    PDH vs rest: delta=+0.209R, WR_delta=+5.3pp, p=0.0213
    PDL vs rest: delta=-0.561R, WR_delta=-28.8pp, p=0.0000

  MGC_1000_E0_RR2.0:
       PDH_taken: N=   67, avgR=-0.127, WR=44.8%
       PDL_taken: N=   11, avgR=-0.383, WR=27.3%
         Neither: N= 4108, avgR=-0.327, WR=36.7%
    PDH vs rest: delta=+0.200R, WR_delta=+8.1pp, p=0.0968
    PDL vs rest: delta=-0.060R, WR_delta=-9.5pp, p=0.8567

  MGC_1000_E1_RR2.0:
       PDH_taken: N=  207, avgR=-0.180, WR=40.1%
       PDL_taken: N=   25, avgR=-0.068, WR=40.0%
         Neither: N=12390, avgR=-0.356, WR=33.0%
    PDH vs rest: delta=+0.175R, WR_delta=+7.1pp, p=0.0139
    PDL vs rest: delta=+0.285R, WR_delta=+6.9pp, p=0.2366

  MES_0900_E0_RR2.0:
       PDH_taken: N=   37, avgR=+0.613, WR=62.2%
       PDL_taken: N=   14, avgR=+0.744, WR=64.3%
         Neither: N= 3036, avgR=-0.165, WR=39.1%
    PDH vs rest: delta=+0.773R, WR_delta=+23.0pp, p=0.0008
    PDL vs rest: delta=+0.900R, WR_delta=+24.9pp, p=0.0278

  MES_0900_E1_RR2.0:
       PDH_taken: N=   87, avgR=+0.535, WR=57.5%
       PDL_taken: N=   37, avgR=+0.797, WR=64.9%
         Neither: N= 8335, avgR=-0.257, WR=31.6%
    PDH vs rest: delta=+0.787R, WR_delta=+25.8pp, p=0.0000
    PDL vs rest: delta=+1.046R, WR_delta=+33.0pp, p=0.0000

  MES_1000_E0_RR2.0:
       PDH_taken: N=   46, avgR=+0.088, WR=45.7%
       PDL_taken: N=   25, avgR=+0.080, WR=44.0%
         Neither: N= 3144, avgR=-0.049, WR=41.0%
    PDH vs rest: delta=+0.136R, WR_delta=+4.6pp, p=0.4584
    PDL vs rest: delta=+0.127R, WR_delta=+2.9pp, p=0.6199

  MES_1000_E1_RR2.0:
       PDH_taken: N=  112, avgR=-0.209, WR=30.4%
       PDL_taken: N=   57, avgR=+0.117, WR=43.9%
         Neither: N= 8761, avgR=-0.169, WR=33.5%
    PDH vs rest: delta=-0.041R, WR_delta=-3.2pp, p=0.7216
    PDL vs rest: delta=+0.287R, WR_delta=+10.4pp, p=0.0975

======================================================================
PART 2: YEAR-BY-YEAR (MNQ 0900, PDH_taken)
======================================================================

  MNQ 0900 E0 RR2.0 PDH_taken (N=27):
    2024: N=   8, avgR=+1.336, WR=87.5% [+]
    2025: N=  19, avgR=+0.737, WR=63.2% [+]

  MNQ 0900 E1 RR2.0 PDH_taken (N=65):
    2024: N=  25, avgR=+0.551, WR=56.0% [+]
    2025: N=  40, avgR=+0.125, WR=40.0% [+]

======================================================================
BH FDR CORRECTION
======================================================================

  Total tests: 21
  BH survivors (q=0.10): 12
    MGC_0900_E1_RR2.0_PDL: raw_p=0.0000, p_bh=0.0000, N=20, delta=-0.561R, WR_delta=-28.8pp, HURTS
    MES_0900_E1_RR2.0_PDH: raw_p=0.0000, p_bh=0.0000, N=87, delta=+0.787R, WR_delta=+25.8pp, HELPS
    MES_0900_E1_RR2.0_PDL: raw_p=0.0000, p_bh=0.0002, N=37, delta=+1.046R, WR_delta=+33.0pp, HELPS
    MGC_0900_E0_RR2.0_PDH: raw_p=0.0005, p_bh=0.0025, N=50, delta=+0.612R, WR_delta=+25.2pp, HELPS
    MES_0900_E0_RR2.0_PDH: raw_p=0.0008, p_bh=0.0034, N=37, delta=+0.773R, WR_delta=+23.0pp, HELPS
    MNQ_0900_E0_RR2.0_PDH: raw_p=0.0011, p_bh=0.0039, N=27, delta=+0.905R, WR_delta=+32.6pp, HELPS
    MNQ_0900_E1_RR2.0_PDL: raw_p=0.0133, p_bh=0.0365, N=15, delta=+1.061R, WR_delta=+33.2pp, HELPS
    MGC_1000_E1_RR2.0_PDH: raw_p=0.0139, p_bh=0.0365, N=207, delta=+0.175R, WR_delta=+7.1pp, HELPS
    MNQ_1000_E1_RR2.0_PDL: raw_p=0.0173, p_bh=0.0403, N=25, delta=+0.735R, WR_delta=+24.7pp, HELPS
    MGC_0900_E1_RR2.0_PDH: raw_p=0.0213, p_bh=0.0447, N=156, delta=+0.209R, WR_delta=+5.3pp, HELPS
    MES_0900_E0_RR2.0_PDL: raw_p=0.0278, p_bh=0.0531, N=14, delta=+0.900R, WR_delta=+24.9pp, HELPS
    MNQ_0900_E1_RR2.0_PDH: raw_p=0.0377, p_bh=0.0660, N=65, delta=+0.374R, WR_delta=+12.8pp, HELPS

======================================================================
HONEST SUMMARY
======================================================================

### CAVEATS
  - 'PDH taken' here means GAPPED above prev day high (daily_open > prev_high)
  - Does NOT capture intra-overnight PDH take (would need bars_1m analysis)
  - Gap-above-PDH is a relatively rare event, so N may be small
  - MNQ only ~2 years of data
  - Not DST-split