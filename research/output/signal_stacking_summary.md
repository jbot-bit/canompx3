# Signal Stacking Research — Honest Summary

**Date:** 2026-02-20
**Sessions:** 1000, 0900, 0030
**Instruments:** MES, MGC, MNQ
**Reference:** E3 / CB=1 / RR=2.0
**Period:** 2016-02-01 00:00:00 to 2026-02-11 00:00:00
**Outcomes loaded:** 4,850 (E3/CB1/RR2.0)
**In-sample only.** No walk-forward validation.

## SURVIVED SCRUTINY

- MES 0030 summer G4 L4 (size+dir+conc+vol): avgR=+0.418 N=136 p_bh=0.0232 [PRELIMINARY]
- MES 0030 summer G5 L4 (size+dir+conc+vol): avgR=+0.411 N=124 p_bh=0.0275 [PRELIMINARY]

## DID NOT SURVIVE

- 216 total (instrument × session × g_filter × dst × stack_level) cells tested
- 184/216 had positive avg_r
- 42 had raw p<0.05 (before correction)
- 2 survived BH FDR correction at p<0.05

## CAVEATS

1. **In-sample only.** No walk-forward or OOS validation.
2. **Fixed reference params** (E3/CB1/RR2.0). Stack lift may differ at E1 or other RR targets.
3. **Small N at full stack.** G6+/G8+ + all filters may drop below REGIME threshold (N<30).
4. **Concordance requires all instruments trading.** On thin days, missing one instrument falsely downgrades tier.
5. **Vol filter is fail-closed.** Days with missing bars_1m data are excluded — may skew toward liquid days.
6. **DST split (0900/0030) reduces N further** — winter/summer cells may be INVALID.

## NEXT STEPS

- If any stack survives BH correction: test at E1 entry (production entry model for 1000)
- If conviction_score > 1.5 at N>=30: add to position sizing overlay (not new strategy)
- If concordance consistently adds lift: wire into execution_engine as a position-size multiplier
- If nothing survives: signals are already captured in G-filter; stop stacking

## Analysis 1: Incremental Layer Lift

### MES 0030

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|  winter |  G4 |    L0 |                 size |   160 |   1.00 | -0.0447 |  0.34 | -0.535 | 0.6703 | 0.8834 | PRELIMINARY |
|  winter |  G4 |    L1 |             size+dir |   160 |   1.00 | -0.0447 |  0.34 | -0.535 | 0.6703 | 0.8834 | PRELIMINARY |
|  winter |  G4 |    L2 |            size+conc |   126 |   0.79 | +0.0134 |  0.37 |  0.158 | 0.9111 | 0.9111 | PRELIMINARY |
|  winter |  G4 |    L3 |        size+dir+conc |   126 |   0.79 | +0.0134 |  0.37 |  0.158 | 0.9111 | 0.9111 | PRELIMINARY |
|  winter |  G4 |    L4 |    size+dir+conc+vol |    76 |   0.47 | +0.1071 |  0.39 |  1.231 | 0.5010 | 0.8293 |      REGIME |
|  winter |  G4 |    L5 | size+dir+conc+vol+cal |    69 |   0.43 | +0.0577 |  0.38 |  0.668 | 0.7277 | 0.8834 |      REGIME |
|  winter |  G5 |    L0 |                 size |   157 |   1.00 | -0.0425 |  0.34 | -0.509 | 0.6887 | 0.8834 | PRELIMINARY |
|  winter |  G5 |    L1 |             size+dir |   157 |   1.00 | -0.0425 |  0.34 | -0.509 | 0.6887 | 0.8834 | PRELIMINARY |
|  winter |  G5 |    L2 |            size+conc |   123 |   0.78 | +0.0176 |  0.37 |  0.207 | 0.8852 | 0.9111 | PRELIMINARY |
|  winter |  G5 |    L3 |        size+dir+conc |   123 |   0.78 | +0.0176 |  0.37 |  0.207 | 0.8852 | 0.9111 | PRELIMINARY |
|  winter |  G5 |    L4 |    size+dir+conc+vol |    76 |   0.48 | +0.1071 |  0.39 |  1.231 | 0.5010 | 0.8293 |      REGIME |
|  winter |  G5 |    L5 | size+dir+conc+vol+cal |    69 |   0.44 | +0.0577 |  0.38 |  0.668 | 0.7277 | 0.8834 |      REGIME |
|  winter |  G6 |    L0 |                 size |   139 |   1.00 | -0.0135 |  0.35 | -0.160 | 0.9057 | 0.9111 | PRELIMINARY |
|  winter |  G6 |    L1 |             size+dir |   139 |   1.00 | -0.0135 |  0.35 | -0.160 | 0.9057 | 0.9111 | PRELIMINARY |
|  winter |  G6 |    L2 |            size+conc |   108 |   0.78 | +0.0366 |  0.37 |  0.428 | 0.7798 | 0.8912 | PRELIMINARY |
|  winter |  G6 |    L3 |        size+dir+conc |   108 |   0.78 | +0.0366 |  0.37 |  0.428 | 0.7798 | 0.8912 | PRELIMINARY |
|  winter |  G6 |    L4 |    size+dir+conc+vol |    71 |   0.51 | +0.1481 |  0.41 |  1.688 | 0.3732 | 0.8143 |      REGIME |
|  winter |  G6 |    L5 | size+dir+conc+vol+cal |    64 |   0.46 | +0.0993 |  0.39 |  1.139 | 0.5681 | 0.8522 |      REGIME |
|  winter |  G8 |    L0 |                 size |   100 |   1.00 | +0.0464 |  0.37 |  0.536 | 0.7361 | 0.8834 | PRELIMINARY |
|  winter |  G8 |    L1 |             size+dir |   100 |   1.00 | +0.0464 |  0.37 |  0.536 | 0.7361 | 0.8834 | PRELIMINARY |
|  winter |  G8 |    L2 |            size+conc |    79 |   0.79 | +0.1088 |  0.39 |  1.244 | 0.4882 | 0.8293 |      REGIME |
|  winter |  G8 |    L3 |        size+dir+conc |    79 |   0.79 | +0.1088 |  0.39 |  1.244 | 0.4882 | 0.8293 |      REGIME |
|  winter |  G8 |    L4 |    size+dir+conc+vol |    58 |   0.58 | +0.2201 |  0.43 |  2.470 | 0.2409 | 0.6086 |      REGIME |
|  winter |  G8 |    L5 | size+dir+conc+vol+cal |    52 |   0.52 | +0.1982 |  0.42 |  2.227 | 0.3165 | 0.7235 |      REGIME |
|  summer |  G4 |    L0 |                 size |   263 |   1.00 | +0.1812 |  0.44 |  2.140 | 0.0297 | 0.1188 |        CORE |
|  summer |  G4 |    L1 |             size+dir |   263 |   1.00 | +0.1812 |  0.44 |  2.140 | 0.0297 | 0.1188 |        CORE |
|  summer |  G4 |    L2 |            size+conc |   222 |   0.84 | +0.2197 |  0.45 |  2.580 | 0.0162 | 0.0975 |        CORE |
|  summer |  G4 |    L3 |        size+dir+conc |   222 |   0.84 | +0.2197 |  0.45 |  2.580 | 0.0162 | 0.0975 |        CORE |
|  summer |  G4 |    L4 |    size+dir+conc+vol |   136 |   0.52 | +0.4183 |  0.52 |  4.869 | 0.0005 | 0.0232 | PRELIMINARY |
|  summer |  G4 |    L5 | size+dir+conc+vol+cal |   121 |   0.46 | +0.3682 |  0.50 |  4.284 | 0.0036 | 0.0578 | PRELIMINARY |
|  summer |  G5 |    L0 |                 size |   220 |   1.00 | +0.2020 |  0.44 |  2.362 | 0.0284 | 0.1188 |        CORE |
|  summer |  G5 |    L1 |             size+dir |   220 |   1.00 | +0.2020 |  0.44 |  2.362 | 0.0284 | 0.1188 |        CORE |
|  summer |  G5 |    L2 |            size+conc |   189 |   0.86 | +0.2425 |  0.46 |  2.822 | 0.0155 | 0.0975 | PRELIMINARY |
|  summer |  G5 |    L3 |        size+dir+conc |   189 |   0.86 | +0.2425 |  0.46 |  2.822 | 0.0155 | 0.0975 | PRELIMINARY |
|  summer |  G5 |    L4 |    size+dir+conc+vol |   124 |   0.56 | +0.4105 |  0.52 |  4.748 | 0.0011 | 0.0275 | PRELIMINARY |
|  summer |  G5 |    L5 | size+dir+conc+vol+cal |   110 |   0.50 | +0.3647 |  0.50 |  4.219 | 0.0063 | 0.0752 | PRELIMINARY |
|  summer |  G6 |    L0 |                 size |   173 |   1.00 | +0.1334 |  0.41 |  1.553 | 0.1998 | 0.5329 | PRELIMINARY |
|  summer |  G6 |    L1 |             size+dir |   173 |   1.00 | +0.1334 |  0.41 |  1.553 | 0.1998 | 0.5329 | PRELIMINARY |
|  summer |  G6 |    L2 |            size+conc |   152 |   0.88 | +0.1989 |  0.43 |  2.298 | 0.0763 | 0.2443 | PRELIMINARY |
|  summer |  G6 |    L3 |        size+dir+conc |   152 |   0.88 | +0.1989 |  0.43 |  2.298 | 0.0763 | 0.2443 | PRELIMINARY |
|  summer |  G6 |    L4 |    size+dir+conc+vol |   103 |   0.60 | +0.2893 |  0.47 |  3.309 | 0.0368 | 0.1359 | PRELIMINARY |
|  summer |  G6 |    L5 | size+dir+conc+vol+cal |    91 |   0.53 | +0.2448 |  0.45 |  2.809 | 0.0948 | 0.2845 |      REGIME |
|  summer |  G8 |    L0 |                 size |   104 |   1.00 | +0.0804 |  0.38 |  0.929 | 0.5518 | 0.8522 | PRELIMINARY |
|  summer |  G8 |    L1 |             size+dir |   104 |   1.00 | +0.0804 |  0.38 |  0.929 | 0.5518 | 0.8522 | PRELIMINARY |
|  summer |  G8 |    L2 |            size+conc |    93 |   0.89 | +0.1170 |  0.40 |  1.344 | 0.4163 | 0.8293 |      REGIME |
|  summer |  G8 |    L3 |        size+dir+conc |    93 |   0.89 | +0.1170 |  0.40 |  1.344 | 0.4163 | 0.8293 |      REGIME |
|  summer |  G8 |    L4 |    size+dir+conc+vol |    67 |   0.64 | +0.1749 |  0.42 |  1.987 | 0.3093 | 0.7235 |      REGIME |
|  summer |  G8 |    L5 | size+dir+conc+vol+cal |    57 |   0.55 | +0.1340 |  0.40 |  1.529 | 0.4702 | 0.8293 |      REGIME |

### MNQ 0030

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|  winter |  G4 |    L0 |                 size |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G4 |    L1 |             size+dir |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G4 |    L2 |            size+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G4 |    L3 |        size+dir+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G4 |    L4 |    size+dir+conc+vol |    82 |   0.53 | +0.3583 |  0.46 |  3.867 | 0.0302 | 0.1168 |      REGIME |
|  winter |  G4 |    L5 | size+dir+conc+vol+cal |    72 |   0.47 | +0.3441 |  0.46 |  3.712 | 0.0511 | 0.1168 |      REGIME |
|  winter |  G5 |    L0 |                 size |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G5 |    L1 |             size+dir |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G5 |    L2 |            size+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G5 |    L3 |        size+dir+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G5 |    L4 |    size+dir+conc+vol |    82 |   0.53 | +0.3583 |  0.46 |  3.867 | 0.0302 | 0.1168 |      REGIME |
|  winter |  G5 |    L5 | size+dir+conc+vol+cal |    72 |   0.47 | +0.3441 |  0.46 |  3.712 | 0.0511 | 0.1168 |      REGIME |
|  winter |  G6 |    L0 |                 size |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G6 |    L1 |             size+dir |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G6 |    L2 |            size+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G6 |    L3 |        size+dir+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G6 |    L4 |    size+dir+conc+vol |    82 |   0.53 | +0.3583 |  0.46 |  3.867 | 0.0302 | 0.1168 |      REGIME |
|  winter |  G6 |    L5 | size+dir+conc+vol+cal |    72 |   0.47 | +0.3441 |  0.46 |  3.712 | 0.0511 | 0.1168 |      REGIME |
|  winter |  G8 |    L0 |                 size |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G8 |    L1 |             size+dir |   154 |   1.00 | +0.1399 |  0.39 |  1.551 | 0.2271 | 0.2726 | PRELIMINARY |
|  winter |  G8 |    L2 |            size+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G8 |    L3 |        size+dir+conc |   120 |   0.78 | +0.2442 |  0.42 |  2.668 | 0.0681 | 0.1168 | PRELIMINARY |
|  winter |  G8 |    L4 |    size+dir+conc+vol |    82 |   0.53 | +0.3583 |  0.46 |  3.867 | 0.0302 | 0.1168 |      REGIME |
|  winter |  G8 |    L5 | size+dir+conc+vol+cal |    72 |   0.47 | +0.3441 |  0.46 |  3.712 | 0.0511 | 0.1168 |      REGIME |
|  summer |  G4 |    L0 |                 size |   314 |   1.00 | +0.0401 |  0.36 |  0.461 | 0.6070 | 0.6070 |        CORE |
|  summer |  G4 |    L1 |             size+dir |   314 |   1.00 | +0.0401 |  0.36 |  0.461 | 0.6070 | 0.6070 |        CORE |
|  summer |  G4 |    L2 |            size+conc |   258 |   0.82 | +0.2105 |  0.42 |  2.356 | 0.0179 | 0.1073 |        CORE |
|  summer |  G4 |    L3 |        size+dir+conc |   258 |   0.82 | +0.2105 |  0.42 |  2.356 | 0.0179 | 0.1073 |        CORE |
|  summer |  G4 |    L4 |    size+dir+conc+vol |   138 |   0.44 | +0.2527 |  0.43 |  2.798 | 0.0403 | 0.1168 | PRELIMINARY |
|  summer |  G4 |    L5 | size+dir+conc+vol+cal |   124 |   0.39 | +0.1852 |  0.41 |  2.064 | 0.1502 | 0.2252 | PRELIMINARY |
|  summer |  G5 |    L0 |                 size |   314 |   1.00 | +0.0401 |  0.36 |  0.461 | 0.6070 | 0.6070 |        CORE |
|  summer |  G5 |    L1 |             size+dir |   314 |   1.00 | +0.0401 |  0.36 |  0.461 | 0.6070 | 0.6070 |        CORE |
|  summer |  G5 |    L2 |            size+conc |   258 |   0.82 | +0.2105 |  0.42 |  2.356 | 0.0179 | 0.1073 |        CORE |
|  summer |  G5 |    L3 |        size+dir+conc |   258 |   0.82 | +0.2105 |  0.42 |  2.356 | 0.0179 | 0.1073 |        CORE |
|  summer |  G5 |    L4 |    size+dir+conc+vol |   138 |   0.44 | +0.2527 |  0.43 |  2.798 | 0.0403 | 0.1168 | PRELIMINARY |
|  summer |  G5 |    L5 | size+dir+conc+vol+cal |   124 |   0.39 | +0.1852 |  0.41 |  2.064 | 0.1502 | 0.2252 | PRELIMINARY |
|  summer |  G6 |    L0 |                 size |   313 |   1.00 | +0.0434 |  0.36 |  0.499 | 0.5784 | 0.6070 |        CORE |
|  summer |  G6 |    L1 |             size+dir |   313 |   1.00 | +0.0434 |  0.36 |  0.499 | 0.5784 | 0.6070 |        CORE |
|  summer |  G6 |    L2 |            size+conc |   257 |   0.82 | +0.2152 |  0.42 |  2.407 | 0.0158 | 0.1073 |        CORE |
|  summer |  G6 |    L3 |        size+dir+conc |   257 |   0.82 | +0.2152 |  0.42 |  2.407 | 0.0158 | 0.1073 |        CORE |
|  summer |  G6 |    L4 |    size+dir+conc+vol |   138 |   0.44 | +0.2527 |  0.43 |  2.798 | 0.0403 | 0.1168 | PRELIMINARY |
|  summer |  G6 |    L5 | size+dir+conc+vol+cal |   124 |   0.40 | +0.1852 |  0.41 |  2.064 | 0.1502 | 0.2252 | PRELIMINARY |
|  summer |  G8 |    L0 |                 size |   312 |   1.00 | +0.0468 |  0.37 |  0.537 | 0.5504 | 0.6070 |        CORE |
|  summer |  G8 |    L1 |             size+dir |   312 |   1.00 | +0.0468 |  0.37 |  0.537 | 0.5504 | 0.6070 |        CORE |
|  summer |  G8 |    L2 |            size+conc |   256 |   0.82 | +0.2200 |  0.43 |  2.459 | 0.0139 | 0.1073 |        CORE |
|  summer |  G8 |    L3 |        size+dir+conc |   256 |   0.82 | +0.2200 |  0.43 |  2.459 | 0.0139 | 0.1073 |        CORE |
|  summer |  G8 |    L4 |    size+dir+conc+vol |   138 |   0.44 | +0.2527 |  0.43 |  2.798 | 0.0403 | 0.1168 | PRELIMINARY |
|  summer |  G8 |    L5 | size+dir+conc+vol+cal |   124 |   0.40 | +0.1852 |  0.41 |  2.064 | 0.1502 | 0.2252 | PRELIMINARY |

### MGC 0900

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|  winter |  G4 |    L0 |                 size |    69 |   1.00 | +0.3583 |  0.42 |  4.780 | 0.0148 | 0.1382 |      REGIME |
|  winter |  G4 |    L1 |             size+dir |    69 |   1.00 | +0.3583 |  0.42 |  4.780 | 0.0148 | 0.1382 |      REGIME |
|  winter |  G4 |    L2 |            size+conc |    69 |   1.00 | +0.3583 |  0.42 |  4.780 | 0.0148 | 0.1382 |      REGIME |
|  winter |  G4 |    L3 |        size+dir+conc |    69 |   1.00 | +0.3583 |  0.42 |  4.780 | 0.0148 | 0.1382 |      REGIME |
|  winter |  G4 |    L4 |    size+dir+conc+vol |    55 |   0.80 | +0.3612 |  0.42 |  4.787 | 0.0295 | 0.1415 |      REGIME |
|  winter |  G4 |    L5 | size+dir+conc+vol+cal |    54 |   0.78 | +0.3695 |  0.43 |  4.858 | 0.0287 | 0.1415 |      REGIME |
|  winter |  G5 |    L0 |                 size |    58 |   1.00 | +0.3685 |  0.41 |  4.869 | 0.0230 | 0.1382 |      REGIME |
|  winter |  G5 |    L1 |             size+dir |    58 |   1.00 | +0.3685 |  0.41 |  4.869 | 0.0230 | 0.1382 |      REGIME |
|  winter |  G5 |    L2 |            size+conc |    58 |   1.00 | +0.3685 |  0.41 |  4.869 | 0.0230 | 0.1382 |      REGIME |
|  winter |  G5 |    L3 |        size+dir+conc |    58 |   1.00 | +0.3685 |  0.41 |  4.869 | 0.0230 | 0.1382 |      REGIME |
|  winter |  G5 |    L4 |    size+dir+conc+vol |    48 |   0.83 | +0.3432 |  0.40 |  4.544 | 0.0532 | 0.2129 |      REGIME |
|  winter |  G5 |    L5 | size+dir+conc+vol+cal |    47 |   0.81 | +0.3524 |  0.40 |  4.622 | 0.0519 | 0.2129 |      REGIME |
|  winter |  G6 |    L0 |                 size |    46 |   1.00 | +0.2935 |  0.37 |  3.875 | 0.1048 | 0.2515 |      REGIME |
|  winter |  G6 |    L1 |             size+dir |    46 |   1.00 | +0.2935 |  0.37 |  3.875 | 0.1048 | 0.2515 |      REGIME |
|  winter |  G6 |    L2 |            size+conc |    46 |   1.00 | +0.2935 |  0.37 |  3.875 | 0.1048 | 0.2515 |      REGIME |
|  winter |  G6 |    L3 |        size+dir+conc |    46 |   1.00 | +0.2935 |  0.37 |  3.875 | 0.1048 | 0.2515 |      REGIME |
|  winter |  G6 |    L4 |    size+dir+conc+vol |    41 |   0.89 | +0.2986 |  0.37 |  3.958 | 0.1183 | 0.2580 |      REGIME |
|  winter |  G6 |    L5 | size+dir+conc+vol+cal |    40 |   0.87 | +0.3082 |  0.38 |  4.040 | 0.1156 | 0.2580 |      REGIME |
|  winter |  G8 |    L0 |                 size |    33 |   1.00 | +0.3487 |  0.36 |  4.680 | 0.1001 | 0.2515 |      REGIME |
|  winter |  G8 |    L1 |             size+dir |    33 |   1.00 | +0.3487 |  0.36 |  4.680 | 0.1001 | 0.2515 |      REGIME |
|  winter |  G8 |    L2 |            size+conc |    33 |   1.00 | +0.3487 |  0.36 |  4.680 | 0.1001 | 0.2515 |      REGIME |
|  winter |  G8 |    L3 |        size+dir+conc |    33 |   1.00 | +0.3487 |  0.36 |  4.680 | 0.1001 | 0.2515 |      REGIME |
|  winter |  G8 |    L4 |    size+dir+conc+vol |    31 |   0.94 | +0.3199 |  0.35 |  4.272 | 0.1445 | 0.2889 |      REGIME |
|  winter |  G8 |    L5 | size+dir+conc+vol+cal |    30 |   0.91 | +0.3335 |  0.37 |  4.388 | 0.1409 | 0.2889 |      REGIME |
|  summer |  G4 |    L0 |                 size |    51 |   1.00 | +0.0423 |  0.33 |  0.572 | 0.7981 | 0.8150 |      REGIME |
|  summer |  G4 |    L1 |             size+dir |    51 |   1.00 | +0.0423 |  0.33 |  0.572 | 0.7981 | 0.8150 |      REGIME |
|  summer |  G4 |    L2 |            size+conc |    51 |   1.00 | +0.0423 |  0.33 |  0.572 | 0.7981 | 0.8150 |      REGIME |
|  summer |  G4 |    L3 |        size+dir+conc |    51 |   1.00 | +0.0423 |  0.33 |  0.572 | 0.7981 | 0.8150 |      REGIME |
|  summer |  G4 |    L4 |    size+dir+conc+vol |    46 |   0.90 | -0.0140 |  0.30 | -0.194 | 0.9344 | 0.9344 |      REGIME |
|  summer |  G4 |    L5 | size+dir+conc+vol+cal |    44 |   0.86 | -0.0936 |  0.27 | -1.342 | 0.5779 | 0.7299 |      REGIME |
|  summer |  G5 |    L0 |                 size |    36 |   1.00 | +0.2537 |  0.42 |  3.265 | 0.2254 | 0.3865 |      REGIME |
|  summer |  G5 |    L1 |             size+dir |    36 |   1.00 | +0.2537 |  0.42 |  3.265 | 0.2254 | 0.3865 |      REGIME |
|  summer |  G5 |    L2 |            size+conc |    36 |   1.00 | +0.2537 |  0.42 |  3.265 | 0.2254 | 0.3865 |      REGIME |
|  summer |  G5 |    L3 |        size+dir+conc |    36 |   1.00 | +0.2537 |  0.42 |  3.265 | 0.2254 | 0.3865 |      REGIME |
|  summer |  G5 |    L4 |    size+dir+conc+vol |    34 |   0.94 | +0.1710 |  0.38 |  2.226 | 0.4194 | 0.6778 |      REGIME |
|  summer |  G5 |    L5 | size+dir+conc+vol+cal |    32 |   0.89 | +0.0730 |  0.34 |  0.975 | 0.7307 | 0.8150 |      REGIME |
|  summer |  G6 |    L0 |                 size |    23 |   1.00 | +0.1448 |  0.35 |  1.900 | 0.5718 | 0.7299 |     INVALID |
|  summer |  G6 |    L1 |             size+dir |    23 |   1.00 | +0.1448 |  0.35 |  1.900 | 0.5718 | 0.7299 |     INVALID |
|  summer |  G6 |    L2 |            size+conc |    23 |   1.00 | +0.1448 |  0.35 |  1.900 | 0.5718 | 0.7299 |     INVALID |
|  summer |  G6 |    L3 |        size+dir+conc |    23 |   1.00 | +0.1448 |  0.35 |  1.900 | 0.5718 | 0.7299 |     INVALID |
|  summer |  G6 |    L4 |    size+dir+conc+vol |    22 |   0.96 | +0.0721 |  0.32 |  0.965 | 0.7783 | 0.8150 |     INVALID |
|  summer |  G6 |    L5 | size+dir+conc+vol+cal |    20 |   0.87 | -0.0945 |  0.25 | -1.352 | 0.7076 | 0.8150 |     INVALID |
|  summer |  G8 |    L0 |                 size |    11 |   1.00 | +0.3158 |  0.45 |  3.628 | 0.4660 | 0.6778 |     INVALID |
|  summer |  G8 |    L1 |             size+dir |    11 |   1.00 | +0.3158 |  0.45 |  3.628 | 0.4660 | 0.6778 |     INVALID |
|  summer |  G8 |    L2 |            size+conc |    11 |   1.00 | +0.3158 |  0.45 |  3.628 | 0.4660 | 0.6778 |     INVALID |
|  summer |  G8 |    L3 |        size+dir+conc |    11 |   1.00 | +0.3158 |  0.45 |  3.628 | 0.4660 | 0.6778 |     INVALID |
|  summer |  G8 |    L4 |    size+dir+conc+vol |    10 |   0.91 | +0.1730 |  0.40 |  2.007 | 0.6986 | 0.8150 |     INVALID |
|  summer |  G8 |    L5 | size+dir+conc+vol+cal |     8 |   0.73 | -0.2183 |  0.25 | -2.799 | 0.6333 | 0.7795 |     INVALID |

### MES 1000

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|     all |  G4 |    L0 |                 size |   170 |   1.00 | +0.0586 |  0.34 |  0.791 | 0.5169 | 0.9471 | PRELIMINARY |
|     all |  G4 |    L1 |             size+dir |    87 |   0.51 | +0.0351 |  0.31 |  0.490 | 0.7740 | 0.9471 |      REGIME |
|     all |  G4 |    L2 |            size+conc |   166 |   0.98 | +0.0514 |  0.34 |  0.696 | 0.5729 | 0.9471 | PRELIMINARY |
|     all |  G4 |    L3 |        size+dir+conc |    86 |   0.51 | +0.0157 |  0.30 |  0.221 | 0.8975 | 0.9471 |      REGIME |
|     all |  G4 |    L4 |    size+dir+conc+vol |    73 |   0.43 | +0.0582 |  0.32 |  0.808 | 0.6649 | 0.9471 |      REGIME |
|     all |  G4 |    L5 | size+dir+conc+vol+cal |    66 |   0.39 | +0.1103 |  0.33 |  1.520 | 0.4393 | 0.9471 |      REGIME |
|     all |  G5 |    L0 |                 size |   116 |   1.00 | +0.0248 |  0.31 |  0.338 | 0.8190 | 0.9471 | PRELIMINARY |
|     all |  G5 |    L1 |             size+dir |    64 |   0.55 | +0.0358 |  0.30 |  0.502 | 0.8009 | 0.9471 |      REGIME |
|     all |  G5 |    L2 |            size+conc |   112 |   0.97 | +0.0128 |  0.30 |  0.176 | 0.9070 | 0.9471 | PRELIMINARY |
|     all |  G5 |    L3 |        size+dir+conc |    63 |   0.54 | +0.0094 |  0.29 |  0.133 | 0.9471 | 0.9471 |      REGIME |
|     all |  G5 |    L4 |    size+dir+conc+vol |    57 |   0.49 | +0.0829 |  0.32 |  1.144 | 0.5887 | 0.9471 |      REGIME |
|     all |  G5 |    L5 | size+dir+conc+vol+cal |    50 |   0.43 | +0.1552 |  0.34 |  2.122 | 0.3492 | 0.9471 |      REGIME |
|     all |  G6 |    L0 |                 size |    82 |   1.00 | +0.0272 |  0.30 |  0.367 | 0.8346 | 0.9471 |      REGIME |
|     all |  G6 |    L1 |             size+dir |    42 |   0.51 | +0.1282 |  0.33 |  1.710 | 0.4892 | 0.9471 |      REGIME |
|     all |  G6 |    L2 |            size+conc |    78 |   0.95 | +0.0100 |  0.29 |  0.137 | 0.9393 | 0.9471 |      REGIME |
|     all |  G6 |    L3 |        size+dir+conc |    41 |   0.50 | +0.0898 |  0.32 |  1.210 | 0.6282 | 0.9471 |      REGIME |
|     all |  G6 |    L4 |    size+dir+conc+vol |    38 |   0.46 | +0.1573 |  0.34 |  2.090 | 0.4222 | 0.9471 |      REGIME |
|     all |  G6 |    L5 | size+dir+conc+vol+cal |    32 |   0.39 | +0.2503 |  0.38 |  3.282 | 0.2511 | 0.9471 |      REGIME |
|     all |  G8 |    L0 |                 size |    40 |   1.00 | -0.0529 |  0.28 | -0.710 | 0.7787 | 0.9471 |      REGIME |
|     all |  G8 |    L1 |             size+dir |    19 |   0.47 | +0.0938 |  0.32 |  1.218 | 0.7418 | 0.9471 |     INVALID |
|     all |  G8 |    L2 |            size+conc |    37 |   0.93 | -0.0503 |  0.27 | -0.681 | 0.7958 | 0.9471 |      REGIME |
|     all |  G8 |    L3 |        size+dir+conc |    19 |   0.47 | +0.0938 |  0.32 |  1.218 | 0.7418 | 0.9471 |     INVALID |
|     all |  G8 |    L4 |    size+dir+conc+vol |    19 |   0.47 | +0.0938 |  0.32 |  1.218 | 0.7418 | 0.9471 |     INVALID |
|     all |  G8 |    L5 | size+dir+conc+vol+cal |    17 |   0.42 | +0.1914 |  0.35 |  2.422 | 0.5382 | 0.9471 |     INVALID |

### MGC 1000

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|     all |  G4 |    L0 |                 size |   136 |   1.00 | +0.0296 |  0.35 |  0.389 | 0.7753 | 0.8746 | PRELIMINARY |
|     all |  G4 |    L1 |             size+dir |    73 |   0.54 | +0.0376 |  0.36 |  0.486 | 0.7944 | 0.8746 |      REGIME |
|     all |  G4 |    L2 |            size+conc |   119 |   0.88 | +0.0101 |  0.34 |  0.132 | 0.9280 | 0.9280 | PRELIMINARY |
|     all |  G4 |    L3 |        size+dir+conc |    60 |   0.44 | +0.1009 |  0.38 |  1.264 | 0.5397 | 0.8346 |      REGIME |
|     all |  G4 |    L4 |    size+dir+conc+vol |    50 |   0.37 | +0.2051 |  0.42 |  2.539 | 0.2635 | 0.7905 |      REGIME |
|     all |  G4 |    L5 | size+dir+conc+vol+cal |    48 |   0.35 | +0.2554 |  0.44 |  3.156 | 0.1750 | 0.6269 |      REGIME |
|     all |  G5 |    L0 |                 size |    91 |   1.00 | +0.0908 |  0.36 |  1.175 | 0.4818 | 0.8259 |      REGIME |
|     all |  G5 |    L1 |             size+dir |    50 |   0.55 | +0.2850 |  0.44 |  3.561 | 0.1192 | 0.6269 |      REGIME |
|     all |  G5 |    L2 |            size+conc |    85 |   0.93 | +0.1224 |  0.38 |  1.567 | 0.3653 | 0.7971 |      REGIME |
|     all |  G5 |    L3 |        size+dir+conc |    45 |   0.49 | +0.3991 |  0.49 |  4.927 | 0.0432 | 0.3456 |      REGIME |
|     all |  G5 |    L4 |    size+dir+conc+vol |    37 |   0.41 | +0.5450 |  0.54 |  6.797 | 0.0133 | 0.1595 |      REGIME |
|     all |  G5 |    L5 | size+dir+conc+vol+cal |    37 |   0.41 | +0.5450 |  0.54 |  6.797 | 0.0133 | 0.1595 |      REGIME |
|     all |  G6 |    L0 |                 size |    63 |   1.00 | -0.0542 |  0.30 | -0.709 | 0.7241 | 0.8746 |      REGIME |
|     all |  G6 |    L1 |             size+dir |    35 |   0.56 | +0.1181 |  0.37 |  1.455 | 0.5912 | 0.8346 |      REGIME |
|     all |  G6 |    L2 |            size+conc |    61 |   0.97 | -0.0336 |  0.31 | -0.435 | 0.8313 | 0.8746 |      REGIME |
|     all |  G6 |    L3 |        size+dir+conc |    33 |   0.52 | +0.1666 |  0.39 |  2.020 | 0.4701 | 0.8259 |      REGIME |
|     all |  G6 |    L4 |    size+dir+conc+vol |    26 |   0.41 | +0.3581 |  0.46 |  4.265 | 0.1828 | 0.6269 |     INVALID |
|     all |  G6 |    L5 | size+dir+conc+vol+cal |    26 |   0.41 | +0.3581 |  0.46 |  4.265 | 0.1828 | 0.6269 |     INVALID |
|     all |  G8 |    L0 |                 size |    34 |   1.00 | -0.1609 |  0.26 | -2.098 | 0.4463 | 0.8259 |      REGIME |
|     all |  G8 |    L1 |             size+dir |    18 |   0.53 | +0.0638 |  0.33 |  0.776 | 0.8382 | 0.8746 |     INVALID |
|     all |  G8 |    L2 |            size+conc |    32 |   0.94 | -0.1283 |  0.28 | -1.634 | 0.5645 | 0.8346 |      REGIME |
|     all |  G8 |    L3 |        size+dir+conc |    16 |   0.47 | +0.1572 |  0.38 |  1.842 | 0.6491 | 0.8655 |     INVALID |
|     all |  G8 |    L4 |    size+dir+conc+vol |    13 |   0.38 | +0.3868 |  0.46 |  4.360 | 0.3416 | 0.7971 |     INVALID |
|     all |  G8 |    L5 | size+dir+conc+vol+cal |    13 |   0.38 | +0.3868 |  0.46 |  4.360 | 0.3416 | 0.7971 |     INVALID |

### MNQ 1000

|     dst |   G | Level |                Label |     N |  %base |    avgR |    WR | Sharpe |   p_raw |    p_bh |       Class |
|----------------------------------------------------------------------------------------------------------------------------|
|     all |  G4 |    L0 |                 size |   501 |   1.00 | -0.0643 |  0.30 | -0.852 | 0.2303 | 0.9210 |        CORE |
|     all |  G4 |    L1 |             size+dir |   229 |   0.46 | -0.0029 |  0.33 | -0.037 | 0.9719 | 0.9719 |        CORE |
|     all |  G4 |    L2 |            size+conc |   496 |   0.99 | -0.0672 |  0.30 | -0.890 | 0.2126 | 0.9210 |        CORE |
|     all |  G4 |    L3 |        size+dir+conc |   224 |   0.45 | -0.0078 |  0.33 | -0.101 | 0.9241 | 0.9642 |        CORE |
|     all |  G4 |    L4 |    size+dir+conc+vol |   147 |   0.29 | +0.0563 |  0.35 |  0.711 | 0.5877 | 0.9385 | PRELIMINARY |
|     all |  G4 |    L5 | size+dir+conc+vol+cal |   138 |   0.28 | +0.0787 |  0.36 |  0.990 | 0.4652 | 0.9385 | PRELIMINARY |
|     all |  G5 |    L0 |                 size |   496 |   1.00 | -0.0655 |  0.30 | -0.866 | 0.2250 | 0.9210 |        CORE |
|     all |  G5 |    L1 |             size+dir |   227 |   0.46 | -0.0143 |  0.32 | -0.186 | 0.8603 | 0.9385 |        CORE |
|     all |  G5 |    L2 |            size+conc |   491 |   0.99 | -0.0684 |  0.30 | -0.904 | 0.2076 | 0.9210 |        CORE |
|     all |  G5 |    L3 |        size+dir+conc |   222 |   0.45 | -0.0196 |  0.32 | -0.254 | 0.8121 | 0.9385 |        CORE |
|     all |  G5 |    L4 |    size+dir+conc+vol |   146 |   0.29 | +0.0478 |  0.34 |  0.604 | 0.6465 | 0.9385 | PRELIMINARY |
|     all |  G5 |    L5 | size+dir+conc+vol+cal |   137 |   0.28 | +0.0698 |  0.35 |  0.877 | 0.5187 | 0.9385 | PRELIMINARY |
|     all |  G6 |    L0 |                 size |   480 |   1.00 | -0.0696 |  0.30 | -0.920 | 0.2048 | 0.9210 |        CORE |
|     all |  G6 |    L1 |             size+dir |   221 |   0.46 | -0.0201 |  0.32 | -0.260 | 0.8081 | 0.9385 |        CORE |
|     all |  G6 |    L2 |            size+conc |   475 |   0.99 | -0.0726 |  0.29 | -0.960 | 0.1882 | 0.9210 |        CORE |
|     all |  G6 |    L3 |        size+dir+conc |   216 |   0.45 | -0.0256 |  0.31 | -0.331 | 0.7595 | 0.9385 |        CORE |
|     all |  G6 |    L4 |    size+dir+conc+vol |   141 |   0.29 | +0.0340 |  0.33 |  0.430 | 0.7483 | 0.9385 | PRELIMINARY |
|     all |  G6 |    L5 | size+dir+conc+vol+cal |   132 |   0.28 | +0.0560 |  0.34 |  0.703 | 0.6120 | 0.9385 | PRELIMINARY |
|     all |  G8 |    L0 |                 size |   432 |   1.00 | -0.0476 |  0.30 | -0.626 | 0.4126 | 0.9385 |        CORE |
|     all |  G8 |    L1 |             size+dir |   205 |   0.47 | -0.0178 |  0.31 | -0.230 | 0.8357 | 0.9385 |        CORE |
|     all |  G8 |    L2 |            size+conc |   429 |   0.99 | -0.0488 |  0.30 | -0.641 | 0.4033 | 0.9385 |        CORE |
|     all |  G8 |    L3 |        size+dir+conc |   202 |   0.47 | -0.0198 |  0.31 | -0.256 | 0.8190 | 0.9385 |        CORE |
|     all |  G8 |    L4 |    size+dir+conc+vol |   134 |   0.31 | +0.0322 |  0.33 |  0.406 | 0.7676 | 0.9385 | PRELIMINARY |
|     all |  G8 |    L5 | size+dir+conc+vol+cal |   125 |   0.29 | +0.0552 |  0.34 |  0.692 | 0.6267 | 0.9385 | PRELIMINARY |


## Analysis 2: Conviction Profile

Rows shown: CORE/REGIME/PRELIMINARY with avg_r > 0

| inst | sess | dst | G | level | label | N | freq/yr | avgR | conviction | sizing |
|------|------|-----|---|-------|-------|---|---------|------|------------|--------|
| MGC | 1000 | all | G4 | L0 | size | 136 | 0.5 | +0.0296 | 1.000 | 1x |
| MGC | 1000 | all | G4 | L1 | size+dir | 73 | 0.3 | +0.0376 | 1.273 | 1x |
| MGC | 1000 | all | G4 | L2 | size+conc | 119 | 0.5 | +0.0101 | 0.340 | 1x |
| MGC | 1000 | all | G4 | L3 | size+dir+conc | 60 | 0.2 | +0.1009 | 3.414 | 3x |
| MGC | 1000 | all | G4 | L4 | size+dir+conc+vol | 50 | 0.2 | +0.2051 | 6.942 | 3x |
| MGC | 1000 | all | G4 | L5 | size+dir+conc+vol+cal | 48 | 0.2 | +0.2554 | 8.641 | 3x |
| MGC | 1000 | all | G5 | L0 | size | 91 | 0.4 | +0.0908 | 1.000 | 1x |
| MGC | 1000 | all | G5 | L1 | size+dir | 50 | 0.2 | +0.2850 | 3.139 | 3x |
| MGC | 1000 | all | G5 | L2 | size+conc | 85 | 0.3 | +0.1224 | 1.348 | 1x |
| MGC | 1000 | all | G5 | L3 | size+dir+conc | 45 | 0.2 | +0.3991 | 4.395 | 3x |
| MGC | 1000 | all | G5 | L4 | size+dir+conc+vol | 37 | 0.1 | +0.5450 | 6.001 | 3x |
| MGC | 1000 | all | G5 | L5 | size+dir+conc+vol+cal | 37 | 0.1 | +0.5450 | 6.001 | 3x |
| MGC | 1000 | all | G6 | L1 | size+dir | 35 | 0.1 | +0.1181 | -- | 1x |
| MGC | 1000 | all | G6 | L3 | size+dir+conc | 33 | 0.1 | +0.1666 | -- | 1x |
| MES | 1000 | all | G4 | L0 | size | 170 | 0.7 | +0.0586 | 1.000 | 1x |
| MES | 1000 | all | G4 | L1 | size+dir | 87 | 0.3 | +0.0351 | 0.598 | 1x |
| MES | 1000 | all | G4 | L2 | size+conc | 166 | 0.7 | +0.0514 | 0.876 | 1x |
| MES | 1000 | all | G4 | L3 | size+dir+conc | 86 | 0.3 | +0.0157 | 0.268 | 1x |
| MES | 1000 | all | G4 | L4 | size+dir+conc+vol | 73 | 0.3 | +0.0582 | 0.992 | 1x |
| MES | 1000 | all | G4 | L5 | size+dir+conc+vol+cal | 66 | 0.3 | +0.1103 | 1.881 | 2x |
| MES | 1000 | all | G5 | L0 | size | 116 | 0.5 | +0.0248 | 1.000 | 1x |
| MES | 1000 | all | G5 | L1 | size+dir | 64 | 0.3 | +0.0358 | 1.447 | 1x |
| MES | 1000 | all | G5 | L2 | size+conc | 112 | 0.4 | +0.0128 | 0.515 | 1x |
| MES | 1000 | all | G5 | L3 | size+dir+conc | 63 | 0.3 | +0.0094 | 0.380 | 1x |
| MES | 1000 | all | G5 | L4 | size+dir+conc+vol | 57 | 0.2 | +0.0829 | 3.346 | 3x |
| MES | 1000 | all | G5 | L5 | size+dir+conc+vol+cal | 50 | 0.2 | +0.1552 | 6.263 | 3x |
| MES | 1000 | all | G6 | L0 | size | 82 | 0.3 | +0.0272 | 1.000 | 1x |
| MES | 1000 | all | G6 | L1 | size+dir | 42 | 0.2 | +0.1282 | 4.718 | 3x |
| MES | 1000 | all | G6 | L2 | size+conc | 78 | 0.3 | +0.0100 | 0.370 | 1x |
| MES | 1000 | all | G6 | L3 | size+dir+conc | 41 | 0.2 | +0.0898 | 3.306 | 3x |
| MES | 1000 | all | G6 | L4 | size+dir+conc+vol | 38 | 0.2 | +0.1573 | 5.791 | 3x |
| MES | 1000 | all | G6 | L5 | size+dir+conc+vol+cal | 32 | 0.1 | +0.2503 | 9.215 | 3x |
| MNQ | 1000 | all | G4 | L4 | size+dir+conc+vol | 147 | 0.6 | +0.0563 | -- | 1x |
| MNQ | 1000 | all | G4 | L5 | size+dir+conc+vol+cal | 138 | 0.6 | +0.0787 | -- | 1x |
| MNQ | 1000 | all | G5 | L4 | size+dir+conc+vol | 146 | 0.6 | +0.0478 | -- | 1x |
| MNQ | 1000 | all | G5 | L5 | size+dir+conc+vol+cal | 137 | 0.5 | +0.0698 | -- | 1x |
| MNQ | 1000 | all | G6 | L4 | size+dir+conc+vol | 141 | 0.6 | +0.0340 | -- | 1x |
| MNQ | 1000 | all | G6 | L5 | size+dir+conc+vol+cal | 132 | 0.5 | +0.0560 | -- | 1x |
| MNQ | 1000 | all | G8 | L4 | size+dir+conc+vol | 134 | 0.5 | +0.0322 | -- | 1x |
| MNQ | 1000 | all | G8 | L5 | size+dir+conc+vol+cal | 125 | 0.5 | +0.0552 | -- | 1x |
| MGC | 0900 | winter | G4 | L0 | size | 69 | 0.3 | +0.3583 | 1.000 | 1x |
| MGC | 0900 | winter | G4 | L1 | size+dir | 69 | 0.3 | +0.3583 | 1.000 | 1x |
| MGC | 0900 | winter | G4 | L2 | size+conc | 69 | 0.3 | +0.3583 | 1.000 | 1x |
| MGC | 0900 | winter | G4 | L3 | size+dir+conc | 69 | 0.3 | +0.3583 | 1.000 | 1x |
| MGC | 0900 | winter | G4 | L4 | size+dir+conc+vol | 55 | 0.2 | +0.3612 | 1.008 | 1x |
| MGC | 0900 | winter | G4 | L5 | size+dir+conc+vol+cal | 54 | 0.2 | +0.3695 | 1.031 | 1x |
| MGC | 0900 | winter | G5 | L0 | size | 58 | 0.2 | +0.3685 | 1.000 | 1x |
| MGC | 0900 | winter | G5 | L1 | size+dir | 58 | 0.2 | +0.3685 | 1.000 | 1x |
| MGC | 0900 | winter | G5 | L2 | size+conc | 58 | 0.2 | +0.3685 | 1.000 | 1x |
| MGC | 0900 | winter | G5 | L3 | size+dir+conc | 58 | 0.2 | +0.3685 | 1.000 | 1x |
| MGC | 0900 | winter | G5 | L4 | size+dir+conc+vol | 48 | 0.2 | +0.3432 | 0.932 | 1x |
| MGC | 0900 | winter | G5 | L5 | size+dir+conc+vol+cal | 47 | 0.2 | +0.3524 | 0.956 | 1x |
| MGC | 0900 | winter | G6 | L0 | size | 46 | 0.2 | +0.2935 | 1.000 | 1x |
| MGC | 0900 | winter | G6 | L1 | size+dir | 46 | 0.2 | +0.2935 | 1.000 | 1x |
| MGC | 0900 | winter | G6 | L2 | size+conc | 46 | 0.2 | +0.2935 | 1.000 | 1x |
| MGC | 0900 | winter | G6 | L3 | size+dir+conc | 46 | 0.2 | +0.2935 | 1.000 | 1x |
| MGC | 0900 | winter | G6 | L4 | size+dir+conc+vol | 41 | 0.2 | +0.2986 | 1.017 | 1x |
| MGC | 0900 | winter | G6 | L5 | size+dir+conc+vol+cal | 40 | 0.2 | +0.3082 | 1.050 | 1x |
| MGC | 0900 | winter | G8 | L0 | size | 33 | 0.1 | +0.3487 | 1.000 | 1x |
| MGC | 0900 | winter | G8 | L1 | size+dir | 33 | 0.1 | +0.3487 | 1.000 | 1x |
| MGC | 0900 | winter | G8 | L2 | size+conc | 33 | 0.1 | +0.3487 | 1.000 | 1x |
| MGC | 0900 | winter | G8 | L3 | size+dir+conc | 33 | 0.1 | +0.3487 | 1.000 | 1x |
| MGC | 0900 | winter | G8 | L4 | size+dir+conc+vol | 31 | 0.1 | +0.3199 | 0.917 | 1x |
| MGC | 0900 | winter | G8 | L5 | size+dir+conc+vol+cal | 30 | 0.1 | +0.3335 | 0.956 | 1x |
| MGC | 0900 | summer | G4 | L0 | size | 51 | 0.2 | +0.0423 | 1.000 | 1x |
| MGC | 0900 | summer | G4 | L1 | size+dir | 51 | 0.2 | +0.0423 | 1.000 | 1x |
| MGC | 0900 | summer | G4 | L2 | size+conc | 51 | 0.2 | +0.0423 | 1.000 | 1x |
| MGC | 0900 | summer | G4 | L3 | size+dir+conc | 51 | 0.2 | +0.0423 | 1.000 | 1x |
| MGC | 0900 | summer | G5 | L0 | size | 36 | 0.1 | +0.2537 | 1.000 | 1x |
| MGC | 0900 | summer | G5 | L1 | size+dir | 36 | 0.1 | +0.2537 | 1.000 | 1x |
| MGC | 0900 | summer | G5 | L2 | size+conc | 36 | 0.1 | +0.2537 | 1.000 | 1x |
| MGC | 0900 | summer | G5 | L3 | size+dir+conc | 36 | 0.1 | +0.2537 | 1.000 | 1x |
| MGC | 0900 | summer | G5 | L4 | size+dir+conc+vol | 34 | 0.1 | +0.1710 | 0.674 | 1x |
| MGC | 0900 | summer | G5 | L5 | size+dir+conc+vol+cal | 32 | 0.1 | +0.0730 | 0.288 | 1x |
| MES | 0030 | winter | G4 | L2 | size+conc | 126 | 0.5 | +0.0134 | -- | 1x |
| MES | 0030 | winter | G4 | L3 | size+dir+conc | 126 | 0.5 | +0.0134 | -- | 1x |
| MES | 0030 | winter | G4 | L4 | size+dir+conc+vol | 76 | 0.3 | +0.1071 | -- | 1x |
| MES | 0030 | winter | G4 | L5 | size+dir+conc+vol+cal | 69 | 0.3 | +0.0577 | -- | 1x |
| MES | 0030 | winter | G5 | L2 | size+conc | 123 | 0.5 | +0.0176 | -- | 1x |
| MES | 0030 | winter | G5 | L3 | size+dir+conc | 123 | 0.5 | +0.0176 | -- | 1x |
| MES | 0030 | winter | G5 | L4 | size+dir+conc+vol | 76 | 0.3 | +0.1071 | -- | 1x |
| MES | 0030 | winter | G5 | L5 | size+dir+conc+vol+cal | 69 | 0.3 | +0.0577 | -- | 1x |
| MES | 0030 | winter | G6 | L2 | size+conc | 108 | 0.4 | +0.0366 | -- | 1x |
| MES | 0030 | winter | G6 | L3 | size+dir+conc | 108 | 0.4 | +0.0366 | -- | 1x |
| MES | 0030 | winter | G6 | L4 | size+dir+conc+vol | 71 | 0.3 | +0.1481 | -- | 1x |
| MES | 0030 | winter | G6 | L5 | size+dir+conc+vol+cal | 64 | 0.3 | +0.0993 | -- | 1x |
| MES | 0030 | winter | G8 | L0 | size | 100 | 0.4 | +0.0464 | 1.000 | 1x |
| MES | 0030 | winter | G8 | L1 | size+dir | 100 | 0.4 | +0.0464 | 1.000 | 1x |
| MES | 0030 | winter | G8 | L2 | size+conc | 79 | 0.3 | +0.1088 | 2.346 | 2x |
| MES | 0030 | winter | G8 | L3 | size+dir+conc | 79 | 0.3 | +0.1088 | 2.346 | 2x |
| MES | 0030 | winter | G8 | L4 | size+dir+conc+vol | 58 | 0.2 | +0.2201 | 4.745 | 3x |
| MES | 0030 | winter | G8 | L5 | size+dir+conc+vol+cal | 52 | 0.2 | +0.1982 | 4.274 | 3x |
| MES | 0030 | summer | G4 | L0 | size | 263 | 1.1 | +0.1812 | 1.000 | 1x |
| MES | 0030 | summer | G4 | L1 | size+dir | 263 | 1.1 | +0.1812 | 1.000 | 1x |
| MES | 0030 | summer | G4 | L2 | size+conc | 222 | 0.9 | +0.2197 | 1.213 | 1x |
| MES | 0030 | summer | G4 | L3 | size+dir+conc | 222 | 0.9 | +0.2197 | 1.213 | 1x |
| MES | 0030 | summer | G4 | L4 | size+dir+conc+vol | 136 | 0.5 | +0.4183 | 2.309 | 2x |
| MES | 0030 | summer | G4 | L5 | size+dir+conc+vol+cal | 121 | 0.5 | +0.3682 | 2.032 | 2x |
| MES | 0030 | summer | G5 | L0 | size | 220 | 0.9 | +0.2020 | 1.000 | 1x |
| MES | 0030 | summer | G5 | L1 | size+dir | 220 | 0.9 | +0.2020 | 1.000 | 1x |
| MES | 0030 | summer | G5 | L2 | size+conc | 189 | 0.8 | +0.2425 | 1.201 | 1x |
| MES | 0030 | summer | G5 | L3 | size+dir+conc | 189 | 0.8 | +0.2425 | 1.201 | 1x |
| MES | 0030 | summer | G5 | L4 | size+dir+conc+vol | 124 | 0.5 | +0.4105 | 2.032 | 2x |
| MES | 0030 | summer | G5 | L5 | size+dir+conc+vol+cal | 110 | 0.4 | +0.3647 | 1.806 | 2x |
| MES | 0030 | summer | G6 | L0 | size | 173 | 0.7 | +0.1334 | 1.000 | 1x |
| MES | 0030 | summer | G6 | L1 | size+dir | 173 | 0.7 | +0.1334 | 1.000 | 1x |
| MES | 0030 | summer | G6 | L2 | size+conc | 152 | 0.6 | +0.1989 | 1.491 | 1x |
| MES | 0030 | summer | G6 | L3 | size+dir+conc | 152 | 0.6 | +0.1989 | 1.491 | 1x |
| MES | 0030 | summer | G6 | L4 | size+dir+conc+vol | 103 | 0.4 | +0.2893 | 2.169 | 2x |
| MES | 0030 | summer | G6 | L5 | size+dir+conc+vol+cal | 91 | 0.4 | +0.2448 | 1.835 | 2x |
| MES | 0030 | summer | G8 | L0 | size | 104 | 0.4 | +0.0804 | 1.000 | 1x |
| MES | 0030 | summer | G8 | L1 | size+dir | 104 | 0.4 | +0.0804 | 1.000 | 1x |
| MES | 0030 | summer | G8 | L2 | size+conc | 93 | 0.4 | +0.1170 | 1.455 | 1x |
| MES | 0030 | summer | G8 | L3 | size+dir+conc | 93 | 0.4 | +0.1170 | 1.455 | 1x |
| MES | 0030 | summer | G8 | L4 | size+dir+conc+vol | 67 | 0.3 | +0.1749 | 2.175 | 2x |
| MES | 0030 | summer | G8 | L5 | size+dir+conc+vol+cal | 57 | 0.2 | +0.1340 | 1.666 | 2x |
| MNQ | 0030 | winter | G4 | L0 | size | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G4 | L1 | size+dir | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G4 | L2 | size+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G4 | L3 | size+dir+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G4 | L4 | size+dir+conc+vol | 82 | 0.3 | +0.3583 | 2.561 | 3x |
| MNQ | 0030 | winter | G4 | L5 | size+dir+conc+vol+cal | 72 | 0.3 | +0.3441 | 2.460 | 2x |
| MNQ | 0030 | winter | G5 | L0 | size | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G5 | L1 | size+dir | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G5 | L2 | size+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G5 | L3 | size+dir+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G5 | L4 | size+dir+conc+vol | 82 | 0.3 | +0.3583 | 2.561 | 3x |
| MNQ | 0030 | winter | G5 | L5 | size+dir+conc+vol+cal | 72 | 0.3 | +0.3441 | 2.460 | 2x |
| MNQ | 0030 | winter | G6 | L0 | size | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G6 | L1 | size+dir | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G6 | L2 | size+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G6 | L3 | size+dir+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G6 | L4 | size+dir+conc+vol | 82 | 0.3 | +0.3583 | 2.561 | 3x |
| MNQ | 0030 | winter | G6 | L5 | size+dir+conc+vol+cal | 72 | 0.3 | +0.3441 | 2.460 | 2x |
| MNQ | 0030 | winter | G8 | L0 | size | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G8 | L1 | size+dir | 154 | 0.6 | +0.1399 | 1.000 | 1x |
| MNQ | 0030 | winter | G8 | L2 | size+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G8 | L3 | size+dir+conc | 120 | 0.5 | +0.2442 | 1.746 | 2x |
| MNQ | 0030 | winter | G8 | L4 | size+dir+conc+vol | 82 | 0.3 | +0.3583 | 2.561 | 3x |
| MNQ | 0030 | winter | G8 | L5 | size+dir+conc+vol+cal | 72 | 0.3 | +0.3441 | 2.460 | 2x |
| MNQ | 0030 | summer | G4 | L0 | size | 314 | 1.3 | +0.0401 | 1.000 | 1x |
| MNQ | 0030 | summer | G4 | L1 | size+dir | 314 | 1.3 | +0.0401 | 1.000 | 1x |
| MNQ | 0030 | summer | G4 | L2 | size+conc | 258 | 1.0 | +0.2105 | 5.248 | 3x |
| MNQ | 0030 | summer | G4 | L3 | size+dir+conc | 258 | 1.0 | +0.2105 | 5.248 | 3x |
| MNQ | 0030 | summer | G4 | L4 | size+dir+conc+vol | 138 | 0.6 | +0.2527 | 6.301 | 3x |
| MNQ | 0030 | summer | G4 | L5 | size+dir+conc+vol+cal | 124 | 0.5 | +0.1852 | 4.617 | 3x |
| MNQ | 0030 | summer | G5 | L0 | size | 314 | 1.3 | +0.0401 | 1.000 | 1x |
| MNQ | 0030 | summer | G5 | L1 | size+dir | 314 | 1.3 | +0.0401 | 1.000 | 1x |
| MNQ | 0030 | summer | G5 | L2 | size+conc | 258 | 1.0 | +0.2105 | 5.248 | 3x |
| MNQ | 0030 | summer | G5 | L3 | size+dir+conc | 258 | 1.0 | +0.2105 | 5.248 | 3x |
| MNQ | 0030 | summer | G5 | L4 | size+dir+conc+vol | 138 | 0.6 | +0.2527 | 6.301 | 3x |
| MNQ | 0030 | summer | G5 | L5 | size+dir+conc+vol+cal | 124 | 0.5 | +0.1852 | 4.617 | 3x |
| MNQ | 0030 | summer | G6 | L0 | size | 313 | 1.3 | +0.0434 | 1.000 | 1x |
| MNQ | 0030 | summer | G6 | L1 | size+dir | 313 | 1.3 | +0.0434 | 1.000 | 1x |
| MNQ | 0030 | summer | G6 | L2 | size+conc | 257 | 1.0 | +0.2152 | 4.955 | 3x |
| MNQ | 0030 | summer | G6 | L3 | size+dir+conc | 257 | 1.0 | +0.2152 | 4.955 | 3x |
| MNQ | 0030 | summer | G6 | L4 | size+dir+conc+vol | 138 | 0.6 | +0.2527 | 5.819 | 3x |
| MNQ | 0030 | summer | G6 | L5 | size+dir+conc+vol+cal | 124 | 0.5 | +0.1852 | 4.264 | 3x |
| MNQ | 0030 | summer | G8 | L0 | size | 312 | 1.2 | +0.0468 | 1.000 | 1x |
| MNQ | 0030 | summer | G8 | L1 | size+dir | 312 | 1.2 | +0.0468 | 1.000 | 1x |
| MNQ | 0030 | summer | G8 | L2 | size+conc | 256 | 1.0 | +0.2200 | 4.702 | 3x |
| MNQ | 0030 | summer | G8 | L3 | size+dir+conc | 256 | 1.0 | +0.2200 | 4.702 | 3x |
| MNQ | 0030 | summer | G8 | L4 | size+dir+conc+vol | 138 | 0.6 | +0.2527 | 5.403 | 3x |
| MNQ | 0030 | summer | G8 | L5 | size+dir+conc+vol+cal | 124 | 0.5 | +0.1852 | 3.959 | 3x |

## Analysis 3: Concordance × Size Interaction (MGC 1000)

| G | tier | N | %g_days | avgR | WR | p_raw | p_bh | class |
|---|------|---|---------|------|----|-------|------|-------|
| G4 | concordant_3 | 54 | 0.40 | +0.2016 | 0.43 | 0.2489 | 0.5101 | REGIME |
| G4 | majority_2 | 65 | 0.48 | -0.1491 | 0.28 | 0.2975 | 0.5101 | REGIME |
| G4 | remaining | 17 | 0.12 | +0.1660 | 0.41 | 0.5717 | 0.6860 | INVALID |
| G5 | concordant_3 | 41 | 0.45 | +0.3380 | 0.46 | 0.0973 | 0.5101 | REGIME |
| G5 | majority_2 | 44 | 0.48 | -0.0785 | 0.30 | 0.6628 | 0.7231 | REGIME |
| G5 | remaining | 6 | 0.07 | -0.3568 | 0.17 | 0.4199 | 0.5957 | INVALID |
| G6 | concordant_3 | 31 | 0.49 | +0.1764 | 0.39 | 0.4468 | 0.5957 | REGIME |
| G6 | majority_2 | 30 | 0.48 | -0.2506 | 0.23 | 0.2451 | 0.5101 | REGIME |
| G6 | remaining | 2 | 0.03 | -0.6831 | 0.00 | 0.2765 | 0.5101 | INVALID |
| G8 | concordant_3 | 14 | 0.41 | +0.1160 | 0.36 | 0.7467 | 0.7467 | INVALID |
| G8 | majority_2 | 18 | 0.53 | -0.3183 | 0.22 | 0.2727 | 0.5101 | INVALID |
| G8 | remaining | 2 | 0.06 | -0.6831 | 0.00 | 0.2765 | 0.5101 | INVALID |

## Mandatory Disclosures

- **N trades per stack level:** see Analysis 1 table above
- **Period:** 2016-02-01 00:00:00 to 2026-02-11 00:00:00 (in-sample)
- **Validation:** IS only — no walk-forward, no true OOS
- **Fixed params:** E3/CB1/RR2.0 only (not grid-searched)
- **Multiple comparisons:** BH FDR applied within each (session × instrument) group
- **Sensitivity:** vol threshold 1.2 vs 1.5 tested (see sensitivity CSV)
- **Concordance sensitivity:** majority_2 vs concordant_3 naturally split in Analysis 3
- **Mechanism (if survived):** Full stack selects days where:
  - ORB is large (G-filter) → sufficient cost absorption
  - Direction is long (1000) → H5 confirmed, shorts are noise
  - Concordance: all instruments break same direction → macro alignment
  - Volume spike → institutional participation (not retail thin market)
  - No NFP/OPEX → clean macro signal (no scheduled volatility event)
- **What could kill it:** regime change, reduced CME liquidity, filter over-fit to 14-month window
