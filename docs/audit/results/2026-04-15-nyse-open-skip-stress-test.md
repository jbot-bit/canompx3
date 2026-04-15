# NYSE_OPEN SKIP_GARCH_70 — Adversarial Stress Test

**Date:** 2026-04-15
**Trigger:** User demanded stress-test before implementing NYSE_OPEN SKIP filter. Red-team the finding to ensure no bias, lookahead, or pigeonholing.

## Base claim

MNQ_NYSE_OPEN across 54 (apt × RR × dir × threshold) test cells showed 46/54 negative sr_lift on garch_pct overlay. Binomial sign test p=6.9×10⁻⁸. Identified as strongest SKIP candidate on the project.

---

## S1 — Shuffle control

Real data: 0.148 positive fraction across all NYSE_OPEN cells.
Shuffled (100 runs): median positive frac = 0.500, range [0.315, 0.704].
Shuffle p-value: 0.0099

**Verdict:** PASS — methodology distinguishes real signal from shuffled noise.

---

## S2 — Per-year breakdown

| Year | N cells | % positive | avg lift |
|---|---|---|---|
| 2020 | 16 | 31.2% | -0.366 |
| 2021 | 18 | 5.6% | -0.156 |
| 2022 | 18 | 0.0% | -0.338 |
| 2023 | 16 | 12.5% | -0.380 |
| 2024 | 18 | 50.0% | +0.034 |
| 2025 | 18 | 33.3% | -0.053 |

**3 of 6 years** show <30% positive fraction (strongly inverse).
**Verdict:** YEAR-DEPENDENT

---

## S3 — Direction split

LONG cells: 0 positive / 27 negative
SHORT cells: 8 positive / 19 negative

LONG inverse fraction: 100.0%
SHORT inverse fraction: 70.4%

**Verdict:** BOTH directions inverse

---

## S4 — Event-day exclusion

| apt | rr | dir | full sr_lift (N_on) | clean sr_lift (N_on) | delta |
|---|---|---|---|---|---|
| O5 | 1.0 | long | -0.079 (176) | -0.060 (159) | +0.020 |
| O5 | 1.5 | long | -0.096 (175) | -0.081 (158) | +0.015 |
| O15 | 1.0 | long | -0.203 (174) | -0.183 (158) | +0.020 |
| O15 | 1.5 | long | -0.094 (164) | -0.078 (148) | +0.015 |
| O5 | 1.0 | short | +0.037 (201) | -0.024 (182) | -0.062 |
| O5 | 1.5 | short | -0.078 (197) | -0.100 (179) | -0.022 |
| O15 | 1.0 | short | -0.025 (184) | -0.031 (166) | -0.006 |
| O15 | 1.5 | short | -0.111 (170) | -0.124 (153) | -0.013 |

**6 of 8 cells** still show inverse after event-day exclusion.

---

## S5 — Break-direction confounder

P(break=long | garch>=70): 0.468
P(break=long | garch<70): 0.520
Diff: -0.051

**Verdict:** CONFOUNDER PRESENT — garch IS associated with break direction at NYSE_OPEN.

---

## S6 — MAE/MFE decomposition

Is the inverse signal from WR change, bigger losses, or smaller wins?

| dir | WR on | WR off | AvgWin on | AvgWin off | AvgLoss on | AvgLoss off | MAE on | MAE off |
|---|---|---|---|---|---|---|---|---|
| long | 52.8% | 57.3% | +0.950 | +0.930 | -1.000 | -1.000 | +0.699 | +0.691 |
| short | 58.2% | 57.0% | +0.952 | +0.928 | -1.000 | -1.000 | +0.677 | +0.654 |

---

## S7 — Continuous regression

If garch is a clean regime indicator, linear slope on pnl_r should be consistent.

| Direction | N | slope | r | p |
|---|---|---|---|---|
| long | 5181 | -0.00197 | -0.052 | 0.0002 |
| short | 5216 | -0.00024 | -0.007 | 0.6375 |

Negative slope = garch higher → pnl lower (inverse signal confirmed continuously).

---

## S8 — Tail behavior

If the effect collapses or reverses at threshold 90, it's a mid-tail artifact, not a robust regime.

| apt | rr | dir | @70 lift | @80 lift | @90 lift |
|---|---|---|---|---|---|
| O5 | 1.0 | long | -0.079 (176) | -0.081 (124) | -0.005 (72) |
| O5 | 1.0 | short | +0.037 (201) | +0.082 (145) | -0.008 (76) |
| O5 | 1.5 | long | -0.096 (175) | -0.087 (123) | -0.063 (72) |
| O5 | 1.5 | short | -0.078 (197) | -0.019 (144) | +0.003 (76) |
| O5 | 2.0 | long | -0.105 (171) | -0.070 (121) | +0.043 (72) |
| O5 | 2.0 | short | -0.034 (195) | +0.012 (142) | -0.037 (75) |
| O15 | 1.0 | long | -0.203 (174) | -0.204 (119) | -0.156 (73) |
| O15 | 1.0 | short | -0.025 (184) | -0.037 (135) | +0.006 (69) |
| O15 | 1.5 | long | -0.094 (164) | -0.098 (111) | -0.062 (71) |
| O15 | 1.5 | short | -0.111 (170) | -0.070 (127) | -0.031 (65) |
| O15 | 2.0 | long | -0.129 (143) | -0.121 (96) | -0.034 (62) |
| O15 | 2.0 | short | -0.110 (161) | -0.115 (120) | +0.002 (61) |
| O30 | 1.0 | long | -0.125 (151) | -0.013 (102) | +0.022 (60) |
| O30 | 1.0 | short | -0.051 (156) | -0.056 (117) | -0.053 (66) |
| O30 | 1.5 | long | -0.213 (117) | -0.044 (77) | +0.023 (48) |
| O30 | 1.5 | short | -0.031 (133) | -0.061 (98) | +0.034 (56) |
| O30 | 2.0 | long | -0.273 (99) | -0.137 (62) | -0.041 (38) |
| O30 | 2.0 | short | -0.069 (114) | -0.108 (85) | +0.017 (49) |

---

## Final stress-test verdict

Each S-test either confirms or falsifies part of the base claim. Consolidated below.
