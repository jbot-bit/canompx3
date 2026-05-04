# P4: E3 x Direction at 1000

## Hypothesis
1000 session has documented LONG bias at E1. E3 (retrace entry) on LONG days
= buying the dip into an upward-biased session.
break_dir is known at entry time — NOT look-ahead.

## Method
- E3 outcomes joined to daily_features.orb_{label}_break_dir
- JOIN includes orb_minutes (prevents 3x row inflation)
- Sessions tested: 1000 (primary), 0900, 1800
- BH FDR at q=0.1 across 216 rows
- Min N = 30

## Summary: 216 groups tested
- LONG groups with BH-sig positive avg_r: 0

## DID NOT SURVIVE

No LONG-filtered E3 groups passed BH FDR with avg_r > 0.
E3 x LONG direction filter is NOT supported as a standalone improvement.

## Top ExpR Lifts: LONG vs ALL

| Symbol | Session | RR | CB | avg_r_ALL | avg_r_LONG | lift |
|--------|---------|----|----|-----------|------------|------|
| MNQ | 1000 | 4.0 | 1 | -0.0006 | +0.0762 | +0.0768 |
| MNQ | 1000 | 2.0 | 1 | -0.0771 | -0.0131 | +0.0640 |
| MNQ | 1000 | 1.5 | 1 | -0.0549 | +0.0064 | +0.0613 |
| MES | 1000 | 4.0 | 1 | -0.1805 | -0.1290 | +0.0515 |
| MES | 1800 | 3.0 | 1 | -0.2220 | -0.1712 | +0.0508 |
| MNQ | 1000 | 3.0 | 1 | -0.0863 | -0.0385 | +0.0478 |
| MGC | 1800 | 4.0 | 1 | -0.4048 | -0.3580 | +0.0468 |
| MGC | 1800 | 2.0 | 1 | -0.3655 | -0.3190 | +0.0465 |
| MES | 1000 | 2.5 | 1 | -0.1552 | -0.1100 | +0.0452 |
| MGC | 1800 | 3.0 | 1 | -0.3995 | -0.3553 | +0.0442 |

## MGC 1000 — Year-by-Year (ALL vs LONG)

| Year | Direction | N | avg_r | WR |
|------|-----------|---|-------|----|
| 2016 | ALL | 1296 | -0.4351 | 27.5% |
| 2016 | LONG | 654 | -0.4181 | 28.6% |
| 2017 | ALL | 1440 | -0.5526 | 20.9% |
| 2017 | LONG | 624 | -0.6074 | 17.3% |
| 2018 | ALL | 1409 | -0.5693 | 20.3% |
| 2018 | LONG | 714 | -0.6347 | 17.1% |
| 2019 | ALL | 1398 | -0.5677 | 17.9% |
| 2019 | LONG | 678 | -0.5689 | 17.1% |
| 2020 | ALL | 1428 | -0.2753 | 32.3% |
| 2020 | LONG | 720 | -0.3871 | 26.4% |
| 2021 | ALL | 1398 | -0.3921 | 29.4% |
| 2021 | LONG | 606 | -0.3299 | 32.2% |
| 2022 | ALL | 1410 | -0.4408 | 27.2% |
| 2022 | LONG | 702 | -0.4976 | 24.5% |
| 2023 | ALL | 1422 | -0.4586 | 26.7% |
| 2023 | LONG | 756 | -0.4206 | 29.5% |
| 2024 | ALL | 1404 | -0.4127 | 27.9% |
| 2024 | LONG | 540 | -0.3006 | 36.1% |
| 2025 | ALL | 1403 | -0.1300 | 34.5% |
| 2025 | LONG | 749 | -0.1227 | 34.8% |
| 2026 | ALL | 126 | -0.1870 | 28.6% |
| 2026 | LONG | 78 | -0.1113 | 28.2% |

## MNQ 1000 — Year-by-Year (ALL vs LONG)

| Year | Direction | N | avg_r | WR |
|------|-----------|---|-------|----|
| 2024 | ALL | 1284 | -0.0765 | 30.6% |
| 2024 | LONG | 498 | -0.0334 | 32.9% |
| 2025 | ALL | 1446 | -0.0754 | 28.3% |
| 2025 | LONG | 756 | -0.0417 | 29.6% |
| 2026 | ALL | 114 | +0.0618 | 33.3% |
| 2026 | LONG | 42 | +0.6126 | 50.0% |

## MES 1000 — Year-by-Year (ALL vs LONG)

| Year | Direction | N | avg_r | WR |
|------|-----------|---|-------|----|
| 2019 | ALL | 1251 | -0.2275 | 31.3% |
| 2019 | LONG | 645 | -0.1427 | 36.0% |
| 2020 | ALL | 1373 | -0.2024 | 28.7% |
| 2020 | LONG | 659 | -0.1571 | 30.0% |
| 2021 | ALL | 1397 | -0.1640 | 32.9% |
| 2021 | LONG | 701 | -0.1978 | 32.1% |
| 2022 | ALL | 1422 | -0.0676 | 32.4% |
| 2022 | LONG | 648 | -0.0574 | 31.8% |
| 2023 | ALL | 1434 | -0.2465 | 32.1% |
| 2023 | LONG | 690 | -0.2315 | 33.6% |
| 2024 | ALL | 1440 | -0.1984 | 32.9% |
| 2024 | LONG | 576 | -0.1533 | 33.9% |
| 2025 | ALL | 1470 | -0.0738 | 31.6% |
| 2025 | LONG | 714 | -0.0817 | 30.8% |
| 2026 | ALL | 168 | +0.0271 | 35.7% |
| 2026 | LONG | 78 | +0.2386 | 42.3% |

## CAVEATS
- E3 has lower fill rate than E1 — sample is smaller
- break_dir is NOT look-ahead: the break has already occurred when E3 waits for retrace
- DST not split for 0900/1800 (1000 is clean — no DST contamination)
- ORB_G4+ filter not applied — validated strategy quality days may differ

## NEXT STEPS
- If SURVIVED: add direction_filter='long' to E3 LiveStrategySpec entries
- Sensitivity: does lift hold at RR2.0 and RR2.5? If only RR1.0 → curve-fit risk
- Check: is LONG pct ~50%? If not, investigate data issue