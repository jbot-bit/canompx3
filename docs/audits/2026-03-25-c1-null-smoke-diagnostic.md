# C1 Time-Varying Null — Smoke Diagnostic (5 seeds)

**Date:** 2026-03-25
**Commit:** 0dc2f29
**Method:** Per-year trimmed_std from real bars_1m, 2020-2025, 2026 excluded

## Sigma Schedule (MGC)
| Year | Trimmed Std | Source |
|------|-------------|--------|
| 2020 | 0.53 | bars_1m 2020 only |
| 2021 | 0.37 | bars_1m 2021 only |
| 2022 | 0.41 | bars_1m 2022 only |
| 2023 | 0.36 | bars_1m 2023 only |
| 2024 | 0.51 | bars_1m 2024 only |
| 2025 | 0.99 | bars_1m 2025 only |

## ORB Distribution Comparison (median across 5 seeds)

| Year | sigma | R p50 | S p50 | R p90 | S p90 | R p95 | S p95 | R p99 | S p99 | R G4% | S G4% | R G5% | S G5% |
|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 2020 | 0.53 | 1.50 | 1.60 | 3.30 | 2.49 | 4.00 | 2.90 | 5.00 | 3.54 | 5.8% | 0.0% | 1.6% | 0.0% |
| 2021 | 0.37 | 1.10 | 1.10 | 1.90 | 1.70 | 2.32 | 2.00 | 3.14 | 2.14 | 0.8% | 0.0% | 0.4% | 0.0% |
| 2022 | 0.41 | 1.20 | 1.20 | 2.00 | 1.90 | 2.50 | 2.20 | 3.54 | 2.50 | 0.8% | 0.0% | 0.4% | 0.0% |
| 2023 | 0.36 | 0.90 | 1.10 | 1.80 | 1.70 | 2.12 | 1.90 | 3.34 | 2.38 | 0.8% | 0.0% | 0.4% | 0.0% |
| 2024 | 0.51 | 1.20 | 1.50 | 2.22 | 2.40 | 3.00 | 2.79 | 5.34 | 3.40 | 2.3% | 0.0% | 1.5% | 0.0% |
| 2025 | 0.99 | 3.25 | 3.10 | 7.73 | 4.80 | 9.12 | 5.40 | 15.17 | 6.50 | 37.6% | 21.5% | 26.4% | 8.8% |

## Lower-Bound Check
All years PASS: Synth G4% <= Real G4% in every year.

## 2025 Key Metric
- Synth G4% median=21.5% vs Real=37.6% (ratio=0.57 >= 0.5 threshold)
- Synth G4+ days median: 56 (real: 97)
- ADEQUATE lower-bound null

## Results
- 5/5 seeds: 0 survivors (E1 and E2)
- Runtime: ~24 min/seed (1417-1432s)
