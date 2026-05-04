# MNQ NYSE_CLOSE Failure-Mode Audit

Date: 2026-04-19

## Scope

Audit why `MNQ NYSE_CLOSE` remains unvalidated and undeployed despite a
positive canonical broad baseline on parts of the surface.

This is a failure-mode / blocker audit, not a new discovery sweep.

Canonical proof:

- `gold.db::orb_outcomes`
- `gold.db::daily_features`

Comparison-only blocker context:

- `gold.db::experimental_strategies`
- `gold.db::validated_setups`
- `gold.db::deployable_validated_setups`
- `trading_app/portfolio.py`

## Executive Verdict

`MNQ NYSE_CLOSE` is **not canonically dead**. Broad `RR1.0` remains
positive on all three audited apertures (`O5`, `O15`, `O30`) pre-2026.

But the current unvalidated state is also **not random neglect**:

- validated rows: `0`
- deployable rows: `0`
- experimental rows on record: `10`
- NO_FILTER experimental coverage present: `False`
- O30 experimental coverage present: `False`

The tested surface is narrow and mostly O5-filtered, and those candidates
were rejected mainly for instability rather than outright lack of in-sample edge.

So the honest conclusion is:

- broad session-family still looks alive at RR1.0
- prior attempted filters were mostly unstable
- the unresolved issue is a **pathway / blocker problem**, not a blank discovery void

## Canonical Baseline

| Aperture | RR | N IS | Avg IS | p IS | Pos Years | Long Avg | Short Avg | N OOS | Avg OOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O5 | 1.0 | 805 | +0.0838 | 0.0083 | 5/7 | +0.1180 | +0.0499 | 42 | +0.4832 |
| O5 | 1.5 | 612 | -0.0103 | 0.8213 | 3/7 | +0.0154 | -0.0334 | 29 | +0.2186 |
| O5 | 2.0 | 515 | -0.1678 | 0.0026 | 2/7 | -0.1427 | -0.1894 | 26 | +0.1951 |
| O5 | 2.5 | 459 | -0.3163 | 0.0000 | 0/7 | -0.2245 | -0.3995 | 22 | +0.0440 |
| O5 | 3.0 | 428 | -0.4595 | 0.0000 | 0/7 | -0.3649 | -0.5433 | 19 | -0.2194 |
| O5 | 4.0 | 397 | -0.6520 | 0.0000 | 0/7 | -0.6739 | -0.6335 | 18 | -0.2198 |
| O15 | 1.0 | 327 | +0.1169 | 0.0217 | 6/7 | +0.1792 | +0.0519 | 16 | +0.3150 |
| O15 | 1.5 | 231 | -0.0233 | 0.7592 | 4/7 | -0.0087 | -0.0365 | 11 | +0.3004 |
| O15 | 2.0 | 191 | -0.1657 | 0.0759 | 1/7 | -0.1643 | -0.1670 | 9 | +0.2747 |
| O15 | 2.5 | 170 | -0.3267 | 0.0016 | 1/7 | -0.3509 | -0.3052 | 7 | -0.0439 |
| O15 | 3.0 | 157 | -0.5017 | 0.0000 | 0/7 | -0.4660 | -0.5351 | 6 | -0.3708 |
| O15 | 4.0 | 147 | -0.6503 | 0.0000 | 0/7 | -0.7329 | -0.5773 | 6 | -0.2135 |
| O30 | 1.0 | 197 | +0.1384 | 0.0334 | 5/7 | +0.1915 | +0.0790 | 6 | +0.9241 |
| O30 | 1.5 | 131 | -0.0667 | 0.5032 | 3/7 | -0.0421 | -0.0910 | 3 | +1.3948 |
| O30 | 2.0 | 105 | -0.2857 | 0.0182 | 1/7 | -0.3009 | -0.2715 | 1 | +1.9052 |
| O30 | 2.5 | 95 | -0.4172 | 0.0017 | 1/7 | -0.5013 | -0.3415 | 1 | +2.3894 |
| O30 | 3.0 | 90 | -0.5016 | 0.0003 | 1/7 | -0.5728 | -0.4365 | 0 | +nan |
| O30 | 4.0 | 85 | -0.6151 | 0.0000 | 1/7 | -0.6637 | -0.5699 | 0 | +nan |

## RR1.0 Year Map (Pre-2026)

| Aperture | Year | N | Avg R |
|---|---:|---:|---:|
| O5 | 2019 | 91 | +0.0801 |
| O5 | 2020 | 137 | +0.2147 |
| O5 | 2021 | 107 | +0.0407 |
| O5 | 2022 | 123 | +0.1680 |
| O5 | 2023 | 103 | -0.0238 |
| O5 | 2024 | 111 | +0.1656 |
| O5 | 2025 | 133 | -0.0767 |
| O15 | 2019 | 35 | +0.0614 |
| O15 | 2020 | 47 | +0.3668 |
| O15 | 2021 | 38 | +0.0052 |
| O15 | 2022 | 51 | +0.2305 |
| O15 | 2023 | 51 | +0.0351 |
| O15 | 2024 | 40 | +0.1781 |
| O15 | 2025 | 65 | -0.0312 |
| O30 | 2019 | 35 | +0.0614 |
| O30 | 2020 | 47 | +0.3669 |
| O30 | 2021 | 27 | -0.0651 |
| O30 | 2022 | 22 | +0.1365 |
| O30 | 2023 | 27 | +0.2205 |
| O30 | 2024 | 16 | -0.0647 |
| O30 | 2025 | 23 | +0.0745 |

## Experimental Surface Actually Tried

| Filter | RR | Aperture | N | ExpR | p | Status | Bucket |
|---|---:|---:|---:|---:|---:|---|---|
| GAP_R015 | 1.0 | O5 | 92 | +0.1744 | 0.0682 | REJECTED | year_stability |
| OVNRNG_100 | 1.0 | O5 | 267 | +0.1510 | 0.0070 | REJECTED | year_stability |
| ORB_G5_NOFRI | 1.0 | O5 | 640 | +0.0734 | 0.0415 | REJECTED | year_stability |
| OVNRNG_100 | 1.5 | O5 | 204 | +0.0835 | 0.3060 | REJECTED | era_instability |
| GAP_R015 | 1.5 | O5 | 64 | +0.0498 | 0.7333 | REJECTED | year_stability |
| X_MES_ATR60 | 1.5 | O5 | 258 | +0.0055 | 0.9387 | REJECTED | year_stability |
| ORB_G5 | 1.5 | O5 | 544 | -0.0078 | 0.8726 | REJECTED | era_instability |
| COST_LT12 | 1.5 | O5 | 461 | -0.0424 | 0.4271 | REJECTED | era_instability |
| ORB_G5 | 1.5 | O15 | 204 | -0.0269 | 0.7396 | REJECTED | negative_expectancy |
| ORB_G5 | 2.0 | O5 | 462 | -0.1569 | 0.0079 | REJECTED | era_instability |

## Rejection Buckets

| Bucket | N |
|---|---:|
| year_stability | 5 |
| era_instability | 4 |
| negative_expectancy | 1 |

## Comparison-Layer Blocker Note

Current portfolio construction still excludes `NYSE_CLOSE` from the raw
baseline path in [trading_app/portfolio.py](/mnt/c/Users/joshd/canompx3/trading_app/portfolio.py:633)
and again in the multi-RR builder at
[trading_app/portfolio.py](/mnt/c/Users/joshd/canompx3/trading_app/portfolio.py:991).

That is not proof the exclusion is wrong, but it is now a **questionable
blocker** because the broad RR1.0 session-family remains canonically positive
while the experimented filter surface was narrow and unstable.

## Bottom Line

`MNQ NYSE_CLOSE` should not be treated as a fresh discovery void, and it
also should not be treated as dead. The right next move is a narrow
`RR1.0` native governance/failure-mode follow-up on the broad session-family,
not another random filter sweep.

## Outputs

- `research/output/mnq_nyse_close_failure_mode_audit_baseline.csv`
- `research/output/mnq_nyse_close_failure_mode_audit_rr1_years.csv`
- `research/output/mnq_nyse_close_failure_mode_audit_experimental.csv`
- `research/output/mnq_nyse_close_failure_mode_audit_rejection_buckets.csv`
