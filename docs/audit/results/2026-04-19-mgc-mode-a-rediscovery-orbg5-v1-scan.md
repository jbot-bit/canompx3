# MGC Mode A rediscovery — ORB_G5 long RR1.5 K=4 scan

**Generated:** 2026-04-18T16:10:26+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1.yaml` (LOCKED, commit_sha=e227ceb3)
**Script:** `research/mgc_mode_a_rediscovery_orbg5_v1_scan.py`
**IS window:** `trading_day < 2026-01-01` (Mode A)

## Summary

Cells: 4 | CONTINUE: 0 | KILL: 4

**K2 baseline sanity smoke-test:** PASS (same-path reproducibility only; see pre-reg § Baseline cross-check).

## Per-cell IS results

| Cell | Session | N_base | N_on | Fire% | ExpR_base | ExpR_on | Δ_IS | t | raw_p | boot_p | q_family | years_pos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_LONDON_METALS | LONDON_METALS | 469 | 41 | 0.087 | -0.1528 | 0.1793 | 0.3321 | 0.985 | 0.3306 | 0.1569 | 0.6611 | 1 |
| H2_COMEX_SETTLE | COMEX_SETTLE | 431 | 33 | 0.077 | -0.2012 | 0.0470 | 0.2482 | 0.232 | 0.8182 | 0.8367 | 0.8182 | 1 |
| H3_US_DATA_1000 | US_DATA_1000 | 439 | 133 | 0.303 | -0.0298 | 0.0671 | 0.0969 | 0.665 | 0.5075 | 0.4676 | 0.6766 | 1 |
| H4_EUROPE_FLOW | EUROPE_FLOW | 493 | 26 | 0.053 | -0.1111 | 0.3377 | 0.4488 | 1.474 | 0.1531 | 0.0560 | 0.6123 | 1 |

## Gate breakdown

| Cell | bh_pass_family | abs_t_IS_ge_3 | N_IS_on_ge_100 | years_positive_ge_3 | bootstrap_p_lt_0.10 | ExpR_on_IS_gt_0 | not_tautology | not_extreme_fire | not_arithmetic_only | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H1_LONDON_METALS | N | N | N | N | N | Y | Y | Y | Y | KILL |
| H2_COMEX_SETTLE | N | N | N | N | N | Y | Y | Y | Y | KILL |
| H3_US_DATA_1000 | N | N | Y | N | N | Y | Y | Y | Y | KILL |
| H4_EUROPE_FLOW | N | N | N | N | Y | Y | Y | Y | Y | KILL |

## Flags & T0 (cross-filter tautology — ORB_G5 is trivially correlated with orb_size; check vs atr_20, overnight_range)

| Cell | fire_rate | corr_orbsize (self, expected ~1) | corr_atr | corr_ovnrng | tautology | extreme_fire | arithmetic_only |
|---|---:|---:|---:|---:|---|---|---|
| H1_LONDON_METALS | 0.087 | 0.765 | 0.513 | 0.613 | N | N | N |
| H2_COMEX_SETTLE | 0.077 | 0.790 | 0.547 | 0.644 | N | N | N |
| H3_US_DATA_1000 | 0.303 | 0.786 | 0.556 | 0.524 | N | N | N |
| H4_EUROPE_FLOW | 0.053 | 0.699 | 0.433 | 0.469 | N | N | N |

## OOS descriptive (NOT used to select or tune)

| Cell | N_OOS_on | ExpR_OOS_on | Δ_OOS | dir_match |
|---|---:|---:|---:|---|
| H1_LONDON_METALS | 36 | -0.1538 | -0.1051 | N |
| H2_COMEX_SETTLE | 29 | -0.1901 | -0.0116 | N |
| H3_US_DATA_1000 | 39 | -0.1366 | 0.0000 | N |
| H4_EUROPE_FLOW | 36 | -0.0895 | 0.0060 | Y |

## Per-year IS breakdown

| Cell | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| H1_LONDON_METALS | — | N=2 | N=2 | +0.245(N=37) |
| H2_COMEX_SETTLE | — | N=1 | — | +0.080(N=32) |
| H3_US_DATA_1000 | N=7 | --0.065(N=10) | --0.015(N=28) | +0.089(N=88) |
| H4_EUROPE_FLOW | N=1 | — | N=3 | +0.263(N=22) |

## Decision

**Verdict: KILL per pre-reg K1.** Zero of 4 cells pass all gate clauses. MGC ORB_G5 long RR1.5 on the 4 pre-registered sessions does NOT yield a Pathway A Chordia-validated edge on 3.5-year Mode A IS. Honest negative evidence on MGC cross-instrument-mirror hypothesis. Pre-reg explicitly anticipated this outcome; baselines (approx_t 0.23-1.47) predicted a KILL. No re-runs with different thresholds.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mgc_mode_a_rediscovery_orbg5_v1_scan.py
```

