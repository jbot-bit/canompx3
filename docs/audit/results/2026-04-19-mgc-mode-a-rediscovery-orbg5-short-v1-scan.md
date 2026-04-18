# MGC Mode A rediscovery — ORB_G5 long RR1.5 K=4 scan

**Generated:** 2026-04-18T22:47:40+00:00
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mgc-mode-a-rediscovery-orbg5-v1.yaml` (LOCKED, commit_sha=e227ceb3)
**Script:** `research/mgc_mode_a_rediscovery_orbg5_v1_scan.py`
**IS window:** `trading_day < 2026-01-01` (Mode A)

## Summary

Cells: 4 | CONTINUE: 0 | KILL: 4

**K2 baseline sanity smoke-test:** PASS (same-path reproducibility only; see pre-reg § Baseline cross-check).

## Per-cell IS results

| Cell | Session | N_base | N_on | Fire% | ExpR_base | ExpR_on | Δ_IS | t | raw_p | boot_p | q_family | years_pos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1_LONDON_METALS | LONDON_METALS | 448 | 32 | 0.071 | -0.1548 | 0.1568 | 0.3116 | 0.754 | 0.4565 | 0.4897 | 0.7665 | 1 |
| H2_COMEX_SETTLE | COMEX_SETTLE | 416 | 20 | 0.048 | -0.1948 | -0.0781 | 0.1167 | -0.301 | 0.7665 | 0.7667 | 0.7665 | 1 |
| H3_US_DATA_1000 | US_DATA_1000 | 384 | 123 | 0.320 | -0.0490 | 0.0359 | 0.0849 | 0.345 | 0.7311 | 0.6972 | 0.7665 | 2 |
| H4_EUROPE_FLOW | EUROPE_FLOW | 424 | 29 | 0.068 | -0.1375 | 0.1147 | 0.2522 | 0.525 | 0.6034 | 0.4574 | 0.7665 | 1 |

## Gate breakdown

| Cell | bh_pass_family | abs_t_IS_ge_3 | N_IS_on_ge_100 | years_positive_ge_3 | bootstrap_p_lt_0.10 | ExpR_on_IS_gt_0 | not_tautology | not_extreme_fire | not_arithmetic_only | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H1_LONDON_METALS | N | N | N | N | N | Y | Y | Y | Y | KILL |
| H2_COMEX_SETTLE | N | N | N | N | N | N | Y | N | Y | KILL |
| H3_US_DATA_1000 | N | N | Y | N | N | Y | Y | Y | Y | KILL |
| H4_EUROPE_FLOW | N | N | N | N | N | Y | Y | Y | N | KILL |

## Flags & T0 (cross-filter tautology — ORB_G5 is trivially correlated with orb_size; check vs atr_20, overnight_range)

| Cell | fire_rate | corr_orbsize (self, expected ~1) | corr_atr | corr_ovnrng | tautology | extreme_fire | arithmetic_only |
|---|---:|---:|---:|---:|---|---|---|
| H1_LONDON_METALS | 0.071 | 0.712 | 0.599 | 0.673 | N | N | N |
| H2_COMEX_SETTLE | 0.048 | 0.728 | 0.468 | 0.445 | N | Y | N |
| H3_US_DATA_1000 | 0.320 | 0.767 | 0.471 | 0.452 | N | N | N |
| H4_EUROPE_FLOW | 0.068 | 0.686 | 0.491 | 0.561 | N | N | Y |

## OOS descriptive (NOT used to select or tune)

| Cell | N_OOS_on | ExpR_OOS_on | Δ_OOS | dir_match |
|---|---:|---:|---:|---|
| H1_LONDON_METALS | 27 | 0.1347 | -0.0294 | N |
| H2_COMEX_SETTLE | 29 | -0.2710 | 0.0039 | Y |
| H3_US_DATA_1000 | 31 | 0.0856 | 0.0000 | N |
| H4_EUROPE_FLOW | 18 | 0.1826 | 0.2434 | Y |

## Per-year IS breakdown

| Cell | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| H1_LONDON_METALS | — | — | — | +0.157(N=32) |
| H2_COMEX_SETTLE | N=1 | — | N=1 | +0.024(N=18) |
| H3_US_DATA_1000 | N=7 | --0.017(N=14) | +0.058(N=35) | +0.075(N=67) |
| H4_EUROPE_FLOW | — | — | N=3 | +0.150(N=26) |

## Decision

**Verdict: KILL per pre-reg K1.** Zero of 4 cells pass all gate clauses. MGC ORB_G5 long RR1.5 on the 4 pre-registered sessions does NOT yield a Pathway A Chordia-validated edge on 3.5-year Mode A IS. Honest negative evidence on MGC cross-instrument-mirror hypothesis. Pre-reg explicitly anticipated this outcome; baselines (approx_t 0.23-1.47) predicted a KILL. No re-runs with different thresholds.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/mgc_mode_a_rediscovery_orbg5_v1_scan.py
```

