# MNQ US_DATA_1000 Bounded Directional State Study

**Date:** 2026-04-22
**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mnq-usdata1000-directional-state-study-v1.yaml`
**Canonical truth layers:** `daily_features`, `orb_outcomes`
**Holdout policy:** Mode A, `2026-01-01` sacred

## Frozen State Machine

- `confluence_both -> long`
- `near_pivot -> short`
- `else -> no trade`

## Raw State Matrix

| Cell | Parent IS AvgR | Policy IS AvgR(on) | Policy EV/day IS | Parent OOS AvgR | Policy EV/day OOS | Inverse EV/day IS | Verdict |
|------|----------------|--------------------|------------------|-----------------|-------------------|-------------------|---------|
| O5_RR1.0 | +0.087 | +0.151 | +0.057 | +0.016 | +0.032 | -0.006 | **PARK** |
| O5_RR1.5 | +0.093 | +0.140 | +0.053 | -0.034 | -0.002 | -0.000 | **KILL** |
| O5_RR2.0 | +0.091 | +0.187 | +0.071 | -0.028 | +0.035 | -0.014 | **PARK** |
| O15_RR1.0 | +0.097 | +0.092 | +0.034 | +0.210 | +0.054 | +0.032 | **KILL** |
| O15_RR1.5 | +0.106 | +0.143 | +0.052 | +0.187 | +0.084 | +0.015 | **KILL** |
| O15_RR2.0 | +0.052 | +0.115 | +0.043 | +0.059 | +0.091 | -0.028 | **KILL** |

## T0-T8 Audit

### O5_RR1.0

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0003`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.077 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.055 | **PASS** | signal-like WR spread |
| T2 | N=638 ExpR_on=+0.151 PolicyEV/day=+0.057 | **PASS** | deployable sample floor met |
| T3 | WFE=0.51 EV_IS=+0.057 EV_OOS=+0.032 | **PASS** | WFE=0.51 healthy, sign match |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O5_RR1.0 |
| T6 | p=0.0180 ExpR_obs=+0.151 | **PASS** | 1000 shuffles |
| T7 | 6/7 positive years | **PASS** | {2019: -0.045221538461538445, 2020: 0.025976842105263177, 2021: 0.24676046511627905, 2022: 0.3078790476190476, 2023: 0.1325087378640777, 2024: 0.14015, 2025: 0.18392777777777777} |
| T8 | Δ_twin=+0.049 | **FAIL** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.030 |
| policy_minus_inverse_pre | +0.063 |
| confluence_long_minus_short_pre | 0.2874645463675928 |
| near_pivot_short_minus_long_pre | 0.14745538679306233 |

### O5_RR1.5

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0064`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.079 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.032 | **INFO** | modest WR spread |
| T2 | N=629 ExpR_on=+0.140 PolicyEV/day=+0.053 | **PASS** | deployable sample floor met |
| T3 | WFE=-0.03 EV_IS=+0.053 EV_OOS=-0.002 | **FAIL** | WFE=-0.03 < 0.50 |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O5_RR1.5 |
| T6 | p=0.1019 ExpR_obs=+0.140 | **FAIL** | 1000 shuffles |
| T7 | 6/7 positive years | **PASS** | {2019: -0.06032968750000001, 2020: 0.08033473684210526, 2021: 0.18155595238095246, 2022: 0.3372057142857143, 2023: 0.07914, 2024: 0.1438340425531915, 2025: 0.13893908045977013} |
| T8 | Δ_twin=+0.082 | **PASS** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.040 |
| policy_minus_inverse_pre | +0.053 |
| confluence_long_minus_short_pre | 0.2806105667193497 |
| near_pivot_short_minus_long_pre | 0.11934223429675969 |

### O5_RR2.0

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0029`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.085 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.055 | **PASS** | signal-like WR spread |
| T2 | N=619 ExpR_on=+0.187 PolicyEV/day=+0.071 | **PASS** | deployable sample floor met |
| T3 | WFE=0.45 EV_IS=+0.071 EV_OOS=+0.035 | **FAIL** | WFE=0.45 < 0.50 |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O5_RR2.0 |
| T6 | p=0.0230 ExpR_obs=+0.187 | **PASS** | 1000 shuffles |
| T7 | 7/7 positive years | **PASS** | {2019: 0.083521875, 2020: 0.026636559139784945, 2021: 0.3334349397590361, 2022: 0.37925148514851487, 2023: 0.08828877551020407, 2024: 0.1421204301075269, 2025: 0.23292873563218391} |
| T8 | Δ_twin=+0.045 | **FAIL** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.020 |
| policy_minus_inverse_pre | +0.085 |
| confluence_long_minus_short_pre | 0.226016770688858 |
| near_pivot_short_minus_long_pre | 0.22316909391966117 |

### O15_RR1.0

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0243`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.064 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.003 | **INFO** | modest WR spread |
| T2 | N=580 ExpR_on=+0.092 PolicyEV/day=+0.034 | **PASS** | deployable sample floor met |
| T3 | WFE=1.53 EV_IS=+0.034 EV_OOS=+0.054 | **FAIL** | WFE=1.53 LEAKAGE_SUSPECT |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O15_RR1.0 |
| T6 | p=0.5395 ExpR_obs=+0.092 | **FAIL** | 1000 shuffles |
| T7 | 7/7 positive years | **PASS** | {2019: 0.1122, 2020: 0.08424318181818181, 2021: 0.008331460674157307, 2022: 0.14430208333333336, 2023: 0.06906987951807231, 2024: 0.10510370370370371, 2025: 0.12813493975903617} |
| T8 | Δ_twin=+0.025 | **FAIL** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.063 |
| policy_minus_inverse_pre | +0.002 |
| confluence_long_minus_short_pre | -0.08953398574362692 |
| near_pivot_short_minus_long_pre | 0.03821257414607719 |

### O15_RR1.5

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0085`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.072 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.025 ExpR_spread=0.058 | **FAIL** | ARITHMETIC_ONLY |
| T2 | N=547 ExpR_on=+0.143 PolicyEV/day=+0.052 | **PASS** | deployable sample floor met |
| T3 | WFE=1.53 EV_IS=+0.052 EV_OOS=+0.084 | **FAIL** | WFE=1.53 LEAKAGE_SUSPECT |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O15_RR1.5 |
| T6 | p=0.1848 ExpR_obs=+0.143 | **FAIL** | 1000 shuffles |
| T7 | 6/7 positive years | **PASS** | {2019: 0.25332372881355936, 2020: 0.16624186046511633, 2021: -0.010042682926829295, 2022: 0.26652087912087913, 2023: 0.1389618421052632, 2024: 0.09969589041095892, 2025: 0.09526625000000001} |
| T8 | Δ_twin=+0.073 | **PASS** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.054 |
| policy_minus_inverse_pre | +0.037 |
| confluence_long_minus_short_pre | -0.01409571428571424 |
| near_pivot_short_minus_long_pre | 0.12844097215701003 |

### O15_RR2.0

**BH-FDR q (family K=6) on on-signal mean-vs-zero p:** `0.0651`

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 | max |corr|=0.078 (ovn80_fire) | **PASS** | no tautology with deployed proxies |
| T1 | WR_spread=0.034 | **INFO** | modest WR spread |
| T2 | N=513 ExpR_on=+0.115 PolicyEV/day=+0.043 | **PASS** | deployable sample floor met |
| T3 | WFE=1.96 EV_IS=+0.043 EV_OOS=+0.091 | **FAIL** | WFE=1.96 LEAKAGE_SUSPECT |
| T4 | N/A | **INFO** | binary frozen state machine — no theta grid |
| T5 | 6/6 positive EV, 0/6 beat parent | **INFO** | family breadth context for O15_RR2.0 |
| T6 | p=0.1099 ExpR_obs=+0.115 | **FAIL** | 1000 shuffles |
| T7 | 6/7 positive years | **PASS** | {2019: 0.24171964285714287, 2020: 0.15806585365853656, 2021: 0.033473076923076925, 2022: 0.22564166666666668, 2023: -0.04294492753623189, 2024: 0.09447164179104477, 2025: 0.09797532467532466} |
| T8 | Δ_twin=+0.044 | **FAIL** | MES twin sign+mag check |

| Adversarial Check | Value |
|-------------------|-------|
| policy_minus_parent_pre | -0.010 |
| policy_minus_inverse_pre | +0.071 |
| confluence_long_minus_short_pre | 0.06285646523716701 |
| near_pivot_short_minus_long_pre | 0.20126598232080903 |

## IS/OOS Direction Match

| Cell | IS Policy EV/day | OOS Policy EV/day | Direction Match |
|------|------------------|-------------------|-----------------|
| O5_RR1.0 | +0.057 | +0.032 | True |
| O5_RR1.5 | +0.053 | -0.002 | False |
| O5_RR2.0 | +0.071 | +0.035 | True |
| O15_RR1.0 | +0.034 | +0.054 | True |
| O15_RR1.5 | +0.052 | +0.084 | True |
| O15_RR2.0 | +0.043 | +0.091 | True |

## Portfolio EV Comparison

- Queue leader from `docs/plans/2026-04-22-recent-pr-followthrough-queue.md`: `PR48 MES/MGC deployable sizer rule`.
- Comparison verdict: **INFERIOR**.
- This path does not beat its own parent comparator on pre-2026 policy EV, so it remains portfolio-EV inferior to the queue leader `PR48 MES/MGC deployable sizer rule`.

## Bottom Line

- Family verdict: **PARK**. The state machine improves selected-trade quality but does not beat the raw parent on portfolio EV.
- Best pre-2026 policy EV/day cell: `O5_RR2.0` at `+0.071R/day` vs parent `+0.091R/day`.
- No OOS tuning was performed. 2026 results are descriptive only.
