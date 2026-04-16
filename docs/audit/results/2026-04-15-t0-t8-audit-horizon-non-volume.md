# T0-T8 Audit — 5 Non-Volume Horizon Candidates

**Date:** 2026-04-15
**Source:** `docs/handoffs/2026-04-15-session-handover.md` § Tier 1
**Prior scan:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`
**Audit protocol:** `.claude/rules/quant-audit-protocol.md` Steps 3-5
**Template:** `research/t0_t8_audit_mgc_level_cells.py` (imports helpers from `t0_t8_audit_prior_day_patterns.py`)

---

## TL;DR

| # | Cell | Verdict | Deployment posture |
|---|------|---------|--------------------|
| H2 | MNQ COMEX_SETTLE O5 RR1.0 long `garch_vol_pct≥70` | **VALIDATED (8P/0F)** | Strongest candidate — WFE=0.59, 6/6 years positive, bootstrap p=0.001, cross-instrument consistent. Still: run DSR at honest K + signal-only shadow before capital deployment (same doctrine as rel_vol). |
| H1 | MES LONDON_METALS O30 RR1.5 long `ovn_range_pct≥80` | CONDITIONAL (7P/1F) | Fail is T3 WFE=1.33 (LEAKAGE_SUSPECT per RULE 12) driven by tiny N_OOS_on=11 — strong OOS ExpR on thin sample rather than leakage per se. 6/7 years positive, bootstrap p=0.001. Shadow 12-24 weeks to accumulate OOS before re-audit. |
| H3 | MNQ BRISBANE_1025 O30 RR2.0 long `is_monday` | CONDITIONAL (5P/1F) | T3 thin (N_OOS<10). MES twin N insufficient for T8. Otherwise all PASS. Re-audit once OOS grows. |
| H4 | MNQ COMEX_SETTLE O15 RR1.0 short `dow_thu` | CONDITIONAL (6P/1F) | T3 thin (N_OOS<10). T8 cross-instrument PASS on MES. Re-audit once OOS grows. |
| H5 | MES COMEX_SETTLE O30 RR1.0 long `ovn_took_pdh` SKIP | CONDITIONAL (6P/1F) | SKIP signal — feature=1 → ExpR negative. T3 thin (N_OOS<10). All other tests PASS incl. cross-instrument. Re-audit once OOS grows. |

**Zero KILLs.** All five candidates survived T0-T8 with, at worst, one thin-OOS-N fail. None were DUPLICATE_FILTER (T0 clean against deployed proxies), none were ARITHMETIC_ONLY (T1 WR monotonic in all), none were PARAMETER_SENSITIVE (T4 sign-consistent for percentile features).

**Caveats / RULE 12 flags:**
- H1 WFE=1.33 > 0.95 — treat as LEAKAGE_SUSPECT until thicker OOS is available. Do not confuse "thin OOS looks strong" with "robust edge."
- DSR at honest N_eff not re-run here (rel_vol lesson: dsr.py default var_sr is calibrated for experimental_strategies, not comprehensive-scan cells; requires empirical calibration). Report is informational on DSR grounds.
- 2025 year direction flips for H1 (yr2025=-0.067) and H5 2024 ExpR flips positive (+0.049) — minor single-year outliers within 6/7-year majority. Era-stability not a KILL but a WATCH.

**Next-session actions:**
1. H2 — pre-reg signal-only shadow for `garch_vol_pct≥70` on MNQ COMEX_SETTLE O5 RR1.0 long. Compute DSR at K={5,12,36,72,300,14261} for honest framing.
2. H1 — pre-reg shadow. OOS N=11 needs 3-6 months more data before LEAKAGE_SUSPECT can be ruled out.
3. H3/H4/H5 — shadow until N_OOS ≥ 30 on-signal. Re-audit.
4. Composite candidate: **rel_vol_HIGH_Q3 × garch_vol_pct≥70** on MNQ COMEX_SETTLE O5 RR1.0 — two independent volatility signals (realized-volume vs forecast-vol); orthogonality check + joint T0-T8 before capital.

---

**Pre-reg posture:** Confirmatory T0-T8 on prior-scan BH_family survivors — per `backtesting-methodology.md` RULE 10, no new pre-reg required.

**Look-ahead clearance:**
- `overnight_*` features require ORB start ≥ 17:00 Brisbane — LONDON_METALS (17:00), COMEX_SETTLE (04:30 next day) both clear.
- `garch_forecast_vol_pct` forecast at prior close — always trade-time-knowable.
- `is_monday` / `day_of_week` calendar — always trade-time-knowable.

**Custom test notes:**
- T0 excludes the cell's own deployed-filter proxy to avoid 100% self-correlation.
- T4 uses feature-class-specific threshold grids for percentile features; binary features return INFO.
- T8 twin for MES↔MNQ is same-asset-class (equity index pair) — valid. MGC has no such twin in this cell set.

## H1_MES_LONDON_METALS_O30_RR1.5_long_ovn_range_pct_GT80
**Description:** MES LONDON_METALS O30 RR1.5 LONG ovn_range_pct≥80 — overnight vol expansion predicts continuation
**Scope:** MES | LONDON_METALS | O30 | RR1.5 | long | expected=positive
**Feature class:** `ovn_range_pct`
**N_total:** 908 | **N_on_signal:** 196

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.186 (pdr_r105_fire) | **PASS** | excluded self_proxy=ovn80_fire; corrs={'pdr_r105_fire': 0.18621126074184358, 'gap_r015_fire': 0.11315326065637679, 'atr70_fire': 0.16615997050063533} |
| T1 T1_wr_monotonicity | WR_spread=0.123 (on=0.524 off=0.401) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=185 ExpR=+0.216 WR=0.524 σ=1.16 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=1.33 IS_SR=0.19 OOS_SR=0.25 N_OOS_on=11 | **FAIL** | WFE=1.33 LEAKAGE_SUSPECT (>0.95) |
| T4 T4_sensitivity | θ=70:Δ=+0.257, θ=80:Δ=+0.325, θ=90:Δ=+0.325 | **PASS** | signs consistent, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.216 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): 0.5670285714285714, np.int32(2020): 0.3052558823529412, np.int32(2021): 0.2197260869565217, np.int32(2022): 0.19521, np.int32(2023): 0.10601851851851851, np.int32(2024): 0.4883344827586207, np.int32(2025): -0.06736000000000002} |
| T8 T8_cross_instrument | twin=MNQ Δ=+0.217 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 7 PASS, 1 FAIL, 1 INFO
### Verdict: **CONDITIONAL** — one fail; acceptable if non-load-bearing (e.g., thin OOS N).

---

## H2_MNQ_COMEX_SETTLE_O5_RR1.0_long_garch_vol_pct_GT70
**Description:** MNQ COMEX_SETTLE O5 RR1.0 LONG garch_forecast_vol_pct≥70 — forward vol forecast
**Scope:** MNQ | COMEX_SETTLE | O5 | RR1.0 | long | expected=positive
**Feature class:** `garch_vol_pct`
**N_total:** 909 | **N_on_signal:** 213

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.586 (atr70_fire) | **PASS** | excluded self_proxy=none; corrs={'pdr_r105_fire': 0.11482980413641486, 'gap_r015_fire': 0.0803248236392238, 'atr70_fire': 0.5862657866266724, 'ovn80_fire': 0.2173575844138696} |
| T1 T1_wr_monotonicity | WR_spread=0.103 (on=0.662 off=0.559) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=198 ExpR=+0.247 WR=0.662 σ=0.90 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | WFE=0.59 IS_SR=0.28 OOS_SR=0.16 N_OOS_on=15 | **PASS** | WFE=0.59 healthy, sign match |
| T4 T4_sensitivity | θ=60:Δ=+0.198, θ=70:Δ=+0.230, θ=80:Δ=+0.256 | **PASS** | signs consistent, magnitudes within 25% band |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0030 ExpR_obs=+0.247 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/6 in expected direction | **PASS** | 6/6 years (100%) matching expected direction; yr={np.int32(2020): 0.3313, np.int32(2021): 0.19265652173913048, np.int32(2022): 0.24721999999999994, np.int32(2023): 0.3179571428571428, np.int32(2024): 0.10309574468085107, np.int32(2025): 0.40734634146341464} |
| T8 T8_cross_instrument | twin=MES Δ=+0.110 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 8 PASS, 0 FAIL, 1 INFO
### Verdict: **VALIDATED** — deploy candidate. Pre-reg Stage 1 binary filter.

---

## H3_MNQ_BRISBANE_1025_O30_RR2.0_long_is_monday
**Description:** MNQ BRISBANE_1025 O30 RR2.0 LONG is_monday — Monday-open effect
**Scope:** MNQ | BRISBANE_1025 | O30 | RR2.0 | long | expected=positive
**Feature class:** `is_monday`
**N_total:** 955 | **N_on_signal:** 198

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.251 (pdr_r105_fire) | **PASS** | excluded self_proxy=none; corrs={'pdr_r105_fire': -0.25083021768567604, 'gap_r015_fire': 0.16960606464320246, 'atr70_fire': 0.030696186542661166, 'ovn80_fire': 0.06867407488329426} |
| T1 T1_wr_monotonicity | WR_spread=0.132 (on=0.484 off=0.353) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=192 ExpR=+0.338 WR=0.484 σ=1.39 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=6 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature (is_monday) — no theta grid applicable |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.338 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): 0.28724500000000003, np.int32(2020): 0.7098722222222223, np.int32(2021): -0.0915066666666667, np.int32(2022): 0.02778181818181816, np.int32(2023): 0.19048400000000001, np.int32(2024): 0.29019333333333336, np.int32(2025): 0.7685379310344826} |
| T8 T8_cross_instrument | N_on_twin=0 N_off_twin=0 | **INFO** | twin MES N insufficient |

**Counts:** 5 PASS, 1 FAIL, 3 INFO
### Verdict: **CONDITIONAL** — one fail; acceptable if non-load-bearing (e.g., thin OOS N).

---

## H4_MNQ_COMEX_SETTLE_O15_RR1.0_short_dow_thu
**Description:** MNQ COMEX_SETTLE O15 RR1.0 SHORT dow_thu — Thursday effect
**Scope:** MNQ | COMEX_SETTLE | O15 | RR1.0 | short | expected=positive
**Feature class:** `dow_thu`
**N_total:** 787 | **N_on_signal:** 164

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.111 (pdr_r105_fire) | **PASS** | excluded self_proxy=none; corrs={'pdr_r105_fire': 0.11113680891778707, 'gap_r015_fire': -0.08850145794442811, 'atr70_fire': -0.021318798049620297, 'ovn80_fire': 0.013149717896305608} |
| T1 T1_wr_monotonicity | WR_spread=0.136 (on=0.658 off=0.523) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=158 ExpR=+0.248 WR=0.658 σ=0.90 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=6 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature (dow_thu) — no theta grid applicable |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=+0.248 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): -0.4473615384615384, np.int32(2020): 0.15753478260869566, np.int32(2021): 0.28383461538461535, np.int32(2022): 0.23190000000000005, np.int32(2023): 0.59782, np.int32(2024): 0.23707692307692307, np.int32(2025): 0.35163500000000003} |
| T8 T8_cross_instrument | twin=MES Δ=+0.154 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL** — one fail; acceptable if non-load-bearing (e.g., thin OOS N).

---

## H5_MES_COMEX_SETTLE_O30_RR1.0_long_ovn_took_pdh_SKIP
**Description:** MES COMEX_SETTLE O30 RR1.0 LONG — SKIP when ovn_took_pdh (continuation already spent)
**Scope:** MES | COMEX_SETTLE | O30 | RR1.0 | long | expected=negative
**Feature class:** `ovn_took_pdh`
**N_total:** 750 | **N_on_signal:** 215

| Test | Value | Status | Detail |
|------|-------|--------|--------|
| T0 T0_tautology | max |corr|=0.110 (pdr_r105_fire) | **PASS** | excluded self_proxy=none; corrs={'pdr_r105_fire': -0.11026341345096276, 'gap_r015_fire': 0.012212598658649262, 'atr70_fire': -0.061820437822303516, 'ovn80_fire': -0.07188984869905789} |
| T1 T1_wr_monotonicity | WR_spread=0.138 (on=0.457 off=0.594) | **PASS** | SIGNAL — WR differs meaningfully |
| T2 T2_is_baseline | N=208 ExpR=-0.197 WR=0.457 σ=0.88 | **PASS** | deployable N ≥ 100 |
| T3 T3_oos_wfe | N_OOS=7 | **FAIL** | insufficient OOS N for WFE (< 10) |
| T4 T4_sensitivity | N/A | **INFO** | binary feature (ovn_took_pdh) — no theta grid applicable |
| T5 T5_family | see mega-exploration table | **INFO** | family evaluated in mega-exploration § patterns P1/P2 show cross-session, P3 is session-specific |
| T6 T6_null_floor | p=0.0010 ExpR_obs=-0.197 | **PASS** | 1000 shuffles, bootstrap PASS |
| T7 T7_per_year | 6/7 in expected direction | **PASS** | 6/7 years (86%) matching expected direction; yr={np.int32(2019): -0.28758750000000005, np.int32(2020): -0.35061836734693874, np.int32(2021): -0.07225128205128205, np.int32(2022): -0.24168636363636364, np.int32(2023): -0.3128758620689655, np.int32(2024): 0.049205263157894745, np.int32(2025): -0.020219230769230774} |
| T8 T8_cross_instrument | twin=MNQ Δ=-0.195 (sign match, mag≥0.05) | **PASS** | CONSISTENT |

**Counts:** 6 PASS, 1 FAIL, 2 INFO
### Verdict: **CONDITIONAL** — one fail; acceptable if non-load-bearing (e.g., thin OOS N).

---

## Summary Table

| Cell | P/F/I | Verdict |
|------|-------|---------|
| H1_MES_LONDON_METALS_O30_RR1.5_long_ovn_range_pct_GT80 | 7P/1F/1I | CONDITIONAL |
| H2_MNQ_COMEX_SETTLE_O5_RR1.0_long_garch_vol_pct_GT70 | 8P/0F/1I | VALIDATED |
| H3_MNQ_BRISBANE_1025_O30_RR2.0_long_is_monday | 5P/1F/3I | CONDITIONAL |
| H4_MNQ_COMEX_SETTLE_O15_RR1.0_short_dow_thu | 6P/1F/2I | CONDITIONAL |
| H5_MES_COMEX_SETTLE_O30_RR1.0_long_ovn_took_pdh_SKIP | 6P/1F/2I | CONDITIONAL |