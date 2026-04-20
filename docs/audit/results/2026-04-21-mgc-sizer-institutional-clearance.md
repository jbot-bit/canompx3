# MGC Sizer Institutional Clearance

Scope-locked to the rel_vol sizer lineage only. Canonical inputs: `orb_outcomes`, `daily_features`, repo gate formulas in `pre_registered_criteria.md`, `walkforward.py`, and `dsr.py`.

## MGC institutional gate walk


| Gate | Canonical number(s) | Binding? | Status |
|---|---|---|---|
| Pre-reg scoped gate | delta=0.03175, t=2.000, p=0.0230, bootstrap95=[0.0011, 0.0626] | pre-reg gate | PASS |
| Sharpe-positive gate | uniform_SR=0.059, sizer_SR=0.081 using SR = mean(pnl_sizer) / std(pnl_sizer, ddof=1) on 2026 OOS | non-waivable Pathway B direction gate | PASS |
| Walk-forward efficiency | WFE=NA because IS_ExpR=-0.10435 <= 0 while OOS_ExpR=0.10130; canonical walkforward.py fails closed when weighted_IS <= 0 | non-waivable Pathway B | FAIL |
| Era stability | 2020-2022: N=1162 sizer=-0.14682; 2023: N=2061 sizer=-0.15956; 2024-2025: N=4262 sizer=-0.06606; 2026: N=601 sizer=0.10130 | non-waivable Pathway B | FAIL |
| 2026 OOS residual untouched since `d227f8ed` | commit_day=2026-04-21, max_data_day=2026-04-16, untouched_distinct_days=0, range=[None, None] | monitor readiness check | FAIL |
| DSR cross-check | repo-canonical n_eff=21, var_sr_E2=0.006712, sr_hat_OOS=0.081143, sr0=0.157507, dsr=0.029276; local K=3 sensitivity dsr=0.610087 | cross-check only (Amendment 2.1) | FAIL |
| MinBTL | required_years=0.244, available_pre_holdout_days=937, available_years=3.718 | pre-reg integrity | PASS |

**Final status: `REJECT`**

Reason: the original OOS pre-reg pass stands, but the broader institutional stack does not clear. The non-waivable failures are `WFE` (undefined / fail-closed because IS sizer ExpR is negative), `era stability` (three load-bearing eras below -0.05 with N>=50), and `untouched residual OOS` (0 days since the pre-reg commit). DSR is also weak, but is cross-check only.

## MGC raw year-by-year breakdown

| Year | Split | N | Uniform ExpR | Sizer ExpR | Delta |
|---|---|---:|---:|---:|---:|
| 2022 | IS | 1162 | -0.18069 | -0.14682 | 0.03386 |
| 2023 | IS | 2061 | -0.18891 | -0.15956 | 0.02935 |
| 2024 | IS | 2106 | -0.17208 | -0.13952 | 0.03256 |
| 2025 | IS | 2156 | -0.01977 | 0.00569 | 0.02545 |
| 2026 | OOS | 601 | 0.06955 | 0.10130 | 0.03175 |

## MGC Criterion 9 era bins

| Era | N | Uniform ExpR | Sizer ExpR | Delta |
|---|---:|---:|---:|---:|
| 2020-2022 | 1162 | -0.18069 | -0.14682 | 0.03386 |
| 2023 | 2061 | -0.18891 | -0.15956 | 0.02935 |
| 2024-2025 | 4262 | -0.09503 | -0.06606 | 0.02897 |
| 2026 | 601 | 0.06955 | 0.10130 | 0.03175 |

## MES sizer closeout

`NOT_DEPLOYABLE_AS_SIZER`. Raw OOS numbers: delta=0.03025, t=2.084, p=0.0188, uniform_SR=-0.082, sizer_SR=-0.050. The sizer reduces losses but both Sharpe values stay negative. That is risk reduction on a losing lane, not alpha. The live MES path is the filter-form pre-reg locked at `docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml` and gated on fresh OOS only.

## MNQ sizer closeout

`DEAD`. Raw OOS numbers: delta=0.00627, t=0.440, p=0.3302, bootstrap95=[-0.0220, 0.0339], Spearman p=0.1161, lane_pos=10, lane_neg=12, neg_share=0.545. This is heterogeneity/noise, not a deployable linear sizer. Possible Q4-peak/Q5-crash biphasic hypothesis belongs to the Claude terminal, not this scope.

## Filter-form lock verification

Verified in `docs/audit/hypotheses/2026-04-21-rel-vol-filter-form-v1.yaml`: `oos_peeked_window`, `fresh_oos_window`, and `>=50 filter-fired trades per instrument` fresh-OOS gate are present. No filter-form execution was run on the contaminated 2026-01-01..2026-04-19 OOS window in this task.
