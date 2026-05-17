---
pooled_finding: true
verdict_test: holm_bonferroni_per_cell_clustered_se
pooled_t_role: descriptive_only
per_cell_breakdown_path: docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.per-cell.csv
measured_pooled_power: 0.5956
clustering_inflation_warning: false
flip_rate_pct: 0.00
---

# MNQ US_DATA_1000 VWAP_MID_ALIGNED family-pooled Holm + clustered SE

**Pre-reg:** `docs/audit/hypotheses/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.yaml`
**Per-cell CSV:** `docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.per-cell.csv`
**By-year CSV:** `docs/audit/results/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.by-year.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`
**statsmodels version:** `0.14.6`
**Holdout boundary (Mode A):** `trading_day >= 2026-01-01`
**Cohort lower bound:** `WF_START_OVERRIDE['MNQ']=2020-01-01`

## Verdict

**MEASURED family verdict:** `PASS_FAMILY_HOLM`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `false`

Both gates are necessary; neither subsumes the other. Per-cell clustered t_clustered >= 3.79 (Chordia 2018 no-theory strict t-hurdle, ASCII) AND per-cell holm_adjusted_p_clustered <= alpha'_i (Holm-Bonferroni FWER at K=4, alpha=0.05). Cluster-skew floor N_unique_trading_days >= 30 and ExpR_IS > 0 are both required for any cell to pass.

## Per-cell table (IS, Mode A, clustered SE at trading_day)

| Strategy | N | N_days | Cluster mean | Cluster max | ExpR | Sharpe | t_naive | t_clustered | p_naive | p_clustered | alpha' | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 806 | 806 | 1.0000 | 1 | 0.1416 | 0.1536 | 4.362 | 4.362 | 0.00001 | 0.00001 | 0.0500 | PASS_CELL |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 806 | 806 | 1.0000 | 1 | 0.2075 | 0.1817 | 5.158 | 5.158 | 0.00000 | 0.00000 | 0.0125 | PASS_CELL |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` | 866 | 866 | 1.0000 | 1 | 0.1332 | 0.1512 | 4.450 | 4.450 | 0.00001 | 0.00001 | 0.0250 | PASS_CELL |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 866 | 866 | 1.0000 | 1 | 0.1687 | 0.1621 | 4.772 | 4.772 | 0.00000 | 0.00000 | 0.0167 | PASS_CELL |

## Directional split (IS, RULE 12)

| Strategy | Long N | Long ExpR | Short N | Short ExpR |
|---|---:|---:|---:|---:|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 452 | 0.1044 | 354 | 0.1890 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 452 | 0.1686 | 354 | 0.2572 |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` | 467 | 0.1360 | 399 | 0.1299 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30` | 467 | 0.1886 | 399 | 0.1453 |

## Pooled descriptive (informational; pooled_t_role = descriptive_only)

- pooled N_trades: **3344**
- pooled N_unique_trading_days: **1024**
- pooled ExpR: **0.1623**
- pooled t_clustered: **5.459**
- pooled p_clustered: **0.00000**
- flip_rate_pct (per-cell ExpR sign vs pooled sign): **0.00**
- heterogeneity_ack: **false** (required when flip_rate_pct >= 25)
- measured_pooled_power: **0.5956** (tier `DIRECTIONAL_ONLY`)

## H3 robustness (K=3 re-rank on prior-PASS cells)

- h3_isolation_pass: **true**

| Strategy | K=4 verdict | K=3 alpha' | K=3 verdict |
|---|---|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | PASS_CELL | 0.0500 | PASS_CELL |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | PASS_CELL | 0.0167 | PASS_CELL |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30` | PASS_CELL | 0.0250 | PASS_CELL |

## Reconciliation vs chordia_audit_log.yaml prior t_stat

- All prior-PASS cells reconcile silently (|delta_t| <= 0.10).

## Method notes

- Canonical source only: `orb_outcomes` JOIN `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied.
- Canonical filter delegation: `research.filter_utils.filter_signal(df, 'VWAP_MID_ALIGNED', 'US_DATA_1000')` (definition='orb_mid', verified at runtime).
- Realized-eod scratch policy: `pnl_r` NULL on `outcome='scratch'` rows coerced to 0.0 before any statistic.
- Clustered SE: `statsmodels.regression.linear_model.OLS` intercept-only fit with `cov_type='cluster'`, `cov_kwds={'groups': trading_day}`.
- Holm-Bonferroni ranking: cells sorted ascending on `p_clustered`, alpha'_i applied by rank index.
- OOS power tier: per-cell and pooled via `research.oos_power.one_sample_power` (NCP = d * sqrt(n)); descriptive only (RULE 3.3).
- No writes to `validated_setups`, `experimental_strategies`, `lane_allocation.json`, `chordia_audit_log.yaml`, `bot_state.json`, or `live_config.json`.

## Harvey-Liu boundary statement (mandatory; pre-reg outputs_required_after_run)

The Harvey-Liu Sharpe-haircut deflates IS Sharpe for multiple-testing inflation; it is NOT an OOS validation substitute. Deployment eligibility is decided by the downstream Stage B3 pre-reg, which applies BOTH (a) the Harvey-Liu haircut to IS Sharpe AND (b) allocator correlation gating. Both gates are orthogonal; both must clear independently.

## Caveats

- Pooled t is descriptive_only per BLOCKING-B compensating control (`primary_schema.pooled_t_role_assert == DESCRIPTIVE_ONLY`). It cannot rescue or override the per-cell Holm verdict.
- `PASS_PARTIAL_HOLM` is not a verdict in this audit (K=4 all-or-nothing). 3-of-4 PASS yields `FAIL_FAMILY`.
- OOS is descriptive at audit time (RULE 3.3 power-tier mandate); binary OOS gates are not applicable here.

## Reproduction

```
python research/vwap_mid_family_pooled_oos_v1.py --hypothesis-file docs/audit/hypotheses/2026-05-17-mnq-usdata1000-vwapmid-family-pooled-oos-v1.yaml
```
