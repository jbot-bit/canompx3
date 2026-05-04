# HTF Path A — prev-week × prev-month overlap decomposition

**Generated:** 2026-04-20T11:49:07+00:00
**Script:** `research/htf_path_a_overlap_decomposition.py`
**IS window:** `trading_day < 2026-01-01` (Mode A, from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`)
**Cell axes:** entry_model=E2, confirm_bars=1, orb_minutes=15, direction=long, rr=2.0

## Purpose

Reproducible verification of the numbers cited in `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` § 'Adversarial-audit addendum — 2026-04-19'. Closes review-finding #1 (unreproducible quantitative claims) from the 2026-04-19 self-review.

## MES EUROPE_FLOW long RR2.0

- prev-week v1 fires (unique trading-days, IS): **213**
- prev-month v1 fires (unique trading-days, IS): **341**
- OVERLAP days (PM ∧ PW): **146**
- PM-only days (PM ∧ ¬PW): **195**
- PW-only days (PW ∧ ¬PM): **67**
- Overlap as % of PM fires: **42.8%**

| Subset | N | mean pnl_r | std | t | raw p |
|---|---:|---:|---:|---:|---:|
| OVERLAP (PM ∧ PW) | 146 | -0.353 | 1.062 | -4.018 | 0.0001 |
| NON-OVERLAP (PM ∧ ¬PW) | 195 | -0.119 | 1.200 | -1.384 | 0.1678 |
| COMBINED (PM) | 341 | -0.219 | 1.147 | -3.529 | 0.0005 |

## MES TOKYO_OPEN long RR2.0

- prev-week v1 fires (unique trading-days, IS): **181**
- prev-month v1 fires (unique trading-days, IS): **309**
- OVERLAP days (PM ∧ PW): **121**
- PM-only days (PM ∧ ¬PW): **188**
- PW-only days (PW ∧ ¬PM): **60**
- Overlap as % of PM fires: **39.2%**

| Subset | N | mean pnl_r | std | t | raw p |
|---|---:|---:|---:|---:|---:|
| OVERLAP (PM ∧ PW) | 121 | -0.193 | 1.140 | -1.863 | 0.0649 |
| NON-OVERLAP (PM ∧ ¬PW) | 188 | -0.275 | 1.121 | -3.367 | 0.0009 |
| COMBINED (PM) | 309 | -0.243 | 1.128 | -3.791 | 0.0002 |

---

## Methodology

- Predicates copied verbatim from the long-direction branch of the two v1 scans: `research/htf_path_a_prev_week_v1_scan.py::_predicate_sql` and `research/htf_path_a_prev_month_v1_scan.py::_predicate_sql` (direction='long').
- Trade loader uses the same JOIN and filter shape as the v1 scans' `_load_cell_trades`, restricted to IS window `trading_day < HOLDOUT_SACRED_FROM`.
- OVERLAP = trading_days present in BOTH predicate fire-sets. NON-OVERLAP = PM-fire days NOT in PW-fire set. COMBINED = full PM-on set.
- Statistics: one-sample two-tailed t-test vs 0 on per-trade `pnl_r`. Formula matches `research/htf_path_a_prev_month_v1_scan.py::_t_test` (t = mean / (std / sqrt(n)); raw p = 2·(1 − scipy.stats.t.cdf(|t|, n−1))).

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/htf_path_a_overlap_decomposition.py
```

Writes the result markdown idempotently. Re-running on the same DB state re-creates the same numbers exactly (canonical IS window, no randomness).

## Interpretation

The addendum's claim is: MES EUROPE_FLOW long RR2.0 is OVERLAP-driven (prev-month v1 is not independent evidence once prev-week v1 fires are stripped out); MES TOKYO_OPEN long RR2.0 is NON-OVERLAP-driven (a genuinely new observation vs prev-week v1). See each cell's | t | row above to verify.

