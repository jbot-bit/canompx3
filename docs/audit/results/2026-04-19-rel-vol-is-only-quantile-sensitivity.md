# Rel_vol_HIGH_Q3 IS-only quantile sensitivity — 13 BH-global survivors

**Generated:** 2026-04-19
**Script:** `research/rel_vol_is_only_quantile_sensitivity.py`
**Parent scan:** `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`
**IS boundary:** `trading_day < 2026-01-01` (Mode A)

## What this audit tests

The parent scan computed rel_vol's 67th percentile on the FULL-sample lane
(IS + OOS rows) and bucketed HIGH_Q3 using that threshold. This audit:

1. Loads each BH-global survivor cell's full lane data.
2. Computes BOTH the full-sample 67th percentile (original) and the IS-only
   67th percentile.
3. Restricts to IS rows and evaluates rel_vol_HIGH_Q3 under each threshold.
4. Flags `survivor_drift = Y` if the cell clears |t| >= 4 (BH-global proxy)
   under one threshold but not the other.

If `survivor_drift = N` across all 13 cells, the 13-survivor narrative holds
under honest IS-only quantile computation. If any cells drift, the 2026-04-15
scan result doc needs an addendum.

## Per-cell results

| Instr | Session | O | RR | Dir | N_IS | thresh_full | thresh_IS | Δ | t_full | t_IS | drift |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| MES | COMEX_SETTLE | 5 | 1.0 | short | 759 | 2.0339 | 2.0382 | +0.0043 | +4.48 | +4.48 | N |
| MGC | LONDON_METALS | 5 | 1.0 | short | 445 | 2.1226 | 2.0529 | -0.0697 | +4.43 | +4.65 | N |
| MES | TOKYO_OPEN | 5 | 1.5 | long | 841 | 2.3289 | 2.3411 | +0.0122 | +4.42 | +4.37 | N |
| MNQ | SINGAPORE_OPEN | 5 | 1.0 | short | 835 | 2.3871 | 2.3781 | -0.0090 | +4.30 | +4.29 | N |
| MES | COMEX_SETTLE | 5 | 1.5 | short | 753 | 2.0068 | 2.0232 | +0.0165 | +3.68 | +3.68 | N |
| MES | SINGAPORE_OPEN | 5 | 1.0 | short | 826 | 2.5170 | 2.5070 | -0.0099 | +3.86 | +3.81 | N |
| MES | SINGAPORE_OPEN | 5 | 1.0 | long | 890 | 2.3677 | 2.3929 | +0.0252 | +4.23 | +4.02 | N |
| MNQ | CME_PRECLOSE | 5 | 1.0 | short | 708 | 1.4465 | 1.4398 | -0.0067 | +3.47 | +3.68 | N |
| MES | SINGAPORE_OPEN | 5 | 1.5 | long | 890 | 2.3677 | 2.3929 | +0.0252 | +3.90 | +3.75 | N |
| MES | CME_PRECLOSE | 5 | 1.0 | short | 702 | 1.4411 | 1.4273 | -0.0139 | +3.47 | +3.66 | N |
| MNQ | BRISBANE_1025 | 5 | 1.0 | long | 905 | 2.0663 | 2.0607 | -0.0056 | +3.90 | +3.81 | N |
| MES | CME_PRECLOSE | 5 | 1.5 | short | 607 | 1.4353 | 1.4246 | -0.0106 | +3.08 | +3.52 | N |
| MGC | LONDON_METALS | 5 | 1.5 | short | 445 | 2.1226 | 2.0529 | -0.0697 | +3.21 | +3.35 | N |

## Summary

- Cells audited: 13
- Cells with survivor drift: **0**

**Verdict:** All cells retain BH-global-proxy status under IS-only quantile. The 13-survivor narrative from the 2026-04-15 comprehensive scan holds under honest IS-only threshold computation. No addendum required to the parent doc's survivor list.

## Reproduction
```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/rel_vol_is_only_quantile_sensitivity.py
```

Read-only. No writes.