# MES E1 rel_vol family v1

**Pre-reg:** `docs/audit/hypotheses/2026-04-22-mes-e1-rel-vol-family-v1.yaml` (locked at commit `6910c598`)
**Script:** `research/verify_mes_e1_rel_vol.py`
**Scope:** MES | 10 sessions x 2 directions | O5 | E1 | RR1.5 | CB1 | IS-only per-cell Q80 rel_vol threshold
**IS boundary:** `trading_day < 2026-01-01`
**BH family K:** `20`

## Verdict: **KILL**

> K1 fired: 0/20 cells survived BH q<0.05, t>=3.79, and positive-mean floor

## Integrity

- Entry model admitted: `E1` only
- E2 / E3 admitted rows: `0`
- Quantile source: `IS-only per cell`
- break_dir vs derived-direction mismatches: `0`

## Family table

| session | dir | Q80 | N_IS | N_on | ExpR_on_IS | ExpR_off_IS | Δ_IS | t_IS | p_IS | p_BH | BH | t>=3.79 | mean>0 | surv | N_OOS_on | ExpR_on_OOS | ExpR_off_OOS | Δ_OOS | t_OOS | p_OOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| CME_PRECLOSE | long | 1.6605 | 498 | 100 | -0.1760 | -0.1056 | -0.0703 | -0.583 | 0.560612 | 0.915211 | N | N | N | N | 7 | -0.3346 | -0.6101 | +0.2754 | +0.576 | 0.578897 |
| CME_PRECLOSE | short | 1.7304 | 490 | 98 | -0.2059 | -0.2679 | +0.0620 | +0.506 | 0.613337 | 0.915211 | N | N | N | N | 8 | -0.7190 | -0.2184 | -0.5005 | -1.236 | 0.231991 |
| CME_REOPEN | long | 3.6988 | 343 | 69 | -0.1652 | -0.3139 | +0.1487 | +1.013 | 0.313775 | 0.819223 | N | N | N | N | 3 | -0.1882 | -0.0517 | -0.1366 | -0.161 | 0.884715 |
| CME_REOPEN | short | 4.3051 | 371 | 75 | -0.2180 | -0.2687 | +0.0507 | +0.375 | 0.708663 | 0.915211 | N | N | N | N | 6 | -0.2354 | -0.2301 | -0.0053 | -0.010 | 0.992449 |
| COMEX_SETTLE | long | 2.2090 | 878 | 176 | -0.0054 | -0.2133 | +0.2079 | +2.278 | 0.023577 | 0.235766 | N | N | N | N | 9 | +0.0493 | -0.6334 | +0.6827 | +1.545 | 0.152656 |
| COMEX_SETTLE | short | 2.6855 | 750 | 150 | -0.2046 | -0.2252 | +0.0206 | +0.213 | 0.831542 | 0.915211 | N | N | N | N | 5 | +0.3627 | -0.0583 | +0.4209 | +0.696 | 0.514575 |
| LONDON_METALS | long | 2.7289 | 872 | 175 | -0.1431 | -0.1530 | +0.0099 | +0.111 | 0.912079 | 0.915211 | N | N | N | N | 7 | +0.3217 | -0.2794 | +0.6010 | +1.200 | 0.265134 |
| LONDON_METALS | short | 2.6343 | 840 | 168 | -0.0825 | -0.2270 | +0.1445 | +1.570 | 0.117675 | 0.470701 | N | N | N | N | 9 | +0.0078 | -0.0913 | +0.0991 | +0.217 | 0.831837 |
| NYSE_CLOSE | long | 2.1451 | 260 | 52 | -0.2278 | -0.2464 | +0.0186 | +0.117 | 0.907141 | 0.915211 | N | N | N | N | 3 | -1.0000 | +0.1238 | -1.1238 | -2.995 | 0.015084 |
| NYSE_CLOSE | short | 2.8550 | 247 | 50 | -0.2043 | -0.4042 | +0.1999 | +1.197 | 0.235172 | 0.783905 | N | N | N | N | 1 | -1.0000 | -0.2008 | -0.7993 | NA | NA |
| NYSE_OPEN | long | 1.8617 | 831 | 167 | +0.0249 | -0.0667 | +0.0916 | +0.922 | 0.357380 | 0.819223 | N | N | Y | N | 8 | -0.1096 | +0.0987 | -0.2083 | -0.416 | 0.684896 |
| NYSE_OPEN | short | 2.0642 | 827 | 166 | -0.0603 | -0.0841 | +0.0238 | +0.241 | 0.810136 | 0.915211 | N | N | N | N | 10 | +0.1823 | -0.2934 | +0.4757 | +1.061 | 0.305931 |
| SINGAPORE_OPEN | long | 3.4173 | 890 | 178 | -0.1883 | -0.2439 | +0.0556 | +0.655 | 0.513097 | 0.915211 | N | N | N | N | 4 | +0.7406 | -0.1880 | +0.9286 | +1.533 | 0.208782 |
| SINGAPORE_OPEN | short | 3.5579 | 826 | 166 | -0.2770 | -0.2862 | +0.0092 | +0.107 | 0.915211 | 0.915211 | N | N | N | N | 7 | -0.0106 | -0.2371 | +0.2264 | +0.443 | 0.668669 |
| TOKYO_OPEN | long | 3.2484 | 841 | 169 | -0.0244 | -0.1995 | +0.1751 | +1.943 | 0.053217 | 0.354780 | N | N | N | N | 10 | +0.3567 | +0.1676 | +0.1891 | +0.444 | 0.663055 |
| TOKYO_OPEN | short | 3.2841 | 875 | 175 | -0.0158 | -0.2399 | +0.2241 | +2.544 | 0.011557 | 0.231147 | N | N | N | N | 6 | +0.0833 | -0.3565 | +0.4398 | +0.841 | 0.429194 |
| US_DATA_1000 | long | 2.0582 | 848 | 170 | -0.1756 | -0.0903 | -0.0853 | -0.901 | 0.368650 | 0.819223 | N | N | N | N | 10 | -0.0504 | -0.0279 | -0.0225 | -0.046 | 0.963494 |
| US_DATA_1000 | short | 2.4948 | 802 | 161 | -0.1091 | -0.0871 | -0.0221 | -0.222 | 0.824170 | 0.915211 | N | N | N | N | 8 | +0.4926 | -0.3010 | +0.7936 | +1.653 | 0.129080 |
| US_DATA_830 | long | 2.8803 | 824 | 165 | -0.2051 | -0.1699 | -0.0352 | -0.379 | 0.704740 | 0.915211 | N | N | N | N | 7 | -0.0159 | -0.1818 | +0.1659 | +0.325 | 0.752589 |
| US_DATA_830 | short | 3.0807 | 814 | 163 | -0.0087 | -0.1682 | +0.1595 | +1.656 | 0.099133 | 0.470701 | N | N | N | N | 6 | -0.5974 | -0.4481 | -0.1493 | -0.337 | 0.745517 |

## Survivors

- None

## Sub-threshold +0.4R OOS cells

These cells printed roughly `+0.4R` or better on OOS `delta_oos`, but they do **not** survive the family decision rule and are not promotion-safe.

- `SINGAPORE_OPEN long` | Δ_IS=+0.0556 | t_IS=+0.655 | p_BH=0.915211 | Δ_OOS=+0.9286
- `US_DATA_1000 short` | Δ_IS=-0.0221 | t_IS=-0.222 | p_BH=0.915211 | Δ_OOS=+0.7936
- `COMEX_SETTLE long` | Δ_IS=+0.2079 | t_IS=+2.278 | p_BH=0.235766 | Δ_OOS=+0.6827
- `LONDON_METALS long` | Δ_IS=+0.0099 | t_IS=+0.111 | p_BH=0.915211 | Δ_OOS=+0.6010
- `NYSE_OPEN short` | Δ_IS=+0.0238 | t_IS=+0.241 | p_BH=0.915211 | Δ_OOS=+0.4757
- `TOKYO_OPEN short` | Δ_IS=+0.2241 | t_IS=+2.544 | p_BH=0.231147 | Δ_OOS=+0.4398
- `COMEX_SETTLE short` | Δ_IS=+0.0206 | t_IS=+0.213 | p_BH=0.915211 | Δ_OOS=+0.4209

## Limitations

- Family verdict governs. A few positive OOS deltas do not rescue a 0-survivor family.
- No cell clears the combined BH q<0.05, t>=3.79, and positive-mean floor gate.
- This is an execution-safe reroute test for `E1` only, not a reopen of broad pooled ORB ML.
- OOS on-signal counts remain thin in several cells, so descriptive OOS wins are not enough for promotion.

## Audit artifacts

- Cell metrics CSV: `research/output/mes_e1_rel_vol_family_v1_metrics.csv`
- Row flags CSV: `research/output/mes_e1_rel_vol_family_v1_row_flags.csv`

## Reproduction

```bash
./.venv-wsl/bin/python research/verify_mes_e1_rel_vol.py
```

Read-only canonical run. No writes to `validated_setups`, `experimental_strategies`, `live_config`, or `lane_allocation.json`.
