# HTF Path A prev-month v1 — Family Scan Results

**Generated:** 2026-04-20T11:49:08+00:00
**Scan script HEAD SHA:** `39cfc98dd23e405e7afda652db7e0eaeb4d6221f`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`
**Holdout (Mode A):** `trading_day >= 2026-01-01` — imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`
**Pre-registration:** `docs/audit/hypotheses/2026-04-18-htf-path-a-prev-month-v1.yaml`

## Scope

- Instruments: ['MNQ', 'MES']
- Sessions: ['TOKYO_OPEN', 'EUROPE_FLOW', 'NYSE_OPEN']
- Aperture: O15
- RR targets: [1.5, 2.0]
- Directions / cells: ['long', 'short'] (pmh_break_long / pml_break_short)
- Entry model: E2 confirm_bars=1
- **Total cells:** 24

## BH-FDR framings (informational)

| Framing | K | Role |
|---|---:|---|
| K_global     | 24     | Informational — equal to K_family under single-family run |
| K_family     | 24     | **PRIMARY promotion gate** (q < 0.05) |
| K_instrument | 12 | Informational — per instrument across sess×dir×RR |
| K_session    | 8    | Informational — per session across inst×dir×RR |
| K_direction  | 12  | Informational — per direction across inst×sess×RR |

## Family verdict

**FAMILY KILL (FK1: zero cells pass all gates + BH_family)**

## Per-cell stats

| # | inst | session | dir | RR | N_is_base | N_is_on | fire% | ExpR_base_IS | ExpR_on_IS | ΔIS | ExpR_on_OOS | ΔOOS | dir_match | t | raw p | q_family | q_inst | q_sess | q_dir |
|--|-----|---------|-----|----|----:|----:|----:|----:|----:|----:|----:|----:|:--:|----:|----:|----:|----:|----:|----:|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | 870 | 304 | 34.9 | 0.098 | 0.067 | -0.032 | 1.378 | 1.246 | — | 1.04 | 0.3011 | 0.615 | 0.799 | 0.333 | 0.571 |
| 2 | MNQ | TOKYO_OPEN | long | 2 | 870 | 304 | 34.9 | 0.116 | 0.073 | -0.042 | 1.854 | 1.496 | — | 0.97 | 0.3331 | 0.615 | 0.799 | 0.333 | 0.571 |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | 851 | 82 | 9.6 | 0.012 | 0.219 | 0.207 | 0.385 | 0.437 | True | 1.65 | 0.1020 | 0.350 | 0.799 | 0.204 | 0.408 |
| 4 | MNQ | TOKYO_OPEN | short | 2 | 851 | 82 | 9.6 | -0.034 | 0.224 | 0.258 | 0.662 | 0.698 | True | 1.42 | 0.1597 | 0.479 | 0.799 | 0.245 | 0.442 |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | 876 | 312 | 35.6 | 0.067 | -0.027 | -0.094 | -1.000 | -1.234 | — | -0.43 | 0.6652 | 0.840 | 0.885 | 0.885 | 0.726 |
| 6 | MNQ | EUROPE_FLOW | long | 2 | 876 | 312 | 35.6 | 0.072 | -0.016 | -0.088 | -1.000 | -1.151 | — | -0.22 | 0.8279 | 0.903 | 0.885 | 0.885 | 0.828 |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | 843 | 82 | 9.7 | -0.001 | -0.039 | -0.038 | 0.366 | 0.176 | False | -0.30 | 0.7667 | 0.876 | 0.885 | 0.885 | 0.920 |
| 8 | MNQ | EUROPE_FLOW | short | 2 | 843 | 82 | 9.7 | 0.004 | -0.022 | -0.026 | 0.223 | 0.248 | False | -0.15 | 0.8848 | 0.923 | 0.885 | 0.885 | 0.965 |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | 880 | 323 | 36.7 | 0.058 | 0.047 | -0.010 | — | — | — | 0.61 | 0.5434 | 0.827 | 0.885 | 0.825 | 0.726 |
| 10 | MNQ | NYSE_OPEN | long | 2 | 880 | 323 | 36.7 | -0.019 | -0.050 | -0.031 | — | — | — | -0.53 | 0.5946 | 0.827 | 0.885 | 0.825 | 0.726 |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | 835 | 91 | 10.9 | 0.078 | -0.072 | -0.150 | 0.238 | -0.265 | True | -0.55 | 0.5871 | 0.827 | 0.885 | 0.825 | 0.881 |
| 12 | MNQ | NYSE_OPEN | short | 2 | 835 | 91 | 10.9 | 0.039 | -0.155 | -0.194 | 0.485 | -0.144 | True | -1.02 | 0.3097 | 0.615 | 0.799 | 0.825 | 0.619 |
| 13 | MES | TOKYO_OPEN | long | 1.5 | 856 | 309 | 36.1 | -0.055 | -0.205 | -0.151 | 0.520 | 0.419 | — | -3.63 | 0.0003 | 0.003 | 0.001 | 0.001 | 0.001 |
| 14 | MES | TOKYO_OPEN | long | 2 | 856 | 309 | 36.1 | -0.031 | -0.243 | -0.212 | 0.824 | 0.576 | — | -3.79 | 0.0002 | 0.002 | 0.001 | 0.001 | 0.001 |
| 15 | MES | TOKYO_OPEN | short | 1.5 | 863 | 83 | 9.6 | -0.142 | 0.167 | 0.309 | 0.329 | 0.406 | True | 1.34 | 0.1840 | 0.491 | 0.315 | 0.245 | 0.442 |
| 16 | MES | TOKYO_OPEN | short | 2 | 863 | 83 | 9.6 | -0.173 | 0.337 | 0.509 | 0.595 | 0.487 | True | 2.25 | 0.0274 | 0.109 | 0.055 | 0.073 | 0.164 |
| 17 | MES | EUROPE_FLOW | long | 1.5 | 888 | 344 | 38.7 | -0.091 | -0.211 | -0.120 | -1.000 | -0.968 | — | -3.92 | 0.0001 | 0.002 | 0.001 | 0.001 | 0.001 |
| 18 | MES | EUROPE_FLOW | long | 2 | 888 | 344 | 38.7 | -0.076 | -0.219 | -0.143 | -1.000 | -1.161 | — | -3.53 | 0.0005 | 0.003 | 0.001 | 0.002 | 0.001 |
| 19 | MES | EUROPE_FLOW | short | 1.5 | 831 | 81 | 9.7 | -0.162 | -0.103 | 0.059 | 0.342 | 0.633 | True | -0.83 | 0.4092 | 0.702 | 0.546 | 0.818 | 0.702 |
| 20 | MES | EUROPE_FLOW | short | 2 | 831 | 81 | 9.7 | -0.171 | -0.306 | -0.135 | -0.211 | 0.174 | False | -2.32 | 0.0227 | 0.109 | 0.054 | 0.060 | 0.164 |
| 21 | MES | NYSE_OPEN | long | 1.5 | 878 | 344 | 39.2 | 0.063 | -0.032 | -0.095 | -0.235 | 0.115 | — | -0.50 | 0.6205 | 0.827 | 0.745 | 0.825 | 0.726 |
| 22 | MES | NYSE_OPEN | long | 2 | 878 | 344 | 39.2 | 0.036 | -0.083 | -0.120 | -0.082 | 0.138 | — | -1.08 | 0.2828 | 0.615 | 0.424 | 0.825 | 0.571 |
| 23 | MES | NYSE_OPEN | short | 1.5 | 840 | 96 | 11.4 | 0.038 | 0.045 | 0.007 | 0.730 | 0.481 | True | 0.36 | 0.7222 | 0.867 | 0.788 | 0.825 | 0.920 |
| 24 | MES | NYSE_OPEN | short | 2 | 840 | 96 | 11.4 | -0.003 | -0.002 | 0.001 | 0.938 | 0.605 | True | -0.01 | 0.9904 | 0.990 | 0.990 | 0.990 | 0.990 |

## WFE, era stability, flags

| # | inst | session | dir | RR | Sharpe_IS | Sharpe_OOS | WFE | era_stable | tautology corr | tautology? | arithmetic_only |
|--|-----|---------|-----|----|----:|----:|----:|:--:|----:|:--:|:--:|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | 0.059 | — | — | False | -0.148 | False | False |
| 2 | MNQ | TOKYO_OPEN | long | 2 | 0.056 | — | — | True | -0.148 | False | False |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | 0.183 | 0.297 | 1.628 | True | 0.230 | False | False |
| 4 | MNQ | TOKYO_OPEN | short | 2 | 0.157 | 0.426 | 2.718 | True | 0.230 | False | False |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | -0.025 | — | — | False | -0.171 | False | False |
| 6 | MNQ | EUROPE_FLOW | long | 2 | -0.012 | — | — | False | -0.171 | False | False |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | -0.033 | 0.287 | — | True | 0.285 | False | False |
| 8 | MNQ | EUROPE_FLOW | short | 2 | -0.016 | 0.146 | — | True | 0.285 | False | False |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | 0.039 | — | — | False | -0.188 | False | False |
| 10 | MNQ | NYSE_OPEN | long | 2 | -0.037 | — | — | False | -0.188 | False | False |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | -0.060 | 0.166 | — | True | 0.174 | False | True |
| 12 | MNQ | NYSE_OPEN | short | 2 | -0.117 | 0.283 | — | True | 0.174 | False | True |
| 13 | MES | TOKYO_OPEN | long | 1.5 | -0.207 | 0.394 | — | False | -0.164 | False | False |
| 14 | MES | TOKYO_OPEN | long | 2 | -0.216 | 0.521 | — | False | -0.164 | False | False |
| 15 | MES | TOKYO_OPEN | short | 1.5 | 0.147 | 0.264 | 1.798 | True | 0.237 | False | False |
| 16 | MES | TOKYO_OPEN | short | 2 | 0.247 | 0.398 | 1.616 | True | 0.237 | False | False |
| 17 | MES | EUROPE_FLOW | long | 1.5 | -0.211 | — | — | False | -0.188 | False | False |
| 18 | MES | EUROPE_FLOW | long | 2 | -0.191 | — | — | False | -0.188 | False | False |
| 19 | MES | EUROPE_FLOW | short | 1.5 | -0.092 | 0.272 | — | True | 0.271 | False | False |
| 20 | MES | EUROPE_FLOW | short | 2 | -0.258 | -0.157 | — | True | 0.271 | False | False |
| 21 | MES | NYSE_OPEN | long | 1.5 | -0.029 | -0.178 | — | False | -0.233 | False | False |
| 22 | MES | NYSE_OPEN | long | 2 | -0.064 | -0.052 | — | False | -0.233 | False | True |
| 23 | MES | NYSE_OPEN | short | 1.5 | 0.038 | 0.618 | 16.431 | True | 0.271 | False | False |
| 24 | MES | NYSE_OPEN | short | 2 | -0.001 | 0.625 | — | True | 0.271 | False | False |

## Era stability detail (per cell, IS-on trades)

| # | inst | session | dir | RR | 2019-2020 (n, ExpR) | 2021-2022 (n, ExpR) | 2023 (n, ExpR) | 2024-2025 (n, ExpR) |
|--|-----|---------|-----|----|----:|----:|----:|----:|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | n=90, ExpR=-0.064 | n=66, ExpR=0.193 | n=49, ExpR=-0.193* | n=99, ExpR=0.230 |
| 2 | MNQ | TOKYO_OPEN | long | 2 | n=90, ExpR=0.005 | n=66, ExpR=0.139 | n=49, ExpR=-0.142* | n=99, ExpR=0.198 |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | n=16, ExpR=-0.251* | n=37, ExpR=0.345* | n=5, ExpR=0.407* | n=24, ExpR=0.298* |
| 4 | MNQ | TOKYO_OPEN | short | 2 | n=16, ExpR=-0.271* | n=37, ExpR=0.312* | n=5, ExpR=0.135* | n=24, ExpR=0.435* |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | n=87, ExpR=0.002 | n=65, ExpR=0.039 | n=55, ExpR=-0.137 | n=105, ExpR=-0.035 |
| 6 | MNQ | EUROPE_FLOW | long | 2 | n=87, ExpR=0.004 | n=65, ExpR=0.142 | n=55, ExpR=-0.054 | n=105, ExpR=-0.109 |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | n=15, ExpR=-0.208* | n=40, ExpR=-0.339* | n=1, ExpR=1.234* | n=26, ExpR=0.472* |
| 8 | MNQ | EUROPE_FLOW | short | 2 | n=15, ExpR=-0.243* | n=40, ExpR=-0.351* | n=1, ExpR=1.681* | n=26, ExpR=0.547* |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | n=100, ExpR=-0.122 | n=78, ExpR=0.098 | n=48, ExpR=0.021* | n=97, ExpR=0.171 |
| 10 | MNQ | NYSE_OPEN | long | 2 | n=100, ExpR=-0.159 | n=78, ExpR=0.002 | n=48, ExpR=-0.147* | n=97, ExpR=0.058 |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | n=25, ExpR=-0.190* | n=35, ExpR=-0.153* | n=7, ExpR=-0.019* | n=24, ExpR=0.174* |
| 12 | MNQ | NYSE_OPEN | short | 2 | n=25, ExpR=-0.358* | n=35, ExpR=-0.387* | n=7, ExpR=0.178* | n=24, ExpR=0.331* |
| 13 | MES | TOKYO_OPEN | long | 1.5 | n=82, ExpR=-0.158 | n=76, ExpR=-0.089 | n=45, ExpR=-0.221* | n=106, ExpR=-0.318 |
| 14 | MES | TOKYO_OPEN | long | 2 | n=82, ExpR=-0.197 | n=76, ExpR=-0.175 | n=45, ExpR=-0.270* | n=106, ExpR=-0.316 |
| 15 | MES | TOKYO_OPEN | short | 1.5 | n=22, ExpR=-0.069* | n=28, ExpR=0.307* | n=9, ExpR=0.383* | n=24, ExpR=0.140* |
| 16 | MES | TOKYO_OPEN | short | 2 | n=22, ExpR=0.004* | n=28, ExpR=0.469* | n=9, ExpR=0.660* | n=24, ExpR=0.367* |
| 17 | MES | EUROPE_FLOW | long | 1.5 | n=93, ExpR=-0.165 | n=80, ExpR=-0.223 | n=52, ExpR=-0.219 | n=119, ExpR=-0.235 |
| 18 | MES | EUROPE_FLOW | long | 2 | n=93, ExpR=-0.218 | n=80, ExpR=-0.225 | n=52, ExpR=-0.195 | n=119, ExpR=-0.227 |
| 19 | MES | EUROPE_FLOW | short | 1.5 | n=18, ExpR=-0.360* | n=30, ExpR=-0.144* | n=7, ExpR=0.172* | n=26, ExpR=0.048* |
| 20 | MES | EUROPE_FLOW | short | 2 | n=18, ExpR=-0.386* | n=30, ExpR=-0.539* | n=7, ExpR=0.052* | n=26, ExpR=-0.077* |
| 21 | MES | NYSE_OPEN | long | 1.5 | n=87, ExpR=-0.099 | n=93, ExpR=-0.039 | n=51, ExpR=0.218 | n=113, ExpR=-0.102 |
| 22 | MES | NYSE_OPEN | long | 2 | n=87, ExpR=-0.138 | n=93, ExpR=-0.162 | n=51, ExpR=0.157 | n=113, ExpR=-0.101 |
| 23 | MES | NYSE_OPEN | short | 1.5 | n=31, ExpR=-0.038* | n=31, ExpR=0.037* | n=11, ExpR=-0.146* | n=23, ExpR=0.273* |
| 24 | MES | NYSE_OPEN | short | 2 | n=31, ExpR=-0.103* | n=31, ExpR=-0.138* | n=11, ExpR=0.025* | n=23, ExpR=0.314* |

*exempt = N < 50 (Criterion 9 threshold)*

## Verdict per cell

| # | inst | session | dir | RR | verdict |
|--|-----|---------|-----|----|---|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.30107372847223046); |t|<3.79 (1.0359023284644642); WFE_none; era_unstable; BH_family_fail |
| 2 | MNQ | TOKYO_OPEN | long | 2 | KILL: raw_p>=0.05 (0.3331238028975352); |t|<3.79 (0.9693899829189487); WFE_none; BH_family_fail |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.10204473799560287); |t|<3.79 (1.6537706894282234); WFE>0.95 LEAKAGE_SUSPECT (1.628); BH_family_fail |
| 4 | MNQ | TOKYO_OPEN | short | 2 | KILL: raw_p>=0.05 (0.15970671537175818); |t|<3.79 (1.4190996225764838); WFE>0.95 LEAKAGE_SUSPECT (2.718); BH_family_fail |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | KILL: raw_p>=0.05 (0.6652115971922123); |t|<3.79 (-0.4331423624127495); WFE_none; era_unstable; BH_family_fail |
| 6 | MNQ | EUROPE_FLOW | long | 2 | KILL: raw_p>=0.05 (0.8279191301610382); |t|<3.79 (-0.2175553929295595); WFE_none; era_unstable; BH_family_fail |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | KILL: raw_p>=0.05 (0.7667143292444858); |t|<3.79 (-0.2976737473435546); dir_mismatch; WFE_none; BH_family_fail |
| 8 | MNQ | EUROPE_FLOW | short | 2 | KILL: raw_p>=0.05 (0.8848333095197831); |t|<3.79 (-0.14530211572134702); dir_mismatch; WFE_none; BH_family_fail |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.5433671067991903); |t|<3.79 (0.6086007278204506); WFE_none; era_unstable; BH_family_fail |
| 10 | MNQ | NYSE_OPEN | long | 2 | KILL: raw_p>=0.05 (0.5945664171282437); |t|<3.79 (-0.5330368150192103); WFE_none; era_unstable; BH_family_fail |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.5871250963690078); |t|<3.79 (-0.5451839519930068); arithmetic_only; WFE_none; BH_family_fail |
| 12 | MNQ | NYSE_OPEN | short | 2 | KILL: raw_p>=0.05 (0.3096717193240561); |t|<3.79 (-1.0227475880416128); arithmetic_only; WFE_none; BH_family_fail |
| 13 | MES | TOKYO_OPEN | long | 1.5 | KILL: |t|<3.79 (-3.633717450326304); WFE_none; era_unstable |
| 14 | MES | TOKYO_OPEN | long | 2 | KILL: WFE_none; era_unstable |
| 15 | MES | TOKYO_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.1840042439812617); |t|<3.79 (1.3398226416013415); WFE>0.95 LEAKAGE_SUSPECT (1.798); BH_family_fail |
| 16 | MES | TOKYO_OPEN | short | 2 | KILL: |t|<3.79 (2.2463649964193118); WFE>0.95 LEAKAGE_SUSPECT (1.616); BH_family_fail |
| 17 | MES | EUROPE_FLOW | long | 1.5 | KILL: WFE_none; era_unstable |
| 18 | MES | EUROPE_FLOW | long | 2 | KILL: |t|<3.79 (-3.5291474188266267); WFE_none; era_unstable |
| 19 | MES | EUROPE_FLOW | short | 1.5 | KILL: raw_p>=0.05 (0.40921846478111745); |t|<3.79 (-0.8296269790842863); WFE_none; BH_family_fail |
| 20 | MES | EUROPE_FLOW | short | 2 | KILL: |t|<3.79 (-2.323662212227902); dir_mismatch; WFE_none; BH_family_fail |
| 21 | MES | NYSE_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.620453839411129); |t|<3.79 (-0.4957267267456075); WFE_none; era_unstable; BH_family_fail |
| 22 | MES | NYSE_OPEN | long | 2 | KILL: raw_p>=0.05 (0.2827781741294546); |t|<3.79 (-1.0761836682161896); arithmetic_only; WFE_none; era_unstable; BH_family_fail |
| 23 | MES | NYSE_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.7222278146557148); |t|<3.79 (0.3566100522574533); WFE>0.95 LEAKAGE_SUSPECT (16.431); BH_family_fail |
| 24 | MES | NYSE_OPEN | short | 2 | KILL: raw_p>=0.05 (0.9903690157852068); |t|<3.79 (-0.012106087163431539); WFE_none; BH_family_fail |

## Methodology notes

- Direction resolved via `daily_features.orb_{session}_break_dir` (orb_outcomes has no direction column; the trade inherits break direction by construction). `long` = up-break, `short` = down-break.
- Base (unfiltered) cell = same (instrument, session, direction, RR, E2, cb=1) lane without the HTF predicate. Delta_IS / Delta_OOS = ExpR_on − ExpR_base.
- t-test: one-sample two-tailed vs 0 on per-trade pnl_r of IS-on trades.
- BH-FDR: classic Benjamini-Hochberg monotone q-value computation. K_family is the primary promotion gate (q < 0.05); other framings reported for honest disclosure per backtesting-methodology.md RULE 4.
- Tautology (T0): Pearson correlation of HTF-fire binary against `orb_{session}_size` (continuous) over IS-window daily_features rows. Proxy for the ORB_G family of size-based filters. |corr| > 0.70 → flagged.
- Arithmetic-only (RULE 8.2): `|WR_on − WR_base| < 0.03` AND `|Δ_IS| > 0.10`. Indicates cost-screen mechanism, not a WR-predictor.
- WFE = Sharpe_OOS_on / Sharpe_IS_on (both per-trade Sharpe on the same scale; annualisation cancels in the ratio). <0.50 fails Criterion 6; >0.95 flagged LEAKAGE_SUSPECT per RULE 3.2.
- Era bins per Criterion 9: 2019-2020, 2021-2022, 2023, 2024-2025. Eras with N < 50 on IS-on trades are exempt.
- Look-ahead: `prev_month_*` populated by canonical `pipeline.build_daily_features._apply_htf_level_fields` from fully-closed prior calendar month only. Drift check 59 (HTF integrity) passes with all 4 divergence classes caught in pressure test (commit 668d2680).
- No writes to `validated_setups` or `experimental_strategies`. Read-only scan.

## Reproduction

```
DUCKDB_PATH=/mnt/c/Users/joshd/canompx3/gold.db python research/htf_path_a_prev_month_v1_scan.py
```
