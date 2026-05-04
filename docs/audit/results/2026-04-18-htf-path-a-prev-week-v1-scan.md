# HTF Path A prev-week v1 — Family Scan Results

**Generated:** 2026-04-20T11:49:07+00:00
**Scan script HEAD SHA:** `39cfc98dd23e405e7afda652db7e0eaeb4d6221f`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`
**Holdout (Mode A):** `trading_day >= 2026-01-01` — imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`
**Pre-registration:** `docs/audit/hypotheses/2026-04-18-htf-path-a-prev-week-v1.yaml`

## Scope

- Instruments: ['MNQ', 'MES']
- Sessions: ['TOKYO_OPEN', 'EUROPE_FLOW', 'NYSE_OPEN']
- Aperture: O15
- RR targets: [1.5, 2.0]
- Directions / cells: ['long', 'short'] (pwh_break_long / pwl_break_short)
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
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | 870 | 216 | 24.8 | 0.098 | 0.047 | -0.052 | 1.339 | 1.207 | — | 0.61 | 0.5423 | 0.801 | 0.820 | 0.686 | 0.813 |
| 2 | MNQ | TOKYO_OPEN | long | 2 | 870 | 216 | 24.8 | 0.116 | 0.029 | -0.086 | 1.807 | 1.449 | — | 0.33 | 0.7418 | 0.841 | 0.820 | 0.742 | 0.820 |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | 851 | 97 | 11.4 | 0.012 | 0.162 | 0.150 | 0.037 | 0.089 | True | 1.35 | 0.1810 | 0.434 | 0.619 | 0.362 | 0.362 |
| 4 | MNQ | TOKYO_OPEN | short | 2 | 851 | 97 | 11.4 | -0.034 | 0.162 | 0.196 | 0.245 | 0.281 | True | 1.14 | 0.2572 | 0.561 | 0.619 | 0.411 | 0.441 |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | 876 | 242 | 27.6 | 0.067 | -0.016 | -0.083 | 0.182 | -0.052 | — | -0.23 | 0.8197 | 0.855 | 0.820 | 0.937 | 0.820 |
| 6 | MNQ | EUROPE_FLOW | long | 2 | 876 | 242 | 27.6 | 0.072 | -0.027 | -0.099 | 0.419 | 0.268 | — | -0.32 | 0.7457 | 0.841 | 0.820 | 0.937 | 0.820 |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | 843 | 101 | 12.0 | -0.001 | 0.178 | 0.179 | -0.203 | -0.393 | False | 1.49 | 0.1383 | 0.415 | 0.619 | 0.277 | 0.332 |
| 8 | MNQ | EUROPE_FLOW | short | 2 | 843 | 101 | 12.0 | 0.004 | 0.244 | 0.240 | -0.044 | -0.018 | False | 1.72 | 0.0878 | 0.415 | 0.619 | 0.234 | 0.332 |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | 880 | 286 | 32.5 | 0.058 | 0.089 | 0.031 | 0.235 | 0.178 | — | 1.08 | 0.2826 | 0.565 | 0.619 | 0.521 | 0.670 |
| 10 | MNQ | NYSE_OPEN | long | 2 | 880 | 286 | 32.5 | -0.019 | 0.029 | 0.048 | -1.000 | -0.821 | — | 0.29 | 0.7710 | 0.841 | 0.820 | 0.771 | 0.820 |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | 835 | 139 | 16.6 | 0.078 | 0.110 | 0.032 | 0.237 | -0.266 | False | 1.02 | 0.3094 | 0.571 | 0.619 | 0.521 | 0.464 |
| 12 | MNQ | NYSE_OPEN | short | 2 | 835 | 139 | 16.6 | 0.039 | 0.057 | 0.018 | 0.485 | -0.144 | False | 0.43 | 0.6671 | 0.841 | 0.820 | 0.762 | 0.728 |
| 13 | MES | TOKYO_OPEN | long | 1.5 | 856 | 181 | 21.1 | -0.055 | -0.103 | -0.048 | 0.668 | 0.566 | — | -1.36 | 0.1765 | 0.434 | 0.303 | 0.362 | 0.530 |
| 14 | MES | TOKYO_OPEN | long | 2 | 856 | 181 | 21.1 | -0.031 | -0.130 | -0.099 | 1.001 | 0.754 | — | -1.49 | 0.1370 | 0.415 | 0.274 | 0.362 | 0.530 |
| 15 | MES | TOKYO_OPEN | short | 1.5 | 863 | 92 | 10.7 | -0.142 | 0.176 | 0.318 | -0.413 | -0.336 | — | 1.52 | 0.1325 | 0.415 | 0.274 | 0.362 | 0.332 |
| 16 | MES | TOKYO_OPEN | short | 2 | 863 | 92 | 10.7 | -0.173 | 0.072 | 0.245 | -0.296 | -0.404 | — | 0.53 | 0.6004 | 0.801 | 0.655 | 0.686 | 0.720 |
| 17 | MES | EUROPE_FLOW | long | 1.5 | 888 | 215 | 24.2 | -0.091 | -0.278 | -0.187 | -1.000 | -0.968 | — | -4.13 | 0.0001 | 0.001 | 0.000 | 0.000 | 0.000 |
| 18 | MES | EUROPE_FLOW | long | 2 | 888 | 215 | 24.2 | -0.076 | -0.306 | -0.230 | -1.000 | -1.161 | — | -4.04 | 0.0001 | 0.001 | 0.000 | 0.000 | 0.000 |
| 19 | MES | EUROPE_FLOW | short | 1.5 | 831 | 95 | 11.4 | -0.162 | -0.009 | 0.153 | -0.099 | 0.192 | True | -0.08 | 0.9379 | 0.938 | 0.938 | 0.938 | 0.938 |
| 20 | MES | EUROPE_FLOW | short | 2 | 831 | 95 | 11.4 | -0.171 | -0.071 | 0.099 | 0.081 | 0.467 | True | -0.54 | 0.5885 | 0.801 | 0.655 | 0.937 | 0.720 |
| 21 | MES | NYSE_OPEN | long | 1.5 | 878 | 265 | 30.2 | 0.063 | -0.069 | -0.132 | -0.541 | -0.191 | True | -0.94 | 0.3466 | 0.594 | 0.520 | 0.521 | 0.670 |
| 22 | MES | NYSE_OPEN | long | 2 | 878 | 265 | 30.2 | 0.036 | -0.076 | -0.112 | -0.449 | -0.229 | True | -0.86 | 0.3909 | 0.625 | 0.521 | 0.521 | 0.670 |
| 23 | MES | NYSE_OPEN | short | 1.5 | 840 | 135 | 16.1 | 0.038 | 0.240 | 0.203 | -0.594 | -0.843 | False | 2.26 | 0.0254 | 0.203 | 0.102 | 0.203 | 0.305 |
| 24 | MES | NYSE_OPEN | short | 2 | 840 | 135 | 16.1 | -0.003 | 0.200 | 0.203 | -0.513 | -0.846 | False | 1.54 | 0.1258 | 0.415 | 0.274 | 0.503 | 0.332 |

## WFE, era stability, flags

| # | inst | session | dir | RR | Sharpe_IS | Sharpe_OOS | WFE | era_stable | tautology corr | tautology? | arithmetic_only |
|--|-----|---------|-----|----|----:|----:|----:|:--:|----:|:--:|:--:|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | 0.042 | 24.090 | 580.099 | False | -0.132 | False | False |
| 2 | MNQ | TOKYO_OPEN | long | 2 | 0.022 | 27.094 | 1207.250 | False | -0.132 | False | False |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | 0.137 | 0.029 | 0.212 | True | 0.132 | False | False |
| 4 | MNQ | TOKYO_OPEN | short | 2 | 0.116 | 0.158 | 1.363 | True | 0.132 | False | False |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | -0.015 | 0.109 | — | False | -0.134 | False | False |
| 6 | MNQ | EUROPE_FLOW | long | 2 | -0.021 | 0.209 | — | False | -0.134 | False | False |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | 0.149 | -0.164 | -1.106 | True | 0.209 | False | False |
| 8 | MNQ | EUROPE_FLOW | short | 2 | 0.172 | -0.029 | -0.171 | True | 0.209 | False | False |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | 0.073 | 0.135 | 1.838 | False | -0.168 | False | False |
| 10 | MNQ | NYSE_OPEN | long | 2 | 0.021 | — | — | False | -0.168 | False | False |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | 0.090 | 0.166 | 1.842 | False | 0.188 | False | False |
| 12 | MNQ | NYSE_OPEN | short | 2 | 0.040 | 0.283 | 7.093 | False | 0.188 | False | False |
| 13 | MES | TOKYO_OPEN | long | 1.5 | -0.101 | 0.598 | — | False | -0.118 | False | False |
| 14 | MES | TOKYO_OPEN | long | 2 | -0.111 | 0.747 | — | False | -0.118 | False | False |
| 15 | MES | TOKYO_OPEN | short | 1.5 | 0.158 | -0.352 | -2.226 | True | 0.129 | False | False |
| 16 | MES | TOKYO_OPEN | short | 2 | 0.055 | -0.210 | -3.836 | True | 0.129 | False | False |
| 17 | MES | EUROPE_FLOW | long | 1.5 | -0.282 | — | — | False | -0.107 | False | False |
| 18 | MES | EUROPE_FLOW | long | 2 | -0.277 | — | — | False | -0.107 | False | False |
| 19 | MES | EUROPE_FLOW | short | 1.5 | -0.008 | -0.080 | — | True | 0.186 | False | False |
| 20 | MES | EUROPE_FLOW | short | 2 | -0.056 | 0.055 | — | True | 0.186 | False | False |
| 21 | MES | NYSE_OPEN | long | 1.5 | -0.061 | -0.527 | — | False | -0.173 | False | False |
| 22 | MES | NYSE_OPEN | long | 2 | -0.058 | -0.365 | — | False | -0.173 | False | True |
| 23 | MES | NYSE_OPEN | short | 1.5 | 0.201 | -0.597 | -2.976 | True | 0.234 | False | False |
| 24 | MES | NYSE_OPEN | short | 2 | 0.141 | -0.430 | -3.041 | True | 0.234 | False | False |

## Era stability detail (per cell, IS-on trades)

| # | inst | session | dir | RR | 2019-2020 (n, ExpR) | 2021-2022 (n, ExpR) | 2023 (n, ExpR) | 2024-2025 (n, ExpR) |
|--|-----|---------|-----|----|----:|----:|----:|----:|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | n=67, ExpR=-0.086 | n=50, ExpR=0.348 | n=36, ExpR=-0.274* | n=63, ExpR=0.132 |
| 2 | MNQ | TOKYO_OPEN | long | 2 | n=67, ExpR=-0.104 | n=50, ExpR=0.287 | n=36, ExpR=-0.277* | n=63, ExpR=0.143 |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | n=16, ExpR=0.016* | n=40, ExpR=0.173* | n=12, ExpR=-0.429* | n=29, ExpR=0.471* |
| 4 | MNQ | TOKYO_OPEN | short | 2 | n=16, ExpR=0.040* | n=40, ExpR=0.128* | n=12, ExpR=-0.546* | n=29, ExpR=0.568* |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | n=77, ExpR=-0.036 | n=54, ExpR=0.041 | n=37, ExpR=0.148* | n=74, ExpR=-0.120 |
| 6 | MNQ | EUROPE_FLOW | long | 2 | n=77, ExpR=0.108 | n=54, ExpR=-0.051 | n=37, ExpR=0.102* | n=74, ExpR=-0.213 |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | n=13, ExpR=0.073* | n=56, ExpR=0.026 | n=9, ExpR=0.554* | n=23, ExpR=0.460* |
| 8 | MNQ | EUROPE_FLOW | short | 2 | n=13, ExpR=0.079* | n=56, ExpR=0.076 | n=9, ExpR=0.864* | n=23, ExpR=0.505* |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | n=87, ExpR=-0.088 | n=78, ExpR=0.043 | n=47, ExpR=0.123* | n=74, ExpR=0.302 |
| 10 | MNQ | NYSE_OPEN | long | 2 | n=87, ExpR=-0.077 | n=78, ExpR=-0.103 | n=47, ExpR=0.184* | n=74, ExpR=0.178 |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | n=23, ExpR=0.159* | n=59, ExpR=-0.225 | n=16, ExpR=1.115* | n=41, ExpR=0.164* |
| 12 | MNQ | NYSE_OPEN | short | 2 | n=23, ExpR=-0.026* | n=59, ExpR=-0.291 | n=16, ExpR=1.194* | n=41, ExpR=0.217* |
| 13 | MES | TOKYO_OPEN | long | 1.5 | n=51, ExpR=-0.113 | n=43, ExpR=-0.052* | n=29, ExpR=-0.141* | n=58, ExpR=-0.112 |
| 14 | MES | TOKYO_OPEN | long | 2 | n=51, ExpR=-0.135 | n=43, ExpR=-0.039* | n=29, ExpR=-0.054* | n=58, ExpR=-0.231 |
| 15 | MES | TOKYO_OPEN | short | 1.5 | n=18, ExpR=0.106* | n=36, ExpR=0.044* | n=13, ExpR=-0.192* | n=25, ExpR=0.608* |
| 16 | MES | TOKYO_OPEN | short | 2 | n=18, ExpR=0.191* | n=36, ExpR=-0.040* | n=13, ExpR=-0.410* | n=25, ExpR=0.399* |
| 17 | MES | EUROPE_FLOW | long | 1.5 | n=61, ExpR=-0.189 | n=49, ExpR=-0.412* | n=29, ExpR=-0.159* | n=76, ExpR=-0.308 |
| 18 | MES | EUROPE_FLOW | long | 2 | n=61, ExpR=-0.136 | n=49, ExpR=-0.351* | n=29, ExpR=-0.238* | n=76, ExpR=-0.438 |
| 19 | MES | EUROPE_FLOW | short | 1.5 | n=19, ExpR=0.051* | n=46, ExpR=-0.097* | n=9, ExpR=0.653* | n=21, ExpR=-0.153* |
| 20 | MES | EUROPE_FLOW | short | 2 | n=19, ExpR=0.124* | n=46, ExpR=-0.343* | n=9, ExpR=0.983* | n=21, ExpR=-0.107* |
| 21 | MES | NYSE_OPEN | long | 1.5 | n=72, ExpR=-0.228 | n=75, ExpR=0.113 | n=42, ExpR=-0.110* | n=76, ExpR=-0.081 |
| 22 | MES | NYSE_OPEN | long | 2 | n=72, ExpR=-0.241 | n=75, ExpR=0.059 | n=42, ExpR=-0.242* | n=76, ExpR=0.035 |
| 23 | MES | NYSE_OPEN | short | 1.5 | n=29, ExpR=0.227* | n=53, ExpR=0.102 | n=17, ExpR=0.234* | n=36, ExpR=0.444* |
| 24 | MES | NYSE_OPEN | short | 2 | n=29, ExpR=0.130* | n=53, ExpR=0.064 | n=17, ExpR=0.481* | n=36, ExpR=0.305* |

*exempt = N < 50 (Criterion 9 threshold)*

## Verdict per cell

| # | inst | session | dir | RR | verdict |
|--|-----|---------|-----|----|---|
| 1 | MNQ | TOKYO_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.5422872715705909); |t|<3.79 (0.6103308232966004); WFE>0.95 LEAKAGE_SUSPECT (580.099); era_unstable; BH_family_fail |
| 2 | MNQ | TOKYO_OPEN | long | 2 | KILL: raw_p>=0.05 (0.7418399451674178); |t|<3.79 (0.32984272586082936); WFE>0.95 LEAKAGE_SUSPECT (1207.250); era_unstable; BH_family_fail |
| 3 | MNQ | TOKYO_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.18095320977853047); |t|<3.79 (1.347617654123606); WFE<0.5 (0.212); BH_family_fail |
| 4 | MNQ | TOKYO_OPEN | short | 2 | KILL: raw_p>=0.05 (0.25718069245046005); |t|<3.79 (1.1398598565629092); WFE>0.95 LEAKAGE_SUSPECT (1.363); BH_family_fail |
| 5 | MNQ | EUROPE_FLOW | long | 1.5 | KILL: raw_p>=0.05 (0.8196716154978865); |t|<3.79 (-0.22821628681236858); WFE_none; era_unstable; BH_family_fail |
| 6 | MNQ | EUROPE_FLOW | long | 2 | KILL: raw_p>=0.05 (0.7457384508673077); |t|<3.79 (-0.32463719956934756); WFE_none; era_unstable; BH_family_fail |
| 7 | MNQ | EUROPE_FLOW | short | 1.5 | KILL: raw_p>=0.05 (0.1383483152307723); |t|<3.79 (1.4939083388369243); dir_mismatch; WFE<0.5 (-1.106); BH_family_fail |
| 8 | MNQ | EUROPE_FLOW | short | 2 | KILL: raw_p>=0.05 (0.08782550331491223); |t|<3.79 (1.7238482510508364); dir_mismatch; WFE<0.5 (-0.171); BH_family_fail |
| 9 | MNQ | NYSE_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.28261202179893496); |t|<3.79 (1.077168740366025); WFE>0.95 LEAKAGE_SUSPECT (1.838); era_unstable; BH_family_fail |
| 10 | MNQ | NYSE_OPEN | long | 2 | KILL: raw_p>=0.05 (0.7710426186011763); |t|<3.79 (0.2914249028936594); WFE_none; era_unstable; BH_family_fail |
| 11 | MNQ | NYSE_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.30942815132190304); |t|<3.79 (1.0205069560328524); dir_mismatch; WFE>0.95 LEAKAGE_SUSPECT (1.842); era_unstable; BH_family_fail |
| 12 | MNQ | NYSE_OPEN | short | 2 | KILL: raw_p>=0.05 (0.6670791746014895); |t|<3.79 (0.4312607588750078); dir_mismatch; WFE>0.95 LEAKAGE_SUSPECT (7.093); era_unstable; BH_family_fail |
| 13 | MES | TOKYO_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.17651625358392464); |t|<3.79 (-1.3568870856733828); WFE_none; era_unstable; BH_family_fail |
| 14 | MES | TOKYO_OPEN | long | 2 | KILL: raw_p>=0.05 (0.13698487054913167); |t|<3.79 (-1.4937766712652172); WFE_none; era_unstable; BH_family_fail |
| 15 | MES | TOKYO_OPEN | short | 1.5 | KILL: raw_p>=0.05 (0.1324894317040346); |t|<3.79 (1.5179664766574368); WFE<0.5 (-2.226); BH_family_fail |
| 16 | MES | TOKYO_OPEN | short | 2 | KILL: raw_p>=0.05 (0.6004006464772464); |t|<3.79 (0.5256635005910686); WFE<0.5 (-3.836); BH_family_fail |
| 17 | MES | EUROPE_FLOW | long | 1.5 | KILL: WFE_none; era_unstable |
| 18 | MES | EUROPE_FLOW | long | 2 | KILL: WFE_none; era_unstable |
| 19 | MES | EUROPE_FLOW | short | 1.5 | KILL: raw_p>=0.05 (0.9378540474222259); |t|<3.79 (-0.07817622306921117); WFE_none; BH_family_fail |
| 20 | MES | EUROPE_FLOW | short | 2 | KILL: raw_p>=0.05 (0.5885443646710389); |t|<3.79 (-0.542811475748463); WFE_none; BH_family_fail |
| 21 | MES | NYSE_OPEN | long | 1.5 | KILL: raw_p>=0.05 (0.34659240530538504); |t|<3.79 (-0.9431115513752723); WFE_none; era_unstable; BH_family_fail |
| 22 | MES | NYSE_OPEN | long | 2 | KILL: raw_p>=0.05 (0.3909372608990662); |t|<3.79 (-0.8596223272880517); arithmetic_only; WFE_none; era_unstable; BH_family_fail |
| 23 | MES | NYSE_OPEN | short | 1.5 | KILL: |t|<3.79 (2.262105573005065); dir_mismatch; WFE<0.5 (-2.976); BH_family_fail |
| 24 | MES | NYSE_OPEN | short | 2 | KILL: raw_p>=0.05 (0.12584051392522366); |t|<3.79 (1.541632226873137); dir_mismatch; WFE<0.5 (-3.041); BH_family_fail |

## Methodology notes

- Direction resolved via `daily_features.orb_{session}_break_dir` (orb_outcomes has no direction column; the trade inherits break direction by construction). `long` = up-break, `short` = down-break.
- Base (unfiltered) cell = same (instrument, session, direction, RR, E2, cb=1) lane without the HTF predicate. Delta_IS / Delta_OOS = ExpR_on − ExpR_base.
- t-test: one-sample two-tailed vs 0 on per-trade pnl_r of IS-on trades.
- BH-FDR: classic Benjamini-Hochberg monotone q-value computation. K_family is the primary promotion gate (q < 0.05); other framings reported for honest disclosure per backtesting-methodology.md RULE 4.
- Tautology (T0): Pearson correlation of HTF-fire binary against `orb_{session}_size` (continuous) over IS-window daily_features rows. Proxy for the ORB_G family of size-based filters. |corr| > 0.70 → flagged.
- Arithmetic-only (RULE 8.2): `|WR_on − WR_base| < 0.03` AND `|Δ_IS| > 0.10`. Indicates cost-screen mechanism, not a WR-predictor.
- WFE = Sharpe_OOS_on / Sharpe_IS_on (both per-trade Sharpe on the same scale; annualisation cancels in the ratio). <0.50 fails Criterion 6; >0.95 flagged LEAKAGE_SUSPECT per RULE 3.2.
- Era bins per Criterion 9: 2019-2020, 2021-2022, 2023, 2024-2025. Eras with N < 50 on IS-on trades are exempt.
- Look-ahead: `prev_week_*` populated by canonical `pipeline.build_daily_features._apply_htf_level_fields` from fully-closed prior Mon-Sun week only. Drift check 59 (HTF integrity) passes with all 4 divergence classes caught in pressure test (commit 668d2680).
- No writes to `validated_setups` or `experimental_strategies`. Read-only scan.

## Reproduction

```
DUCKDB_PATH=/mnt/c/Users/joshd/canompx3/gold.db python research/htf_path_a_prev_week_v1_scan.py
```
