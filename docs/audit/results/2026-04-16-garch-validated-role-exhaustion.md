# Garch Validated Role Exhaustion

**Date:** 2026-04-16
**Purpose:** Exhaust plausible validated-scope uses of `garch_forecast_vol_pct` before making role claims.

**Grounding:**
- Feature timing and no-lookahead: `pipeline/build_daily_features.py` (`compute_garch_forecast`, prior-rank percentile) and `pipeline/session_guard.py`.
- Role taxonomy: `docs/institutional/mechanism_priors.md` (R1/R3/R7).
- Production gates: `docs/institutional/pre_registered_criteria.md` and `docs/institutional/regime-and-rr-handling-framework.md`.
- Position-size interpretation: `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`.

**Primary family size:** K = 429 tests (216 high-tail + 213 low-tail).

**Thresholds:** HIGH {60,70,80}, LOW {40,30,20}. These were fixed before the run to test upper-tail, lower-tail, and inverse-high possibilities without open-ended fishing.

---

## BH-FDR summary

- Mean-test survivors: **0 / 429**
- Sharpe-permutation survivors: **0 / 429**

If both counts are zero, there is no validated-scope production claim yet for either high-tail or low-tail usage.

---

## Top positive local cells (informational, pre-correction)

| Strategy | Dir | Side | Thr | N on/off | lift | sr_lift | p_sharpe | yrs+ | OOS lift |
|---|---|---|---|---|---|---|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | short | high | 80 | 85/422 | +0.283 | +0.328 | 0.0120 | 4/4 | -0.023 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | high | 80 | 98/151 | +0.378 | +0.322 | 0.0180 | 5/5 | -0.217 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | high | 80 | 115/488 | +0.275 | +0.316 | 0.0030 | 4/4 | -0.170 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | high | 80 | 79/132 | +0.358 | +0.304 | 0.0400 | 5/5 | +0.309 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | high | 70 | 106/105 | +0.344 | +0.292 | 0.0330 | 5/5 | +0.560 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | high | 80 | 143/595 | +0.257 | +0.290 | 0.0050 | 5/5 | -0.430 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | high | 70 | 127/127 | +0.254 | +0.288 | 0.0400 | 4/5 | +0.383 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | high | 60 | 150/104 | +0.247 | +0.278 | 0.0460 | 4/5 | +0.178 |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | long | high | 80 | 99/160 | +0.234 | +0.278 | 0.0380 | 3/4 | -0.097 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | high | 70 | 143/116 | +0.305 | +0.262 | 0.0360 | 5/5 | +0.659 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | high | 70 | 198/540 | +0.233 | +0.260 | 0.0040 | 6/6 | +0.236 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | high | 80 | 141/511 | +0.231 | +0.260 | 0.0100 | 5/5 | -0.379 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | high | 70 | 166/437 | +0.230 | +0.259 | 0.0080 | 6/6 | +0.497 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | high | 80 | 99/155 | +0.223 | +0.254 | 0.0709 | 4/5 | -0.293 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | high | 70 | 106/106 | +0.230 | +0.253 | 0.0789 | 5/5 | +0.616 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | long | high | 80 | 140/581 | +0.353 | +0.245 | 0.0120 | 5/5 | -0.935 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | high | 70 | 125/124 | +0.288 | +0.242 | 0.0629 | 5/5 | +0.795 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | long | high | 80 | 141/590 | +0.286 | +0.242 | 0.0140 | 5/5 | -0.284 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | high | 80 | 79/133 | +0.215 | +0.238 | 0.1199 | 4/5 | +0.168 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | high | 70 | 196/456 | +0.209 | +0.233 | 0.0080 | 5/6 | +0.345 |

---

## Top negative local cells (informational, pre-correction)

| Strategy | Dir | Side | Thr | N on/off | lift | sr_lift | p_sharpe | yrs+ | OOS lift |
|---|---|---|---|---|---|---|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | low | 30 | 19/240 | -0.556 | -0.563 | 0.0270 | 0/2 | n/a |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60 | short | low | 30 | 26/276 | -0.332 | -0.351 | 0.0989 | 1/3 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | low | 40 | 36/223 | -0.373 | -0.338 | 0.0669 | 1/4 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | long | low | 40 | 55/275 | -0.297 | -0.334 | 0.0470 | 1/5 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | long | low | 30 | 27/303 | -0.288 | -0.321 | 0.1449 | 1/3 | n/a |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | long | low | 30 | 43/185 | -0.291 | -0.319 | 0.0639 | 1/5 | -0.950 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | low | 40 | 65/189 | -0.287 | -0.319 | 0.0400 | 0/5 | -0.315 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | long | low | 40 | 56/172 | -0.271 | -0.297 | 0.0569 | 1/5 | -0.657 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | long | low | 20 | 24/204 | -0.273 | -0.296 | 0.1928 | 0/4 | n/a |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | long | low | 20 | 24/204 | -0.327 | -0.286 | 0.1998 | 0/4 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | low | 40 | 64/185 | -0.319 | -0.272 | 0.0609 | 0/5 | +0.057 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60 | short | low | 20 | 11/304 | -0.325 | -0.267 | 0.4006 | 0/2 | n/a |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60 | long | low | 30 | 25/282 | -0.254 | -0.265 | 0.2198 | 0/2 | n/a |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60 | short | low | 40 | 47/255 | -0.245 | -0.260 | 0.0989 | 0/3 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | low | 20 | 31/218 | -0.292 | -0.250 | 0.1728 | 1/3 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | long | low | 40 | 54/270 | -0.269 | -0.230 | 0.1289 | 2/5 | n/a |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | low | 30 | 44/205 | -0.265 | -0.225 | 0.1758 | 1/5 | +0.238 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | low | 40 | 295/357 | -0.196 | -0.216 | 0.0090 | 1/6 | -0.073 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60 | short | high | 70 | 172/143 | -0.256 | -0.213 | 0.0599 | 2/6 | -0.817 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | long | low | 30 | 43/185 | -0.245 | -0.212 | 0.2208 | 2/5 | -1.023 |

---

## Tail-shape diagnostics

NTILE(5) checks whether the signal is truly tail-driven. Per Carver-style continuous sizing, a forecast is easier to justify when the edge improves coherently toward a tail rather than peaking in the middle.

| Strategy | Dir | Best bucket | Tail bias (Q5-Q1) | Spearman rho(bucket, meanR) |
|---|---|---|---|---|
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | long | 4 | +0.231 | +0.600 |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | short | 2 | -0.021 | -0.200 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | 4 | +0.195 | +0.900 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | short | 4 | +0.015 | +0.000 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | 4 | +0.259 | +0.600 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | short | 4 | +0.071 | +0.000 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | 4 | +0.336 | +0.700 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | short | 4 | +0.227 | +0.300 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | 3 | +0.312 | +0.900 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | 3 | +0.146 | +0.900 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | long | 2 | +0.250 | +0.700 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | short | 3 | +0.015 | +0.800 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | long | 4 | +0.288 | +0.700 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | short | 4 | +0.154 | +0.100 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | 3 | +0.431 | +0.800 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | 3 | +0.233 | +0.600 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | long | 3 | +0.223 | +0.900 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | 3 | +0.269 | +0.700 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | long | 4 | +0.377 | +0.900 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | short | 4 | +0.075 | +0.000 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12 | long | 4 | +0.122 | +0.500 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12 | short | 3 | +0.007 | +0.500 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5 | long | 4 | +0.148 | +0.400 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5 | short | 0 | -0.018 | +0.000 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5_NOFRI | long | 4 | +0.098 | +0.300 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5_NOFRI | short | 0 | -0.007 | +0.000 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | long | 4 | +0.310 | +0.900 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | short | 2 | -0.020 | -0.300 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12 | long | 4 | +0.118 | +0.700 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12 | short | 4 | +0.019 | +0.100 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | long | 4 | +0.186 | +0.300 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | short | 0 | -0.036 | +0.000 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | long | 4 | +0.278 | +0.900 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | short | 2 | -0.012 | -0.300 |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | long | 4 | +0.152 | +0.200 |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | short | 2 | +0.043 | +0.300 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | long | 1 | -0.072 | -0.600 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | short | 4 | +0.028 | +0.600 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | long | 1 | -0.059 | -0.600 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | short | 4 | +0.039 | +0.600 |

---

## Role classification by strategy-direction

| Strategy | Dir | Session | RR | Filter | Role guess | Best bucket | Tail bias |
|---|---|---|---|---|---|---|---|
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | long | CME_PRECLOSE | 1.0 | X_MES_ATR60 | upper-tail regime / possible R3-R7 | 4 | +0.231 |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | short | CME_PRECLOSE | 1.0 | X_MES_ATR60 | null / insufficient | 2 | -0.021 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | COMEX_SETTLE | 1.0 | COST_LT12 | upper-tail regime / possible R3-R7 | 4 | +0.195 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | short | COMEX_SETTLE | 1.0 | COST_LT12 | null / insufficient | 4 | +0.015 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | COMEX_SETTLE | 1.0 | ORB_G5 | upper-tail regime / possible R3-R7 | 4 | +0.259 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | short | COMEX_SETTLE | 1.0 | ORB_G5 | null / insufficient | 4 | +0.071 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | COMEX_SETTLE | 1.0 | ORB_G5_NOFRI | upper-tail regime / possible R3-R7 | 4 | +0.336 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | short | COMEX_SETTLE | 1.0 | ORB_G5_NOFRI | upper-tail regime / possible R3-R7 | 4 | +0.227 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | COMEX_SETTLE | 1.0 | OVNRNG_100 | upper-tail binary candidate | 3 | +0.312 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | COMEX_SETTLE | 1.0 | OVNRNG_100 | upper-tail binary candidate | 3 | +0.146 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | long | COMEX_SETTLE | 1.0 | X_MES_ATR60 | upper-tail binary candidate | 2 | +0.250 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | short | COMEX_SETTLE | 1.0 | X_MES_ATR60 | upper-tail binary candidate | 3 | +0.015 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | long | COMEX_SETTLE | 1.5 | ORB_G5 | upper-tail regime / possible R3-R7 | 4 | +0.288 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | short | COMEX_SETTLE | 1.5 | ORB_G5 | upper-tail regime / possible R3-R7 | 4 | +0.154 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | COMEX_SETTLE | 1.5 | OVNRNG_100 | upper-tail binary candidate | 3 | +0.431 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | COMEX_SETTLE | 1.5 | OVNRNG_100 | upper-tail binary candidate | 3 | +0.233 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | long | COMEX_SETTLE | 1.5 | X_MES_ATR60 | null / insufficient | 3 | +0.223 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | COMEX_SETTLE | 1.5 | X_MES_ATR60 | upper-tail binary candidate | 3 | +0.269 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | long | COMEX_SETTLE | 2.0 | ORB_G5 | upper-tail regime / possible R3-R7 | 4 | +0.377 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | short | COMEX_SETTLE | 2.0 | ORB_G5 | null / insufficient | 4 | +0.075 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12 | long | EUROPE_FLOW | 1.0 | COST_LT12 | upper-tail regime / possible R3-R7 | 4 | +0.122 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12 | short | EUROPE_FLOW | 1.0 | COST_LT12 | null / insufficient | 3 | +0.007 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5 | long | EUROPE_FLOW | 1.0 | ORB_G5 | null / insufficient | 4 | +0.148 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5 | short | EUROPE_FLOW | 1.0 | ORB_G5 | null / insufficient | 0 | -0.018 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5_NOFRI | long | EUROPE_FLOW | 1.0 | ORB_G5_NOFRI | null / insufficient | 4 | +0.098 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5_NOFRI | short | EUROPE_FLOW | 1.0 | ORB_G5_NOFRI | null / insufficient | 0 | -0.007 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | long | EUROPE_FLOW | 1.0 | OVNRNG_100 | null / insufficient | 4 | +0.310 |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100 | short | EUROPE_FLOW | 1.0 | OVNRNG_100 | null / insufficient | 2 | -0.020 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12 | long | EUROPE_FLOW | 1.5 | COST_LT12 | null / insufficient | 4 | +0.118 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12 | short | EUROPE_FLOW | 1.5 | COST_LT12 | null / insufficient | 4 | +0.019 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | long | EUROPE_FLOW | 1.5 | ORB_G5 | upper-tail regime / possible R3-R7 | 4 | +0.186 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | short | EUROPE_FLOW | 1.5 | ORB_G5 | null / insufficient | 0 | -0.036 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | long | EUROPE_FLOW | 1.5 | OVNRNG_100 | null / insufficient | 4 | +0.278 |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100 | short | EUROPE_FLOW | 1.5 | OVNRNG_100 | null / insufficient | 2 | -0.012 |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | long | EUROPE_FLOW | 2.0 | ORB_G5 | null / insufficient | 4 | +0.152 |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | short | EUROPE_FLOW | 2.0 | ORB_G5 | null / insufficient | 2 | +0.043 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | long | NYSE_OPEN | 1.0 | COST_LT12 | null / insufficient | 1 | -0.072 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | short | NYSE_OPEN | 1.0 | COST_LT12 | null / insufficient | 4 | +0.028 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | long | NYSE_OPEN | 1.0 | ORB_G5 | null / insufficient | 1 | -0.059 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | short | NYSE_OPEN | 1.0 | ORB_G5 | null / insufficient | 4 | +0.039 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60 | long | NYSE_OPEN | 1.0 | X_MES_ATR60 | null / insufficient | 1 | -0.021 |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60 | short | NYSE_OPEN | 1.0 | X_MES_ATR60 | null / insufficient | 3 | -0.181 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 | long | NYSE_OPEN | 1.5 | COST_LT12 | lower-tail binary candidate | 1 | -0.178 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 | short | NYSE_OPEN | 1.5 | COST_LT12 | null / insufficient | 0 | -0.039 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | long | NYSE_OPEN | 1.5 | ORB_G5 | lower-tail binary candidate | 1 | -0.178 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | short | NYSE_OPEN | 1.5 | ORB_G5 | null / insufficient | 0 | -0.045 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60 | long | NYSE_OPEN | 1.5 | X_MES_ATR60 | null / insufficient | 3 | -0.073 |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60 | short | NYSE_OPEN | 1.5 | X_MES_ATR60 | null / insufficient | 0 | -0.292 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | long | SINGAPORE_OPEN | 1.5 | ATR_P50 | null / insufficient | 3 | +0.116 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | short | SINGAPORE_OPEN | 1.5 | ATR_P50 | null / insufficient | 4 | +0.142 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30 | long | SINGAPORE_OPEN | 1.5 | ATR_P50 | null / insufficient | 2 | +0.152 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30 | short | SINGAPORE_OPEN | 1.5 | ATR_P50 | null / insufficient | 2 | +0.058 |
| MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12 | long | TOKYO_OPEN | 1.0 | COST_LT12 | null / insufficient | 4 | +0.144 |
| MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12 | short | TOKYO_OPEN | 1.0 | COST_LT12 | null / insufficient | 3 | +0.087 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | long | TOKYO_OPEN | 1.5 | COST_LT12 | null / insufficient | 2 | +0.174 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | short | TOKYO_OPEN | 1.5 | COST_LT12 | null / insufficient | 3 | +0.120 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5 | long | TOKYO_OPEN | 1.5 | ORB_G5 | null / insufficient | 4 | +0.178 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5 | short | TOKYO_OPEN | 1.5 | ORB_G5 | null / insufficient | 4 | +0.197 |
| MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5 | long | TOKYO_OPEN | 2.0 | ORB_G5 | null / insufficient | 4 | +0.125 |
| MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5 | short | TOKYO_OPEN | 2.0 | ORB_G5 | null / insufficient | 4 | +0.144 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_NOFRI | long | US_DATA_1000 | 1.0 | ORB_G5_NOFRI | null / insufficient | 2 | +0.027 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G5_NOFRI | short | US_DATA_1000 | 1.0 | ORB_G5_NOFRI | null / insufficient | 4 | +0.183 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 | long | US_DATA_1000 | 1.0 | VWAP_MID_ALIGNED | lower-tail binary candidate | 1 | -0.122 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15 | short | US_DATA_1000 | 1.0 | VWAP_MID_ALIGNED | null / insufficient | 4 | +0.069 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60 | long | US_DATA_1000 | 1.0 | X_MES_ATR60 | null / insufficient | 1 | -0.070 |
| MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60 | short | US_DATA_1000 | 1.0 | X_MES_ATR60 | null / insufficient | 3 | +0.125 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | long | US_DATA_1000 | 1.5 | ORB_G5 | null / insufficient | 2 | +0.004 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | short | US_DATA_1000 | 1.5 | ORB_G5 | null / insufficient | 0 | -0.001 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | long | US_DATA_1000 | 1.5 | VWAP_MID_ALIGNED | null / insufficient | 1 | -0.119 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | short | US_DATA_1000 | 1.5 | VWAP_MID_ALIGNED | null / insufficient | 4 | +0.063 |
| MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15 | long | US_DATA_1000 | 2.0 | VWAP_MID_ALIGNED | null / insufficient | 0 | -0.131 |
| MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15 | short | US_DATA_1000 | 2.0 | VWAP_MID_ALIGNED | null / insufficient | 4 | +0.048 |

---

## Honest institutional interpretation

- `garch` is only production-ready as a filter/sizer if it survives validated-scope family correction. Local cells alone are not enough.
- A strong **upper tail** with best bucket `Q5` supports `R3/R7` exploration better than `R1` binary skip.
- A strong **lower tail** with best bucket `Q1` supports inverse-high avoidance or low-regime preference.
- A best bucket in the middle argues AGAINST Carver-style continuous sizing and suggests the signal is not behaving like a clean forecast.
- Even where local tail behavior is promising, production still requires pre-registration, forward OOS accumulation, Monte Carlo, and live Shiryaev-Roberts monitoring.
