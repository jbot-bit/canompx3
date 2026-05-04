# Garch Broad Exact Role Exhaustion

**Date:** 2026-04-16
**Scope:** `validated_setups` + `experimental_strategies`, exact filter semantics only.

- Strategy rows in scope: **430**
- Primary tests run: **2630**
- BH mean survivors: **1**
- BH sharpe survivors: **0**

**Included filter families:** ORB_G*, ORB_G*_NOFRI, ATR_P*, COST_LT*, OVNRNG_*, X_MES_ATR60, VWAP_MID_ALIGNED, VWAP_BP_ALIGNED, PDR_R080, GAP_R015.

**Excluded families:** GARCH_* self-reference, NO_FILTER, CROSS_*_MOMENTUM, and filters without exact clean semantics in this pass.

---

## Top positive local cells

| Src | Strategy | Dir | Side | Thr | sr_lift | lift | p_sharpe | OOS lift |
|---|---|---|---|---|---|---|---|---|
| experimental | MES_CME_REOPEN_E2_RR1.0_CB1_ORB_G8_NOFRI | short | low | 40 | +0.660 | +0.526 | 0.0729 | n/a |
| experimental | MGC_LONDON_METALS_E2_RR2.0_CB1_PDR_R080 | short | high | 80 | +0.654 | +0.729 | 0.0010 | +1.325 |
| experimental | MGC_LONDON_METALS_E2_RR2.0_CB1_PDR_R080 | short | high | 70 | +0.599 | +0.639 | 0.0020 | n/a |
| experimental | MES_CME_REOPEN_E2_RR1.0_CB1_ATR_P70 | long | low | 30 | +0.593 | +0.482 | 0.0919 | n/a |
| experimental | MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5_O15 | long | high | 80 | +0.568 | +0.663 | 0.0539 | n/a |
| experimental | MNQ_NYSE_CLOSE_E2_RR1.5_CB1_ORB_G5_O15 | short | high | 80 | +0.549 | +0.645 | 0.0659 | +0.793 |
| experimental | MES_CME_REOPEN_E2_RR1.0_CB1_ORB_G8 | short | low | 40 | +0.538 | +0.478 | 0.1119 | n/a |
| experimental | MNQ_NYSE_CLOSE_E2_RR1.5_CB1_X_MES_ATR60 | short | high | 70 | +0.527 | +0.587 | 0.0130 | n/a |
| experimental | MES_US_DATA_830_E2_RR1.0_CB1_COST_LT08 | long | low | 20 | +0.522 | +0.451 | 0.0100 | n/a |
| experimental | MES_CME_REOPEN_E2_RR1.0_CB1_ATR_P70 | long | low | 40 | +0.514 | +0.432 | 0.0220 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70 | long | high | 60 | +0.510 | +0.390 | 0.0010 | -0.216 |
| experimental | MES_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P70 | long | low | 40 | +0.508 | +0.307 | 0.1119 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.0_CB1_ATR_P70 | short | high | 70 | +0.503 | +0.394 | 0.0020 | +0.318 |
| experimental | MES_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25 | long | high | 80 | +0.496 | +0.419 | 0.0050 | +0.181 |
| experimental | MNQ_US_DATA_1000_E2_RR1.5_CB1_X_MES_ATR60_O30 | long | low | 20 | +0.491 | +0.600 | 0.2108 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.5_CB1_ATR_P70 | short | high | 60 | +0.491 | +0.406 | 0.0040 | n/a |
| experimental | MES_US_DATA_830_E2_RR1.5_CB1_COST_LT08 | long | low | 20 | +0.490 | +0.574 | 0.0280 | n/a |
| experimental | MES_CME_REOPEN_E2_RR1.5_CB1_ATR_P70 | long | low | 40 | +0.489 | +0.506 | 0.0889 | n/a |
| validated | MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08 | long | high | 80 | +0.476 | +0.374 | 0.0450 | n/a |
| experimental | MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08 | long | high | 80 | +0.476 | +0.374 | 0.0450 | n/a |
| experimental | MES_CME_REOPEN_E2_RR1.5_CB1_ORB_G8 | short | low | 40 | +0.471 | +0.559 | 0.1778 | n/a |
| experimental | MES_LONDON_METALS_E2_RR1.0_CB1_ATR_P70 | long | high | 60 | +0.465 | +0.384 | 0.0010 | n/a |
| experimental | MES_NYSE_CLOSE_E2_RR1.5_CB1_ATR_P70 | long | low | 40 | +0.461 | +0.475 | 0.1648 | n/a |
| experimental | MES_NYSE_CLOSE_E2_RR1.5_CB1_OVNRNG_25 | short | high | 80 | +0.461 | +0.399 | 0.1169 | +0.634 |
| experimental | MES_EUROPE_FLOW_E2_RR1.5_CB1_ATR_P70 | short | high | 70 | +0.454 | +0.398 | 0.0020 | +0.522 |

---

## Top negative local cells

| Src | Strategy | Dir | Side | Thr | sr_lift | lift | p_sharpe | OOS lift |
|---|---|---|---|---|---|---|---|---|
| experimental | MES_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70 | long | low | 20 | -1.660 | -0.751 | 0.0020 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.0_CB1_ATR_P70 | long | low | 30 | -1.384 | -0.775 | 0.0020 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70 | long | low | 20 | -1.337 | -0.784 | 0.0050 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.5_CB1_ATR_P70 | long | low | 30 | -1.001 | -0.688 | 0.0030 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70 | long | low | 30 | -0.960 | -0.702 | 0.0010 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70 | long | low | 30 | -0.889 | -0.572 | 0.0010 | n/a |
| experimental | MES_LONDON_METALS_E2_RR1.0_CB1_ATR_P70 | long | low | 30 | -0.762 | -0.566 | 0.0030 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.5_CB1_ATR_P70 | short | low | 30 | -0.739 | -0.525 | 0.0070 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.0_CB1_ATR_P70 | short | low | 30 | -0.681 | -0.486 | 0.0070 | n/a |
| experimental | MES_EUROPE_FLOW_E2_RR1.5_CB1_ATR_P70 | short | low | 20 | -0.661 | -0.493 | 0.0609 | n/a |
| experimental | MES_LONDON_METALS_E2_RR1.0_CB1_ATR_P70 | long | low | 20 | -0.647 | -0.488 | 0.0400 | n/a |
| experimental | MNQ_US_DATA_1000_E2_RR1.5_CB1_X_MES_ATR60_O15 | short | low | 20 | -0.647 | -0.652 | 0.0709 | n/a |
| experimental | MES_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25 | short | low | 30 | -0.635 | -0.594 | 0.0040 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P70 | long | low | 40 | -0.626 | -0.554 | 0.0010 | n/a |
| experimental | MES_SINGAPORE_OPEN_E2_RR1.0_CB1_ATR_P70 | long | low | 40 | -0.608 | -0.447 | 0.0010 | n/a |
| experimental | MES_CME_PRECLOSE_E2_RR1.5_CB1_OVNRNG_25 | long | low | 20 | -0.594 | -0.584 | 0.0210 | n/a |
| experimental | MES_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25 | long | low | 20 | -0.587 | -0.504 | 0.0240 | n/a |
| experimental | MES_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25 | short | low | 30 | -0.583 | -0.492 | 0.0050 | n/a |
| experimental | MES_US_DATA_830_E2_RR1.5_CB1_ATR_P70 | short | low | 30 | -0.573 | -0.493 | 0.0300 | n/a |
| validated | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | low | 30 | -0.563 | -0.556 | 0.0270 | n/a |
| experimental | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60 | short | low | 30 | -0.563 | -0.556 | 0.0270 | n/a |
| experimental | MES_NYSE_OPEN_E2_RR1.5_CB1_ATR_P70 | short | low | 20 | -0.559 | -0.557 | 0.0509 | n/a |
| experimental | MES_LONDON_METALS_E2_RR1.0_CB1_OVNRNG_25 | short | low | 20 | -0.552 | -0.472 | 0.0150 | n/a |
| experimental | MES_NYSE_OPEN_E2_RR1.5_CB1_ATR_P70 | short | low | 30 | -0.540 | -0.551 | 0.0140 | n/a |
| experimental | MES_US_DATA_1000_E2_RR1.5_CB1_ATR_P70 | short | low | 30 | -0.527 | -0.539 | 0.0480 | n/a |

---

## Shape summary by session

| Session | Count | Mean tail bias | Mean best bucket |
|---|---|---|---|
| BRISBANE_1025 | 12 | +0.013 | 2.25 |
| CME_PRECLOSE | 40 | +0.112 | 2.55 |
| CME_REOPEN | 16 | -0.033 | 2.56 |
| COMEX_SETTLE | 52 | +0.195 | 3.19 |
| EUROPE_FLOW | 42 | +0.119 | 2.62 |
| LONDON_METALS | 26 | +0.136 | 2.73 |
| NYSE_CLOSE | 20 | +0.043 | 2.35 |
| NYSE_OPEN | 60 | -0.046 | 1.72 |
| SINGAPORE_OPEN | 28 | +0.100 | 2.54 |
| TOKYO_OPEN | 32 | +0.202 | 3.28 |
| US_DATA_1000 | 64 | +0.061 | 2.38 |
| US_DATA_830 | 38 | -0.066 | 1.74 |