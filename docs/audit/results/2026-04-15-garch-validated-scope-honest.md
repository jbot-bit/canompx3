# Garch Overlay — Validated-Scope Honest Test

**Date:** 2026-04-15
**Trigger:** User called out that prior garch tests violated the Validated Universe Rule by running on unfiltered `orb_outcomes`. This test applies each validated strategy's EXACT filter before testing garch overlay.

**K = 180** validated-scope test cells (validated_strategy × direction × garch threshold).

**Scope:** 124 validated_setups in gold.db, filtered to testable filter types (ORB_G5, ORB_G5_NOFRI, COST_LT12, OVNRNG_100, ATR_P50, ATR_P70, VWAP_MID_ALIGNED, VWAP_BP_ALIGNED). Cross-instrument filter types (X_MES_ATR60) and CROSS_*_MOMENTUM filters excluded for complexity.

**Methodology:**
- For each validated strategy, load trades where its exact filter fires
- Split by break_direction (long/short)
- Test garch threshold overlay at 60/70/80
- Welch t-test on mean + permutation test on Sharpe lift + per-year consistency
- BH-FDR at K = 180 (all validated-scope tests)

---

## BH-FDR survivors (validated-scope honest framing)

**Survivors on Sharpe permutation: 0 / 180**
**Survivors on mean t-test: 0 / 180**

_No survivors at validated-scope BH-FDR._

This is the CORRECT test — it confirms that when tested on the actual deployed/validated trade population, garch overlay does NOT produce statistically significant lift after multiple-testing correction.

The prior 'all-sessions universality' claim of 21 surviving families was testing on UNFILTERED orb_outcomes and does not apply to deployable strategies.

---

## Top 20 cells by |sr_lift| (informational — pre-correction)

These are NOT validated signals. Listed for diagnostic purposes only.

| Strategy | Dir | Thr | N | lift | sr_lift | p_sharpe | yrs+ |
|---|---|---|---|---|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | short | 80 | 85/422 | +0.283 | +0.328 | 0.0060 | 4/4 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | 80 | 98/151 | +0.378 | +0.322 | 0.0200 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | 80 | 115/488 | +0.275 | +0.316 | 0.0030 | 4/4 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | 80 | 79/132 | +0.358 | +0.304 | 0.0270 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | short | 70 | 106/105 | +0.344 | +0.292 | 0.0380 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | 80 | 143/595 | +0.257 | +0.290 | 0.0010 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | 70 | 127/127 | +0.254 | +0.288 | 0.0415 | 4/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | 60 | 150/104 | +0.247 | +0.278 | 0.0535 | 4/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | 70 | 198/540 | +0.233 | +0.260 | 0.0025 | 6/6 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | 80 | 141/511 | +0.231 | +0.260 | 0.0085 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_NOFRI | long | 70 | 166/437 | +0.230 | +0.259 | 0.0075 | 6/6 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | long | 80 | 99/155 | +0.223 | +0.254 | 0.0695 | 4/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | 70 | 106/106 | +0.230 | +0.253 | 0.0620 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | long | 80 | 140/581 | +0.353 | +0.245 | 0.0090 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | long | 70 | 125/124 | +0.288 | +0.242 | 0.0585 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | long | 80 | 141/590 | +0.286 | +0.242 | 0.0140 | 5/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100 | short | 80 | 79/133 | +0.215 | +0.238 | 0.1054 | 4/5 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | 70 | 196/456 | +0.209 | +0.233 | 0.0090 | 5/6 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | long | 60 | 253/485 | +0.203 | +0.225 | 0.0060 | 4/6 |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | long | 60 | 248/404 | +0.190 | +0.210 | 0.0115 | 4/6 |

---

## Honest verdict

**Corrects prior claims:** Prior garch overlay tests ran on unfiltered orb_outcomes, violating the Validated Universe Rule in RESEARCH_RULES.md. At validated-scope (filter-conditional) population, the finding changes materially.

**Deployable signal from garch overlay:** NONE at BH-FDR K=180.

**Implication for NYSE_OPEN SKIP idea:** the earlier claim was based on unfiltered orb_outcomes; at validated-scope (L4 ORB_G5 filter applied), the effect does NOT pass BH-FDR. The SKIP hypothesis is NOT a validated discovery candidate.

**What this DOES justify:**
- Shadow log garch_pct alongside live trades (informational; no code change)
- Pre-register as hypothesis for 6-12 month forward OOS accumulation
- Do NOT register SKIP_GARCH_70 as a new filter until OOS validates at proper scope
