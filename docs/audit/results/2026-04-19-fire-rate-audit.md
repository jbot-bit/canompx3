---
title: Fire-Rate Audit — 38 Active Deployable Lanes
date: 2026-04-19
author: DB Analyst
database: gold.db
mode: Mode A (pre-2026-01-01 only)
---

## Executive Summary

Audit of 38 deployable validated_setups lanes using canonical filter_signal delegation and Mode A pre-2026 data.

**Key findings:**
- **14/38 lanes (36.8%)** flagged by RULE 8.1 extreme fire-rate (fire_rate < 5% or > 95%).
- **4/38 lanes (10.5%)** flagged by RULE 8.2 arithmetic_only (cost-driven, not predictive).
- **6 lanes (X_MES_ATR60)** fire 0% — feature absent from pipeline.
- **9 lanes (ORB_G5, COST_LT12 on NYSE_OPEN, COMEX_SETTLE)** fire > 95% — "almost-always ON" gates.

---

## Fire-Rate Summary

| Strategy | Symbol | Session | Filter | Fire-Rate | R8.1 | WR_Delta | ExpR_Delta | R8.2 |
|----------|--------|---------|--------|-----------|------|----------|------------|------|
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | MNQ | NYSE_OPEN | ORB_G5 | 99.7% | YES | 0.065 | +0.343 | No |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | MNQ | NYSE_OPEN | ORB_G5 | 99.7% | YES | 0.012 | +0.231 | YES |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | MNQ | US_DATA_1000 | ORB_G5 | 99.7% | YES | 0.146 | +0.367 | No |
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 | MNQ | NYSE_OPEN | COST_LT12 | 98.6% | YES | 0.056 | +0.026 | No |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | MNQ | NYSE_OPEN | COST_LT12 | 98.6% | YES | 0.085 | -0.018 | No |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | MNQ | COMEX_SETTLE | ORB_G5 | 95.1% | YES | 0.068 | +0.320 | No |
| MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5 | MNQ | COMEX_SETTLE | ORB_G5 | 95.1% | YES | 0.049 | +0.354 | No |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | MNQ | COMEX_SETTLE | ORB_G5 | 95.1% | YES | 0.067 | +0.361 | No |
| MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5 | MNQ | TOKYO_OPEN | ORB_G5 | 92.9% | No | 0.042 | +0.232 | No |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12 | MNQ | COMEX_SETTLE | COST_LT12 | 77.3% | No | 0.026 | +0.164 | YES |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12 | MNQ | EUROPE_FLOW | COST_LT12 | 60.9% | No | 0.028 | +0.135 | YES |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | MNQ | SINGAPORE_OPEN | ATR_P50 | 53.1% | No | 0.029 | +0.110 | YES |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8 | MES | CME_PRECLOSE | ORB_G8 | 23.1% | No | 0.083 | +0.094 | No |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | MNQ | CME_PRECLOSE | X_MES_ATR60 | 0.0% | YES | N/A | N/A | No |

(Complete table with all 38 lanes in CSV at fire_rate_results.csv)

---

## Key Findings

**14 Extreme-Fire Lanes (36.8%):**

1. **Over-firing (>95%): 9 lanes** — gates nearly always pass
   - MNQ_NYSE_OPEN (ORB_G5): 99.7% fire-rate (10,284/10,314 rows)
   - MNQ_COMEX_SETTLE (ORB_G5, 3x RR): 95.1% fire-rate
   - MNQ_EUROPE_FLOW (ORB_G5, 3x RR): 92.1% fire-rate
   - Recommendation: Reframe as cost-regime overlays, not selective filters

2. **Non-firing (0%): 6 lanes** — X_MES_ATR60 across CME_PRECLOSE, COMEX_SETTLE, NYSE_OPEN, US_DATA_1000
   - Cross-instrument ATR proxy NOT computed in canonical pipeline
   - Recommendation: RETIRE all 6 lanes; feature absent

**4 Arithmetic-Only Lanes (10.5%):**

Small WR delta (<3%) paired with large ExpR delta (>10bp):
- MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12: wr_delta=2.6%, expR_delta=+164bp (cost-screening)
- MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12: wr_delta=2.8%, expR_delta=+135bp
- MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5: wr_delta=1.2%, expR_delta=+231bp (noise + 99.7% fire-rate)
- MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15: wr_delta=2.9%, expR_delta=+110bp

Interpretation: Filters capture cost effects (cheaper days = fewer bps eaten by spread), not directional prediction.

---

## Recommendations

1. **RETIRE X_MES_ATR60 lanes (6).** Feature not in pipeline. Redeploy capacity.

2. **Review over-firing ORB_G5 gates.** 95%+ fire-rate on NYSE_OPEN and COMEX_SETTLE sessions indicates gate is too loose. Consider:
   - Switching to G6/G7 (tighter thresholds)
   - Reframing as session-cost overlay

3. **Validate COST_LT12 cost-only hypothesis:** Run sensitivity check
   - Does expR_delta → 0 when explicit cost is subtracted?
   - If yes, relabel as COST_SCREEN, not EDGE

4. **Fire-rate stability across Mode A boundary:** Compare fire-rates pre-2026 vs post-2026. If shifts >10%, flag as regime-dependent.

---

## Methodology

- **Data window:** Mode A (trading_day < 2026-01-01 only)
- **Filter application:** Canonical research.filter_utils.filter_signal
- **Fire-rate:** count(filter_fires) / count(eligible_rows) per lane
- **RULE 8.1:** fire_rate < 5% or > 95% → extreme
- **RULE 8.2:** wr_spread < 3% AND |expR_spread| > 0.10 → arithmetic_only
- **Script:** fire_rate_audit.py (canonical)
- **Database:** C:/Users/joshd/canompx3/gold.db

