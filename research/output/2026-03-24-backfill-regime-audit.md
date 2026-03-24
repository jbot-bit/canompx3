# MNQ Backfill Regime Validity Audit

**Date:** 2026-03-24
**Source:** `orb_outcomes` + `daily_features` (canonical layers only)
**Scope:** MNQ E2 O5 CB1 RR1.0 NO_FILTER, 5 CORE sessions
**Data state:** orb_outcomes through 2026-03-23, bars_1m 2016-02-01 to 2026-03-24

## Finding

Mechanism hypothesis strengthened: edge appears cost-gated, with ATR as amplifier. Current paper spec stays frozen. Any gate implementation requires separate honest-K, WFE, and sensitivity audit.

## Evidence

### Friction gradient (cost/risk decile, 10 years, N=12,483)

| Decile | N | ExpR | cost/risk % | risk$ range |
|--------|---|------|-------------|-------------|
| D1 | 1249 | -0.230 | 33.9% | $3-10 |
| D2 | 1249 | -0.089 | 21.3% | $10-15 |
| D3 | 1249 | +0.012 | 15.0% | $15-21 |
| D4 | 1248 | +0.092 | 11.3% | $21-27 |
| D5 | 1248 | +0.099 | 8.9% | $27-34 |
| D6 | 1248 | +0.096 | 7.1% | $34-44 |
| D7 | 1248 | +0.131 | 5.5% | $44-57 |
| D8 | 1248 | +0.139 | 4.2% | $57-76 |
| D9 | 1248 | +0.156 | 3.0% | $76-109 |
| D10 | 1248 | +0.121 | 1.8% | $109-713 |

MNQ cost model: $2.74 RT (commission $1.24 + spread+slippage $1.50).

### 2x2 friction x ATR matrix (10 years)

| Regime | N | ExpR | p | WR% |
|--------|---|------|---|-----|
| LOW friction + HIGH ATR | 14,429 | +0.143 | <1e-10 | 60.0% |
| LOW friction + LOW ATR | 3,720 | +0.095 | <1e-9 | 58.3% |
| HIGH friction + HIGH ATR | 2,071 | +0.108 | <1e-8 | 64.1% |
| HIGH friction + LOW ATR | 4,881 | -0.063 | <1e-7 | 59.3% |

Friction: cost/risk < 10% vs >= 10%. ATR: atr_20 >= 200 vs < 200.

### Era stability (LOW friction + HIGH ATR)

| Era | N | ExpR | t | p |
|-----|---|------|---|---|
| 2016-2020 | 747 | +0.119 | 3.46 | 0.0005 |
| 2021-2025 | 12,911 | +0.149 | 18.10 | <0.0001 |
| 2026 YTD | 771 | +0.079 | 2.28 | 0.023 |

Positive in ALL eras under the same regime conditions.

### Sensitivity (low friction, ATR threshold +-20%)

| ATR threshold | N | ExpR | t | p |
|---------------|---|------|---|---|
| >= 160 | 16,490 | +0.143 | 19.62 | <1e-10 |
| >= 200 | 14,429 | +0.143 | 18.43 | <1e-10 |
| >= 240 | 11,374 | +0.146 | 16.66 | <1e-10 |

Rock solid. No threshold fragility.

### Cross-instrument (E2 O5 CB1 RR1.0, all sessions, risk$ quintiles)

| Instrument | Q1 ExpR | Q5 ExpR | Q1 cost/risk | Q5 cost/risk |
|------------|---------|---------|-------------|-------------|
| MNQ | -0.250 | +0.127 | 32.7% | 2.7% |
| MGC | -0.386 | +0.048 | 44.5% | 7.6% |
| MES | -0.267 | +0.071 | 30.5% | 5.4% |

Same direction, same gradient, all three instruments.

### Direction (low friction, 10 years)

| Direction | N | ExpR |
|-----------|---|------|
| LONG | 3,814 | +0.102 |
| SHORT | 3,665 | +0.147 |

Shorts outperform. Consistent with faster downside momentum at ORB breaks.

### Relative volume (low friction, preliminary)

| Quintile | N | ExpR | rel_vol range |
|----------|---|------|---------------|
| Q1 | 3,436 | +0.090 | 0.02-0.66 |
| Q5 | 3,435 | +0.198 | 1.54-22.65 |

Research lead only. NOT validated (no honest K, no WFE, no sensitivity).

## What this does NOT change

- Paper book: frozen, pre-registered, unchanged
- LIVE_PORTFOLIO: untouched
- Governance: frozen
- Friction gate: NOT implemented (requires separate honest-K + WFE + sensitivity audit)
- Rel_vol: NOT treated as validated

## Queries used

All queries against `orb_outcomes` joined with `daily_features` where noted.
Friction = `2.74 / risk_dollars * 100`.
ATR from `daily_features.atr_20`.
Rel_vol from `daily_features.rel_vol_{SESSION}`.
Direction inferred from `target_price > entry_price` (LONG) vs `<` (SHORT).
