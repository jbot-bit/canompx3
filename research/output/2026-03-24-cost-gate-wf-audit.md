# Cost/Risk% Gate — Walk-Forward + Honest-K Audit

**Date:** 2026-03-24
**Source:** `orb_outcomes` canonical (gold.db), MNQ E2 O5 RR1.0 CB1, ALL sessions
**Data state:** 2016-02-01 to 2026-03-23, 28,136 trades total, 13,153 after gate (47%)

## Gate Definition

cost/risk% = total_friction / risk_dollars. Gate: < 10%.
MNQ total_friction = $2.74 RT.
Equivalent to: risk_dollars > $27.40 (i.e., ORB risk > ~13.7 NQ points).
Knowable at trade decision time (ORB size known at ORB close, friction is fixed).

## Walk-Forward Result

### GATED (cost/risk% < 10%)
- Windows: 19 (6-month, anchored expanding, 12-month min train)
- Positive: **18/19 (95%)**
- OOS ExpR: **+0.106**
- OOS N: **13,044**
- WFE: **1.00** (OOS matches IS exactly)
- **PASS** (all 4 WF criteria met)
- Only negative window: 2018-02 to 2018-08 (ExpR=-0.006, barely negative)

### UNGATED (all trades)
- Windows: 19
- Positive: 13/19 (68%)
- OOS ExpR: +0.028
- First 6 windows (2017-2020) ALL NEGATIVE
- **PASS** technically but driven entirely by post-2020 regime

## Honest K

- Thresholds tested: 5%, 8%, 10%, 12%, 15%, 20%, 30% = **7 values**
- Instruments: 3 (tested cross-instrument but this audit is MNQ-specific)
- Sessions: ALL (pooled, not selected)
- **K = 7**
- BH threshold at rank 1: 0.05/7 = 0.0071
- MNQ cost<10% p-value: **1.6e-39**
- **Survives BH at K=7: YES**

## Sensitivity (from prior audit, same session)

| Threshold | N | ExpR | p |
|-----------|---|------|---|
| <5% | 5,660 | +0.127 | <1e-8 |
| <8% | 10,277 | +0.113 | <1e-8 |
| <10% | 13,153 | +0.107 | <1e-8 |
| <12% | 15,331 | +0.104 | <1e-8 |
| <15% | 17,909 | +0.096 | <1e-8 |
| <20% | 21,242 | +0.080 | <1e-8 |

Monotonic. No threshold fragility. 10% is not special — any value in the 8-15% range works.

## Cross-Instrument (from prior audit)

| Instrument | cost<10% ExpR | cost>=20% ExpR | Direction |
|------------|--------------|----------------|-----------|
| MNQ | +0.107 | -0.156 | Same |
| MGC | +0.072 | -0.193 | Same |
| MES | +0.063 | -0.202 | Same |

## Era Stability (from prior audit)

| Era | N | ExpR | p |
|-----|---|------|---|
| 2016-2020 | 3,312 | +0.085 | <1e-6 |
| 2021-2025 | 9,262 | +0.115 | <1e-6 |
| 2026 YTD | 579 | +0.101 | 0.010 |

Positive in ALL eras. The "era effect" seen in unfiltered data (negative pre-2021) is
explained by friction composition — pre-2021 had more high-friction trades.

## Verdict

**The cost/risk% < 10% gate survives all institutional gates:**
- Walk-forward: 18/19 positive (95%), WFE=1.00
- BH FDR: p=1.6e-39, K=7, trivially survives
- Sensitivity: monotonic, no threshold fragility
- Cross-instrument: same direction on all 3
- Era stability: positive in all eras
- Mechanism: friction eats the trade (structural, not fitted)

## What This Does NOT Authorize

- NOT a trading rule change (paper spec is frozen)
- NOT a filter implementation (requires separate design + code review)
- NOT tested as a live gate (requires execution realism audit)
- The 10% threshold is not special — 8-15% all work. If implemented, use the
  broadest defensible threshold (e.g., 15%) to maximize trade count.
