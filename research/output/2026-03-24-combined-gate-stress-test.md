# Combined Gate Stress Test — Friction + Timeout

**Date:** 2026-03-24
**Source:** `orb_outcomes` + `daily_features` (canonical layers only)
**Gate:** cost/risk < 10% AND break_delay_min <= 10
**Data:** MNQ E2 O5 CB1 RR1.0, 10 years (2016-02-01 to 2026-03-23)

## RECLASSIFICATION (post quant-audit-protocol merge, 2026-03-24)

**The combined gate has TWO components with different classifications:**

| Component | Classification | Evidence |
|-----------|---------------|---------|
| Friction (cost/risk <10%) | **ARITHMETIC_ONLY** | WR flat across friction bins (<3% spread). Payoff improves because costs eat less of wins. Tautology with G-filters (corr = -1.0 by construction). Correct framing: cost screen / minimum trade size gate. |
| Timeout (delay <=10m) | **SIGNAL** (pending T3-T8) | WR spread 54-64% across delay quintiles. WR improvement persists controlling for friction. Mechanism: order flow concentration at ORB edge. |

**Honest framing: "cost screen + conviction signal", NOT "dual mechanism."**

## Gate Sequence Status

| Gate | Status | Evidence |
|------|--------|---------|
| 1. Mechanism | PARTIAL | Timeout = order flow concentration (Chan). Friction = arithmetic cost screen, not mechanism. |
| 2. Baseline | PASS | Monotonic gradient, 10yr, 3 instruments |
| 3. BH FDR | PASS | K=64 (8x8 grid), all p<0.001. Bootstrap p=0.001 |
| 4. Walk-forward | PASS | 8/8 OOS positive, avg WFE=1.08 |
| 5. Sensitivity | PASS | 3x3 grid all positive (+0.141 to +0.164) |
| 6. Replay | NOT DONE | Needs paper_trader execution |
| 7. Paper trade | EXISTS | Paper book pre-registered (unfiltered, monitoring only) |

## Stress Tests

### 1. Bootstrap Permutation (1000 perms)
Real: N=24,150, ExpR=+0.1472
Permutations exceeding real: 0/1000
Bootstrap p = 0.001 (Phipson & Smyth correction)
VERDICT: PASS — not random selection.

### 2. Included vs Excluded
Included (gate ON):  N=24,150  ExpR=+0.1472
Excluded (gate OFF): N=58,346  ExpR=-0.0574
Spread: +0.2046
VERDICT: PASS — gate separates signal from noise.

### 3. Direction
LONG:  N=12,357  ExpR=+0.1416
SHORT: N=11,793  ExpR=+0.1531
VERDICT: PASS — both sides, no bull market bias.

### 4. Cost Model Stress
| Cost mult | N | Raw ExpR | Adj ExpR | Extra friction |
|-----------|---|----------|----------|----------------|
| 1.0x | 24,150 | +0.1472 | +0.1472 | 0.0000R |
| 1.5x | 15,412 | +0.1645 | +0.1479 | 0.0165R |
| 2.0x | 10,462 | +0.1753 | +0.1478 | 0.0275R |

VERDICT: PASS — self-healing. Higher costs filter more aggressively, per-trade quality holds.

### 5. Cross-Instrument
| Instrument | N | ExpR | Cost used |
|------------|---|------|-----------|
| MNQ | 24,150 | +0.1472 | $2.74 |
| MGC | 3,676 | +0.1477 | $5.74 |
| MES | 8,512 | +0.1156 | $3.74 |

VERDICT: PASS — all 3 positive. Same mechanism.

### 6. RR Target Robustness
| RR | N | ExpR |
|----|---|------|
| 1.0 | 24,150 | +0.1472 |
| 1.5 | 23,278 | +0.1693 |
| 2.0 | 22,631 | +0.1774 |

VERDICT: PASS — strengthens at higher RR. NOTE: suggests optimal RR may be >1.0 under combined gate.

### 7. Aperture
| Aperture | N | ExpR |
|----------|---|------|
| O5m | 24,150 | +0.1472 |
| O15m | 29,492 | +0.1133 |
| O30m | 30,953 | +0.0631 |

VERDICT: PARTIAL — timeout calibrated for O5m. O30m needs separate timeout.

### 8. Drawdown Character (4 sessions)
- Trades: 12,795
- Max consecutive losses: 16
- Max drawdown: 36.6R
- Final cumulative: +1,924.2R
- Reward/DD ratio: 52.5:1

### 9. Per-Year (0/10 negative)
| Year | N | ExpR |
|------|---|------|
| 2016 | 152 | +0.141 |
| 2017 | 154 | +0.244 |
| 2018 | 1,017 | +0.086 |
| 2019 | 748 | +0.111 |
| 2020 | 1,554 | +0.124 |
| 2021 | 1,519 | +0.148 |
| 2022 | 1,949 | +0.194 |
| 2023 | 1,653 | +0.113 |
| 2024 | 1,785 | +0.213 |
| 2025 | 1,836 | +0.158 |
| 2026 | 428 | +0.096 |

### 10. Walk-Forward (8/8 positive)
| OOS Year | IS ExpR | OOS ExpR | WFE |
|----------|---------|----------|-----|
| 2018 | +0.193 | +0.086 | 0.43 |
| 2019 | +0.111 | +0.111 | 1.00 |
| 2020 | +0.111 | +0.124 | 1.09 |
| 2021 | +0.117 | +0.148 | 1.26 |
| 2022 | +0.126 | +0.194 | 1.53 |
| 2023 | +0.145 | +0.113 | 0.78 |
| 2024 | +0.139 | +0.213 | 1.56 |
| 2025 | +0.151 | +0.158 | 1.03 |

Avg WFE: 1.08 | Threshold: >0.50 | Positive OOS: 8/8

### 11. All 12 Sessions (no cherry-pick)
ALL sessions positive at p<0.005 under combined gate.
| Session | N | ExpR | p |
|---------|---|------|---|
| NYSE_CLOSE | 945 | +0.220 | <0.001 |
| CME_PRECLOSE | 2,045 | +0.217 | <0.001 |
| SINGAPORE_OPEN | 992 | +0.210 | <0.001 |
| EUROPE_FLOW | 1,743 | +0.198 | <0.001 |
| TOKYO_OPEN | 1,439 | +0.161 | <0.001 |
| COMEX_SETTLE | 2,691 | +0.141 | <0.001 |
| NYSE_OPEN | 4,284 | +0.141 | <0.001 |
| US_DATA_1000 | 3,775 | +0.131 | <0.001 |
| US_DATA_830 | 2,007 | +0.109 | <0.001 |
| CME_REOPEN | 1,130 | +0.109 | <0.001 |
| BRISBANE_1025 | 782 | +0.096 | 0.003 |
| LONDON_METALS | 2,317 | +0.095 | <0.001 |

### 12. Correlation between dimensions
Pearson r = 0.035 (friction vs delay). INDEPENDENT signals.

### 13. 3x3 Sensitivity Grid
| | t<=8m | t<=10m | t<=12m |
|---|---|---|---|
| f<8% | +0.164 | +0.159 | +0.152 |
| f<10% | +0.157 | +0.150 | +0.144 |
| f<12% | +0.154 | +0.148 | +0.141 |

All 9 cells positive. No cliff. No fragility.

## Honest Flags

1. O30m aperture is weak (+0.063). Timeout is O5m-calibrated.
2. MGC N=3,676 needs per-year stability check.
3. Max 16 consecutive losses = real risk. ~-$1,600 on 2 micros.
4. RR2.0 > RR1.0 under gate — paper book at RR1.0 may not be optimal.
5. Bootstrap shuffles break autocorrelation — conservative block shuffle would be more rigorous.
6. Gate Sequence Gate 6 (Replay) not yet executed.

## Academic Grounding

- Aronson (Evidence-Based TA, Ch 6): parameter perturbation test — all 9 cells pass.
- De Prado (AFML, Ch 8): backtest overfitting probability — K=64, bootstrap p=0.001.
- Bailey & de Prado (Deflated Sharpe, 2014): WFE > 50% threshold — avg WFE=1.08.
- Chan (Algorithmic Trading, Ch 3): breakout timing = order flow concentration.
- Carver (Systematic Trading, Ch 7): simple rules dominate — 2 parameters only.

## Status

This finding is a RESEARCH RESULT, not a trading rule.
Paper book remains frozen and unchanged.
No code changes. No LIVE_PORTFOLIO changes.
Implementation requires: Gate 6 (replay) + explicit human sign-off.
