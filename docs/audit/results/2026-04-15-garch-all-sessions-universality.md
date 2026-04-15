# Garch Overlay — All-Sessions Universality (Trader Discipline)

**Date:** 2026-04-15
**Scope:** 12 sessions × 3 instruments × 2 directions × 3 apertures × 3 RRs × 3 thresholds = 1944 theoretical cells. 1579 testable (N>=60 total, N>=30 both sides).

**Question:** where does garch overlay work, inversely-work, or produce no signal?

**BH-FDR K_global=1579: 0 survivors at q=0.05 on Sharpe-permutation p-values.**

### Per (instrument, session) tally — sr_lift direction

| Inst | Session | Total | Pos | StrongPos(>0.15) | Neg | StrongNeg(<-0.15) | BH-FDR |
|---|---|---|---|---|---|---|---|
| MES | CME_PRECLOSE | 18 | 18 | 8 | 0 | 0 | 0 |
| MES | CME_REOPEN | 51 | 24 | 11 | 27 | 23 | 0 |
| MES | COMEX_SETTLE | 54 | 32 | 5 | 22 | 2 | 0 |
| MES | EUROPE_FLOW | 54 | 39 | 11 | 15 | 0 | 0 |
| MES | LONDON_METALS | 54 | 46 | 8 | 8 | 1 | 0 |
| MES | NYSE_CLOSE | 27 | 15 | 9 | 12 | 5 | 0 |
| MES | NYSE_OPEN | 54 | 33 | 7 | 21 | 6 | 0 |
| MES | SINGAPORE_OPEN | 54 | 45 | 16 | 9 | 0 | 0 |
| MES | TOKYO_OPEN | 54 | 51 | 21 | 3 | 0 | 0 |
| MES | US_DATA_1000 | 54 | 36 | 5 | 18 | 0 | 0 |
| MES | US_DATA_830 | 54 | 45 | 6 | 9 | 0 | 0 |
| MGC | CME_REOPEN | 33 | 29 | 22 | 4 | 0 | 0 |
| MGC | COMEX_SETTLE | 54 | 40 | 20 | 14 | 3 | 0 |
| MGC | EUROPE_FLOW | 54 | 42 | 15 | 12 | 3 | 0 |
| MGC | LONDON_METALS | 54 | 46 | 15 | 8 | 0 | 0 |
| MGC | NYSE_OPEN | 54 | 27 | 2 | 27 | 1 | 0 |
| MGC | SINGAPORE_OPEN | 54 | 40 | 29 | 14 | 4 | 0 |
| MGC | TOKYO_OPEN | 54 | 48 | 8 | 6 | 0 | 0 |
| MGC | US_DATA_1000 | 54 | 18 | 0 | 36 | 17 | 0 |
| MGC | US_DATA_830 | 54 | 35 | 17 | 19 | 10 | 0 |
| MNQ | BRISBANE_1025 | 54 | 48 | 9 | 6 | 0 | 0 |
| MNQ | CME_PRECLOSE | 21 | 21 | 8 | 0 | 0 | 0 |
| MNQ | CME_REOPEN | 54 | 31 | 21 | 23 | 13 | 0 |
| MNQ | COMEX_SETTLE | 54 | 27 | 10 | 27 | 8 | 0 |
| MNQ | EUROPE_FLOW | 54 | 45 | 2 | 9 | 0 | 0 |
| MNQ | LONDON_METALS | 54 | 21 | 0 | 33 | 4 | 0 |
| MNQ | NYSE_CLOSE | 25 | 15 | 7 | 10 | 6 | 0 |
| MNQ | NYSE_OPEN | 54 | 8 | 0 | 46 | 7 | 0 |
| MNQ | SINGAPORE_OPEN | 54 | 45 | 11 | 9 | 0 | 0 |
| MNQ | TOKYO_OPEN | 54 | 51 | 0 | 3 | 0 | 0 |
| MNQ | US_DATA_1000 | 54 | 31 | 1 | 23 | 1 | 0 |
| MNQ | US_DATA_830 | 54 | 22 | 4 | 32 | 1 | 0 |

### Per-session binomial sign test (H0: garch is random, P(pos)=0.5)

Filters: total >= 6 cells. P-value = P(X >= observed_pos | n=total, p=0.5).

| Inst | Session | Pos/Total | Binomial p | Verdict |
|---|---|---|---|---|
| MES | CME_PRECLOSE | 18/18 | 0.0000 | POSITIVE-LIFT |
| MES | CME_REOPEN | 24/51 | 0.7121 | NULL |
| MES | COMEX_SETTLE | 32/54 | 0.1102 | NULL |
| MES | EUROPE_FLOW | 39/54 | 0.0007 | POSITIVE-LIFT |
| MES | LONDON_METALS | 46/54 | 0.0000 | POSITIVE-LIFT |
| MES | NYSE_CLOSE | 15/27 | 0.3506 | NULL |
| MES | NYSE_OPEN | 33/54 | 0.0668 | NULL |
| MES | SINGAPORE_OPEN | 45/54 | 0.0000 | POSITIVE-LIFT |
| MES | TOKYO_OPEN | 51/54 | 0.0000 | POSITIVE-LIFT |
| MES | US_DATA_1000 | 36/54 | 0.0099 | POSITIVE-LIFT |
| MES | US_DATA_830 | 45/54 | 0.0000 | POSITIVE-LIFT |
| MGC | CME_REOPEN | 29/33 | 0.0000 | POSITIVE-LIFT |
| MGC | COMEX_SETTLE | 40/54 | 0.0003 | POSITIVE-LIFT |
| MGC | EUROPE_FLOW | 42/54 | 0.0000 | POSITIVE-LIFT |
| MGC | LONDON_METALS | 46/54 | 0.0000 | POSITIVE-LIFT |
| MGC | NYSE_OPEN | 27/54 | 0.5540 | NULL |
| MGC | SINGAPORE_OPEN | 40/54 | 0.0003 | POSITIVE-LIFT |
| MGC | TOKYO_OPEN | 48/54 | 0.0000 | POSITIVE-LIFT |
| MGC | US_DATA_1000 | 18/54 | 0.9955 | INVERSE (SKIP CANDIDATE) |
| MGC | US_DATA_830 | 35/54 | 0.0201 | POSITIVE-LIFT |
| MNQ | BRISBANE_1025 | 48/54 | 0.0000 | POSITIVE-LIFT |
| MNQ | CME_PRECLOSE | 21/21 | 0.0000 | POSITIVE-LIFT |
| MNQ | CME_REOPEN | 31/54 | 0.1704 | NULL |
| MNQ | COMEX_SETTLE | 27/54 | 0.5540 | NULL |
| MNQ | EUROPE_FLOW | 45/54 | 0.0000 | POSITIVE-LIFT |
| MNQ | LONDON_METALS | 21/54 | 0.9620 | NULL |
| MNQ | NYSE_CLOSE | 15/25 | 0.2122 | NULL |
| MNQ | NYSE_OPEN | 8/54 | 1.0000 | INVERSE (SKIP CANDIDATE) |
| MNQ | SINGAPORE_OPEN | 45/54 | 0.0000 | POSITIVE-LIFT |
| MNQ | TOKYO_OPEN | 51/54 | 0.0000 | POSITIVE-LIFT |
| MNQ | US_DATA_1000 | 31/54 | 0.1704 | NULL |
| MNQ | US_DATA_830 | 22/54 | 0.9332 | NULL |

### BH-FDR survivors (K=1579, q=0.05)

_No survivors at global K-correction. See per-session tally above for directional evidence._

---

## How to read this

- **POSITIVE-LIFT sessions** (binomial p<0.05, strong_pos dominates): candidates for **R5 SIZER** overlay — size up on garch=HIGH days within this session.
- **INVERSE sessions** (strong_neg dominates): candidates for **R1 SKIP** — avoid trading on garch=HIGH days.
- **NULL sessions**: garch adds no information; do not touch.
- **BH-FDR survivors** at K_global are the most defensible single-cell claims. Non-survivors with directional consistency across thresholds should still be pre-registered as family-level hypotheses per RULE 4.1.
