# Full Discovery Execution Plan — Remaining Work

**Date:** 2026-04-09
**Status:** PLAN — for next session(s)
**Context:** This session completed the audit, redesign, Amendment 3.0, and initial G5-only discovery runs. The following work remains to fully test all honest edges.

## What's done (this session)

1. Bailey calendar year metric audit + MGC N=7→5 correction (commit f3523ae)
2. Discovery redesign — K=5 mechanism-grounded hypothesis files (commit f4fc23d)
3. Amendment 3.0 — Pathway B individual hypothesis testing (commit ce450fc)
4. G5-only discovery ran for all 3 instruments (15 strategies total)
5. Results: MNQ 2 FDR sig (NYSE_OPEN, EUROPE_FLOW), MES 0, MGC 0
6. Pathway B candidates: 4 strategies with raw p < 0.05 + positive Sharpe
7. Validator currently outputs 1 validated (MNQ NYSE_OPEN G5) — stratified K kills others

## What remains

### Phase 1: Validator Pathway B support (code change)

**Why:** Amendment 3.0 adds Pathway B to Criterion 3 (individual hypothesis testing at raw p < 0.05). But the validator code (`trading_app/strategy_validator.py`) still applies stratified BH FDR unconditionally. Without a code change, Pathway B is a document, not a working pipeline.

**Change:** When the hypothesis file declares `testing_mode: individual`, the validator should:
- Skip the BH FDR gate
- Apply raw p < 0.05 + positive Sharpe as the significance gate
- ENFORCE criteria 6 (WFE ≥ 0.50), 8 (2026 OOS), 9 (era stability) as non-waivable
- Report cumulative hypothesis count for transparency

**Blast radius:** `trading_app/strategy_validator.py` only. The hypothesis_loader already reads metadata.

### Phase 2: Comprehensive Pathway B hypothesis files

Write individual Pathway B hypothesis files covering the FULL honest filter toolset per instrument. Each hypothesis is a single mechanism-session-filter prediction, tested at raw p < 0.05.

**MNQ hypotheses (broadest search space):**
| # | Session | Filter | RR | Mechanism | Prior evidence |
|---|---|---|---|---|---|
| 1 | NYSE_OPEN | ORB_G5 | 2.0 | Crabel commitment | p=0.017 (K=5 redesign) |
| 2 | EUROPE_FLOW | ORB_G5 | 2.0 | Cross-border spillover | p=0.011 (K=5 redesign) |
| 3 | COMEX_SETTLE | ORB_G5 | 2.0 | Settlement microstructure | p=0.054 (near-miss) |
| 4 | NYSE_OPEN | OVNRNG_50 | 2.0 | Overnight info (Ito-Engle-Lin) | p=0.029 (prior K=16) |
| 5 | NYSE_OPEN | ORB_G5 | 1.0 | Crabel minimum continuation | Strongest unfiltered baseline |
| 6 | EUROPE_FLOW | ORB_G5 | 1.0 | Cross-border minimum continuation | Second strongest baseline |
| 7 | CME_PRECLOSE | ORB_G5 | 1.0 | End-of-day rebalancing | Strongest unfiltered session |

**MES hypotheses (narrower — cost constraints):**
| # | Session | Filter | RR | Mechanism | Prior evidence |
|---|---|---|---|---|---|
| 1 | NYSE_OPEN | ORB_G5 | 2.0 | Crabel on S&P benchmark | p=0.026 |
| 2 | NYSE_OPEN | ORB_G6 | 2.0 | Crabel monotonic | p=0.032 |
| 3 | NYSE_OPEN | ORB_G5 | 1.0 | Crabel minimum continuation | Untested |
| 4 | NYSE_OPEN | ORB_G6 | 1.0 | Monotonic at minimum continuation | Untested |
| 5 | US_DATA_1000 | ORB_G5 | 1.0 | Post-equity macro at min continuation | Second viable session |

**MGC hypotheses (tight data — focus on validated pre-session signals):**
| # | Session | Filter | RR | Mechanism | Prior evidence |
|---|---|---|---|---|---|
| 1 | LONDON_METALS | ORB_G5 | 2.0 | London gold pricing | p=0.35 (3.55yr) |
| 2 | LONDON_METALS | PDR_R080 | 2.0 | Prior day vol persistence | p=0.003 (validated) |
| 3 | CME_REOPEN | GAP_R005 | 2.0 | Overnight gap info | p=0.009 (validated) |
| 4 | CME_REOPEN | ORB_G5 | 2.0 | Reopen momentum | p=0.161 (3.55yr) |
| 5 | LONDON_METALS | ORB_G5 | 1.0 | London gold min continuation | Untested at RR1.0 |

**Total: ~17 individual hypotheses across 3 instruments.** Each tested at K=1 (Pathway B).

### Phase 3: Discovery execution

Run discovery for all new hypothesis files. Each instrument sequentially (DuckDB single-writer).

### Phase 4: Validation (Pathway B gates)

For everything with raw p < 0.05 + positive Sharpe:
- Walk-forward efficiency ≥ 0.50 (non-waivable)
- 2026 OOS ExpR > 0 and ≥ 0.40 × IS ExpR (non-waivable)
- Era stability: no era with N ≥ 50 and ExpR < -0.05 (non-waivable)
- Sample size N ≥ 100

### Phase 5: Portfolio construction

Build the deployed portfolio from Pathway B survivors. Update prop_profiles. No grandfather clause.

## Execution order

1. Phase 1 first (validator code change) — gates everything else
2. Phase 2 (write hypothesis files) — can be parallelized across instruments
3. Phase 3 (discovery) — sequential, ~5 min per instrument
4. Phase 4 (validation) — automatic via updated validator
5. Phase 5 (portfolio) — after validation results known

## Expected outcomes (honest)

- MNQ: 3-5 Pathway B survivors (NYSE_OPEN + EUROPE_FLOW confirmed; COMEX_SETTLE, CME_PRECLOSE, RR1.0 variants possible)
- MES: 1-3 Pathway B survivors (NYSE_OPEN G5/G6 at RR2.0; RR1.0 variants may be stronger)
- MGC: 0-2 Pathway B survivors (PDR and GAP have strong prior evidence but 3.55yr data limits detection)
- Portfolio: 4-10 honest, mechanism-grounded, forward-validated strategies
