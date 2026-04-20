# Bull-Short Avoidance — Pooled Deployed OOS

**Script:** `research/bull_short_avoidance_pooled_deployed_oos.py`  
**Date:** 2026-04-20  
**Scope:** pooled shorts across all 6 currently deployed `MNQ` lanes using canonical lane filters

## Verdict

**DEAD**

This is not a hidden allocator or broad sizing edge. The pooled prior-day-direction
short thesis does not exist on the deployed `MNQ` book.

## Canonical result

- IS pooled delta: `+0.0061R`
- IS Welch `p`: `0.8636`
- IS block-bootstrap `p`: `0.8676`
- IS WR spread: `+0.001`
- bear > bull IS years: `5/7`
- OOS delta: `-0.2333R`
- OOS Welch `p`: `0.1625`
- OOS power tier: `STATISTICALLY_USELESS`

## Why this is dead

- The load-bearing kill is the **IS pooled result**, not the OOS sign flip.
- Even before worrying about OOS thinness, the pooled IS signal is basically zero.
- This means the earlier single-lane prior-day direction observation does **not**
  generalize into a book-level allocator or sizing thesis.

## Per-lane decomposition

IS deltas by lane:

- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`: `+0.1574`
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`: `+0.0416`
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`: `+0.0143`
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`: `-0.0295`
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`: `-0.1109`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`: `-0.1251`

The lane effects cancel. There is no coherent pooled signal to allocate against.

## Classification

- **Standalone / allocator / broad sizer use:** `DEAD`
- **Exact single-lane observation:** not resurrected by this test

## Next implication

Do not spend more time on pooled `bull vs bear previous day` short logic.
The live opportunity set is in **prior-day positional states**, not in this pooled
direction split.
