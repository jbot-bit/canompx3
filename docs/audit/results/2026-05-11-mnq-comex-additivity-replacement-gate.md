# MNQ COMEX Additivity / Replacement Gate

**Date:** 2026-05-11
**Profile:** `topstep_50k_mnq_auto`
**Live impact:** None. No DB, schema, broker, validated-setups, or lane-allocation mutation.

## Scope

This is the close-first gate opened by the widened-lens reset. It tests the only two admissible COMEX candidates left by the 2026-05-10 proof gate:

- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5`

Truth sources:

- canonical `gold.db::orb_outcomes`
- canonical `gold.db::daily_features`
- current profile lane definitions / `docs/runtime/lane_allocation.json` context
- canonical filter application via `trading_app.strategy_fitness._load_strategy_outcomes`
- canonical correlation gate algorithm/thresholds from `trading_app.lane_correlation`, applied on the same WF-eligible sample
- instrument eligibility starts from `trading_app.config.WF_START_OVERRIDE`

## Current Book

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

## Common Window

- Common start: `2020-01-29`
- Common end: `2026-04-23`
- IS end: `2025-12-31`
- OOS monitor start: `2026-01-01`

## Current Book Baseline

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current live book | 2020-01-29 | 2025-12-31 | 1546 | 1315 | 1028 | +278.1 | +45.3 | +2.696 | 21.3 | -2.0 | 0.851 |

### 2026 OOS Monitor

| Current live book | 2026-01-01 | 2026-04-23 | 81 | 111 | 70 | +20.9 | +65.1 | +3.119 | 5.4 | -2.0 | 1.370 |

## Decision Matrix

| Candidate | Decision | Add Delta Annual R IS | Add Delta Sharpe IS | Replacement Target | Replace Delta Annual R IS | Replace Delta Sharpe IS | Corr Gate | Worst Rho | Worst Subset |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | `PARK` | +16.7 | +0.136 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | -1.5 | -0.008 | `False` | +0.807 | 68.8% |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | `PARK` | +21.3 | -0.057 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | +3.0 | -0.210 | `False` | +0.807 | 99.8% |

## MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60

- Decision: `PARK`
- Reason: Candidate is same-session with the live COMEX lane and replacement does not improve both common-window IS annualized R and honest Sharpe.
- Corr to aggregate current-book daily R: `+0.329`
- Candidate days overlapping any live lane: `75.6%`
- Canonical correlation gate pass: `False`
- Correlation reject reasons: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100: rho=0.807>0.7`

### IS Snapshots

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 standalone | 2020-01-29 | 2025-12-31 | 1546 | 676 | 676 | +102.7 | +16.7 | +1.748 | 11.5 | -1.0 | 0.437 |
| Current live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 2020-01-29 | 2025-12-31 | 1546 | 1991 | 1193 | +380.8 | +62.1 | +2.831 | 32.3 | -3.0 | 1.288 |
| Replace MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 with MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 2020-01-29 | 2025-12-31 | 1546 | 1472 | 1126 | +268.9 | +43.8 | +2.687 | 16.7 | -2.0 | 0.952 |

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 standalone | 2026-01-01 | 2026-04-23 | 81 | 49 | 49 | +7.5 | +23.4 | +2.025 | 3.1 | -1.0 | 0.605 |
| Current live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 2026-01-01 | 2026-04-23 | 81 | 160 | 71 | +28.5 | +88.5 | +3.117 | 5.5 | -3.0 | 1.975 |
| Replace MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 with MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60 | 2026-01-01 | 2026-04-23 | 81 | 96 | 65 | +20.3 | +63.0 | +3.471 | 6.0 | -2.0 | 1.185 |

## MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5

- Decision: `PARK`
- Reason: Candidate is same-session with the live COMEX lane and replacement does not improve both common-window IS annualized R and honest Sharpe.
- Corr to aggregate current-book daily R: `+0.334`
- Candidate days overlapping any live lane: `67.8%`
- Canonical correlation gate pass: `False`
- Correlation reject reasons: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100: rho=0.807>0.7; subset=99.8%>80%`

### IS Snapshots

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 standalone | 2020-01-29 | 2025-12-31 | 1546 | 1468 | 1468 | +130.4 | +21.3 | +1.516 | 22.4 | -1.0 | 0.950 |
| Current live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | 2020-01-29 | 2025-12-31 | 1546 | 2783 | 1500 | +408.5 | +66.6 | +2.639 | 39.8 | -3.0 | 1.800 |
| Replace MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 with MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | 2020-01-29 | 2025-12-31 | 1546 | 2264 | 1500 | +296.6 | +48.3 | +2.486 | 23.0 | -2.0 | 1.464 |

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 standalone | 2026-01-01 | 2026-04-23 | 81 | 70 | 70 | +5.1 | +16.0 | +1.148 | 5.3 | -1.0 | 0.864 |
| Current live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | 2026-01-01 | 2026-04-23 | 81 | 181 | 72 | +26.1 | +81.1 | +2.660 | 7.9 | -3.0 | 2.235 |
| Replace MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 with MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5 | 2026-01-01 | 2026-04-23 | 81 | 117 | 72 | +17.9 | +55.6 | +2.988 | 7.3 | -2.0 | 1.444 |

## Verdict

Final classifications: `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` = `PARK`, `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` = `PARK`.

No candidate is automatically production-live from this gate. `PASS_REPLACE` means the candidate earns a separate operator preflight / allocation-change review; it is not permission to mutate live state in this pass.

Multiple-testing accounting: fixed K=2 from the proof gate. This pass is role classification against current book, not fresh discovery or threshold search.

Runtime interpretation: both candidates are same-session COMEX O5 alternatives to the deployed COMEX lane. Same-session ADD is not a clean current-stack route; the only admissible live-tradeability role here is replacement, not additive coexistence.

## Reproduction

```bash
./.venv-wsl/bin/python research/mnq_comex_additivity_replacement_gate_2026_05_11.py
```

## Caveats / Limitations

- This gate only classifies the two admissible COMEX candidates left by the
  proof gate.
- Same-session ADD is intentionally treated as runtime-unclean for the current
  profile; replacement is the live-tradeability role tested here.
- A `PARK` result does not delete the research rows. It means no profile/live
  allocation change is justified by this pass.
