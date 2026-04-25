# Prior-Day Geometry Routing Audit

Date: 2026-04-23

## Scope

Resolve the role of the five promoted MNQ prior-day geometry shelf survivors against the current live MNQ book.

Truth used for this audit:

- `gold.db::orb_outcomes`
- `gold.db::daily_features`

Comparison-only deployment context:

- `docs/runtime/lane_allocation.json`
- `validated_setups` for exact live-lane and candidate parameters
- `trading_app/prop_profiles.py`

## Live Book Context

- Profile: `topstep_50k_mnq_auto`
- Current live lane count: `6`
- `max_slots=7`
- Free slots before any replacement question: `1`

Live lanes reconstructed in this audit:

- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` -> EUROPE_FLOW O5 RR1.5 ORB_G5
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` -> SINGAPORE_OPEN O15 RR1.5 ATR_P50
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` -> COMEX_SETTLE O5 RR1.5 ORB_G5
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` -> NYSE_OPEN O5 RR1.0 COST_LT12
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` -> TOKYO_OPEN O5 RR1.5 COST_LT12
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` -> US_DATA_1000 O15 RR1.5 ORB_G5

## Current Live Book Baseline (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Live book | 2019-05-20 | 2025-12-31 | 1728 | 8113 | 1707 | +749.3 | +109.3 | +2.803 | 40.1 | -6.0 | 4.695 |

## Current Live Book Baseline (2026 OOS Monitor Only)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Live book | 2026-01-01 | 2026-04-02 | 66 | 360 | 65 | +52.6 | +200.9 | +4.219 | 8.4 | -4.0 | 5.455 |

## Decision Summary

| Candidate | Session | RR | Standalone Annual R IS | Add Δ Annual R IS | Add Δ Sharpe IS | Replacement target | Replace Δ Annual R IS | Replace Δ Sharpe IS | Decision |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | US_DATA_1000 | 1.0 | +6.9 | +6.9 | +0.137 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -15.4 | -0.146 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | US_DATA_1000 | 1.0 | +7.2 | +7.2 | +0.126 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -15.1 | -0.155 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | US_DATA_1000 | 1.0 | +9.1 | +9.1 | +0.155 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -13.2 | -0.118 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | US_DATA_1000 | 1.5 | +10.3 | +10.3 | +0.133 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -12.0 | -0.114 | `KEEP_ON_SHELF` |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | COMEX_SETTLE | 1.0 | +7.7 | +7.7 | +0.083 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | -12.7 | -0.050 | `KEEP_ON_SHELF` |

## Primary Routing Winner

- No candidate cleared the additive free-slot route test.

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG

- Decision: `KEEP_ON_SHELF`
- Reason: Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / size-down handling. Keep it on shelf until that translation is modeled or shadow-tested.

### Standalone / Add / Replace (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG standalone | 2019-05-20 | 2025-12-31 | 1728 | 205 | 205 | +47.6 | +6.9 | +1.345 | 5.2 | -1.0 | 0.119 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG | 2019-05-20 | 2025-12-31 | 1728 | 8318 | 1707 | +796.8 | +116.2 | +2.939 | 37.7 | -6.0 | 4.814 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG | 2019-05-20 | 2025-12-31 | 1728 | 6837 | 1699 | +643.7 | +93.9 | +2.657 | 45.8 | -5.0 | 3.957 |

### Profile / Routing Fit

- Fits current auto profile static gates: `YES`
- Other active profile fits: `none`
- Inactive configured profile fits: `self_funded_tradovate, topstep_100k_type_a, topstep_50k_type_a, tradeify_100k_type_b, tradeify_50k, tradeify_50k_type_b`
- Watch status if not auto-routed: `KEEP_VISIBLE` (positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)

- Additive annualized R delta vs live book: `+6.9`
- Additive honest Sharpe delta vs live book: `+0.137`
- Same-session replacement target: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Replacement annualized R delta vs live book: `-15.4`
- Replacement honest Sharpe delta vs live book: `-0.146`

### Diversification vs Live Book (IS)

| Live lane | Daily corr | Overlap days | Overlap % of candidate days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | -0.005 | 191 | 93.2% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.013 | 95 | 46.3% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | -0.010 | 194 | 94.6% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.056 | 199 | 97.1% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | -0.019 | 122 | 59.5% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.061 | 183 | 89.3% |

- Corr to aggregate live-book daily R: `+0.040`
- Candidate trade days: `205`
- Candidate days overlapping any live lane: `205` (`100.0%`)
- Avg live lanes active on candidate days: `4.800`

### Same-Session Substitution View

- Current live same-session lane: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Corr vs current same-session lane: `+0.061`
- Overlap vs current same-session lane: `183` days (`89.3%` of candidate days)

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG standalone | 2026-01-01 | 2026-04-02 | 66 | 10 | 10 | +1.7 | +6.7 | +1.101 | 2.0 | -1.0 | 0.152 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG | 2026-01-01 | 2026-04-02 | 66 | 370 | 65 | +54.4 | +207.6 | +4.371 | 7.4 | -4.0 | 5.606 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG | 2026-01-01 | 2026-04-02 | 66 | 315 | 65 | +45.5 | +173.8 | +4.094 | 10.7 | -4.0 | 4.773 |

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG

- Decision: `KEEP_ON_SHELF`
- Reason: Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / size-down handling. Keep it on shelf until that translation is modeled or shadow-tested.

### Standalone / Add / Replace (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG standalone | 2019-05-20 | 2025-12-31 | 1728 | 231 | 231 | +49.3 | +7.2 | +1.317 | 8.0 | -1.0 | 0.134 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | 2019-05-20 | 2025-12-31 | 1728 | 8344 | 1707 | +798.6 | +116.5 | +2.929 | 40.7 | -6.0 | 4.829 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | 2019-05-20 | 2025-12-31 | 1728 | 6863 | 1698 | +645.4 | +94.1 | +2.647 | 48.5 | -6.0 | 3.972 |

### Profile / Routing Fit

- Fits current auto profile static gates: `YES`
- Other active profile fits: `none`
- Inactive configured profile fits: `self_funded_tradovate, topstep_100k_type_a, topstep_50k_type_a, tradeify_100k_type_b, tradeify_50k, tradeify_50k_type_b`
- Watch status if not auto-routed: `KEEP_VISIBLE` (positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)

- Additive annualized R delta vs live book: `+7.2`
- Additive honest Sharpe delta vs live book: `+0.126`
- Same-session replacement target: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Replacement annualized R delta vs live book: `-15.1`
- Replacement honest Sharpe delta vs live book: `-0.155`

### Diversification vs Live Book (IS)

| Live lane | Daily corr | Overlap days | Overlap % of candidate days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | +0.018 | 209 | 90.5% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.014 | 114 | 49.4% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | +0.045 | 215 | 93.1% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.041 | 226 | 97.8% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | -0.020 | 132 | 57.1% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.071 | 209 | 90.5% |

- Corr to aggregate live-book daily R: `+0.074`
- Candidate trade days: `231`
- Candidate days overlapping any live lane: `231` (`100.0%`)
- Avg live lanes active on candidate days: `4.784`

### Same-Session Substitution View

- Current live same-session lane: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Corr vs current same-session lane: `+0.071`
- Overlap vs current same-session lane: `209` days (`90.5%` of candidate days)

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG standalone | 2026-01-01 | 2026-04-02 | 66 | 11 | 11 | +0.7 | +2.8 | +0.434 | 3.1 | -1.0 | 0.167 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | 2026-01-01 | 2026-04-02 | 66 | 371 | 65 | +53.3 | +203.6 | +4.204 | 7.8 | -4.0 | 5.621 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG | 2026-01-01 | 2026-04-02 | 66 | 316 | 65 | +44.5 | +169.9 | +3.916 | 11.7 | -4.0 | 4.788 |

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG

- Decision: `KEEP_ON_SHELF`
- Reason: Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / size-down handling. Keep it on shelf until that translation is modeled or shadow-tested.

### Standalone / Add / Replace (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG standalone | 2019-05-20 | 2025-12-31 | 1728 | 349 | 349 | +62.7 | +9.1 | +1.362 | 10.2 | -1.0 | 0.202 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | 2019-05-20 | 2025-12-31 | 1728 | 8462 | 1707 | +812.0 | +118.4 | +2.958 | 39.0 | -6.0 | 4.897 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | 2019-05-20 | 2025-12-31 | 1728 | 6981 | 1699 | +658.8 | +96.1 | +2.685 | 46.8 | -6.0 | 4.040 |

### Profile / Routing Fit

- Fits current auto profile static gates: `YES`
- Other active profile fits: `none`
- Inactive configured profile fits: `self_funded_tradovate, topstep_100k_type_a, topstep_50k_type_a, tradeify_100k_type_b, tradeify_50k, tradeify_50k_type_b`
- Watch status if not auto-routed: `KEEP_VISIBLE` (positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)

- Additive annualized R delta vs live book: `+9.1`
- Additive honest Sharpe delta vs live book: `+0.155`
- Same-session replacement target: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Replacement annualized R delta vs live book: `-13.2`
- Replacement honest Sharpe delta vs live book: `-0.118`

### Diversification vs Live Book (IS)

| Live lane | Daily corr | Overlap days | Overlap % of candidate days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | +0.009 | 323 | 92.6% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.007 | 172 | 49.3% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | +0.030 | 326 | 93.4% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.056 | 342 | 98.0% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | -0.016 | 197 | 56.4% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.080 | 316 | 90.5% |

- Corr to aggregate live-book daily R: `+0.071`
- Candidate trade days: `349`
- Candidate days overlapping any live lane: `349` (`100.0%`)
- Avg live lanes active on candidate days: `4.802`

### Same-Session Substitution View

- Current live same-session lane: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Corr vs current same-session lane: `+0.080`
- Overlap vs current same-session lane: `316` days (`90.5%` of candidate days)

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG standalone | 2026-01-01 | 2026-04-02 | 66 | 13 | 13 | +2.7 | +10.2 | +1.476 | 3.0 | -1.0 | 0.197 |
| Live book + MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | 2026-01-01 | 2026-04-02 | 66 | 373 | 65 | +55.3 | +211.1 | +4.370 | 7.4 | -4.0 | 5.652 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG | 2026-01-01 | 2026-04-02 | 66 | 318 | 65 | +46.4 | +177.3 | +4.100 | 10.7 | -4.0 | 4.818 |

## MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG

- Decision: `KEEP_ON_SHELF`
- Reason: Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / size-down handling. Keep it on shelf until that translation is modeled or shadow-tested.

### Standalone / Add / Replace (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG standalone | 2019-05-20 | 2025-12-31 | 1728 | 346 | 346 | +71.0 | +10.3 | +1.207 | 8.9 | -1.0 | 0.200 |
| Live book + MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | 2019-05-20 | 2025-12-31 | 1728 | 8459 | 1707 | +820.2 | +119.6 | +2.936 | 39.1 | -7.0 | 4.895 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | 2019-05-20 | 2025-12-31 | 1728 | 6978 | 1699 | +667.1 | +97.3 | +2.689 | 46.2 | -6.0 | 4.038 |

### Profile / Routing Fit

- Fits current auto profile static gates: `YES`
- Other active profile fits: `none`
- Inactive configured profile fits: `self_funded_tradovate, topstep_100k_type_a, topstep_50k_type_a, tradeify_100k_type_b, tradeify_50k, tradeify_50k_type_b`
- Watch status if not auto-routed: `KEEP_VISIBLE` (positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)

- Additive annualized R delta vs live book: `+10.3`
- Additive honest Sharpe delta vs live book: `+0.133`
- Same-session replacement target: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Replacement annualized R delta vs live book: `-12.0`
- Replacement honest Sharpe delta vs live book: `-0.114`

### Diversification vs Live Book (IS)

| Live lane | Daily corr | Overlap days | Overlap % of candidate days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | +0.006 | 320 | 92.5% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.004 | 170 | 49.1% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | +0.035 | 324 | 93.6% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.050 | 339 | 98.0% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | -0.030 | 196 | 56.6% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.157 | 316 | 91.3% |

- Corr to aggregate live-book daily R: `+0.100`
- Candidate trade days: `346`
- Candidate days overlapping any live lane: `346` (`100.0%`)
- Avg live lanes active on candidate days: `4.812`

### Same-Session Substitution View

- Current live same-session lane: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`
- Corr vs current same-session lane: `+0.157`
- Overlap vs current same-session lane: `316` days (`91.3%` of candidate days)

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG standalone | 2026-01-01 | 2026-04-02 | 66 | 13 | 13 | +1.7 | +6.3 | +0.731 | 4.1 | -1.0 | 0.197 |
| Live book + MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | 2026-01-01 | 2026-04-02 | 66 | 373 | 65 | +54.3 | +207.2 | +4.288 | 9.4 | -4.0 | 5.652 |
| Live book with MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 replaced by MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG | 2026-01-01 | 2026-04-02 | 66 | 318 | 65 | +45.4 | +173.4 | +4.050 | 12.2 | -4.0 | 4.818 |

## MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG

- Decision: `KEEP_ON_SHELF`
- Reason: Research-only additive math is positive, but this is a same-session live-book collision. Under the current runtime, same-session adds are not a clean free-slot route: same-aperture adds are blocked and different-aperture adds require execution translation / size-down handling. Keep it on shelf until that translation is modeled or shadow-tested.

### Standalone / Add / Replace (IS)

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG standalone | 2019-05-20 | 2025-12-31 | 1728 | 336 | 336 | +52.6 | +7.7 | +1.238 | 8.9 | -1.0 | 0.194 |
| Live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 2019-05-20 | 2025-12-31 | 1728 | 8449 | 1707 | +801.9 | +116.9 | +2.886 | 41.4 | -7.0 | 4.889 |
| Live book with MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 replaced by MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 2019-05-20 | 2025-12-31 | 1728 | 6902 | 1707 | +662.5 | +96.6 | +2.752 | 32.9 | -6.0 | 3.994 |

### Profile / Routing Fit

- Fits current auto profile static gates: `YES`
- Other active profile fits: `none`
- Inactive configured profile fits: `bulenox_50k, self_funded_tradovate, topstep_100k_type_a, topstep_50k_type_a, tradeify_100k_type_b, tradeify_50k, tradeify_50k_type_b`
- Watch status if not auto-routed: `KEEP_VISIBLE` (positive shelf row; preserve for other-profile or manual review rather than collapsing to dead)

- Additive annualized R delta vs live book: `+7.7`
- Additive honest Sharpe delta vs live book: `+0.083`
- Same-session replacement target: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- Replacement annualized R delta vs live book: `-12.7`
- Replacement honest Sharpe delta vs live book: `-0.050`

### Diversification vs Live Book (IS)

| Live lane | Daily corr | Overlap days | Overlap % of candidate days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | -0.043 | 316 | 94.0% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.022 | 163 | 48.5% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | +0.353 | 318 | 94.6% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.051 | 333 | 99.1% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | +0.002 | 184 | 54.8% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.019 | 303 | 90.2% |

- Corr to aggregate live-book daily R: `+0.173`
- Candidate trade days: `336`
- Candidate days overlapping any live lane: `336` (`100.0%`)
- Avg live lanes active on candidate days: `4.812`

### Same-Session Substitution View

- Current live same-session lane: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- Corr vs current same-session lane: `+0.353`
- Overlap vs current same-session lane: `318` days (`94.6%` of candidate days)

### 2026 OOS Monitor Only

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG standalone | 2026-01-01 | 2026-04-02 | 66 | 14 | 14 | +3.0 | +11.4 | +1.669 | 2.0 | -1.0 | 0.212 |
| Live book + MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 2026-01-01 | 2026-04-02 | 66 | 374 | 65 | +55.6 | +212.3 | +4.360 | 7.8 | -4.0 | 5.667 |
| Live book with MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 replaced by MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 2026-01-01 | 2026-04-02 | 66 | 314 | 65 | +51.6 | +197.0 | +4.895 | 6.5 | -5.0 | 4.758 |

## Verdict

The prior-day geometry branch is no longer a discovery problem.
It is a routing problem.

Selection rule used here:

- free-slot additive test first
- same-session candidates are not treated as direct free-slot adds under the current runtime
- current auto-profile static gates must allow the candidate
- same-session replacement view second
- positive shelf rows that are not first-route winners remain visible for other-profile / manual-review use
- 2026 OOS monitor-only, not selection proof

## Reproduction

```bash
./.venv-wsl/bin/python research/prior_day_geometry_routing_audit.py
```

No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.
