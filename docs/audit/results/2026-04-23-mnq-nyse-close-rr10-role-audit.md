# MNQ NYSE_CLOSE RR1.0 Role Audit

Date: 2026-04-23

## Scope

Resolve the remaining honest question after the exact `ORB_G8` NYSE_CLOSE prereg was killed:
does broad unfiltered `MNQ NYSE_CLOSE E2 RR1.0 CB1 O5` belong as an allocator add, a standalone monitor only, or nowhere?

Truth used for this audit:

- `gold.db::orb_outcomes`
- `gold.db::daily_features`

Comparison-only deployment context:

- `docs/runtime/lane_allocation.json`
- `validated_setups` for exact live-lane parameters
- `trading_app/prop_profiles.py`

## Live Book Context

- Profile: `topstep_50k_mnq_auto`
- `max_slots=7`
- Current deployed MNQ lanes in allocation JSON: `6`
- Free slot exists before any replacement question: `True`
- `NYSE_CLOSE` is currently excluded from `allowed_sessions` and from `build_raw_baseline_portfolio(... exclude_sessions={"NYSE_CLOSE"})`.

Live lanes reconstructed in this audit:

- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` -> EUROPE_FLOW O5 RR1.5 ORB_G5
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` -> SINGAPORE_OPEN O15 RR1.5 ATR_P50
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` -> COMEX_SETTLE O5 RR1.5 ORB_G5
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` -> NYSE_OPEN O5 RR1.0 COST_LT12
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` -> TOKYO_OPEN O5 RR1.5 COST_LT12
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` -> US_DATA_1000 O15 RR1.5 ORB_G5

Candidate under test: `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_NO_FILTER` -> NYSE_CLOSE O5 RR1.0 NO_FILTER

## Honest Portfolio Comparison

Metrics use full business-day calendars on a common window so idle days dilute Sharpe honestly.

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Live 6-lane book | 2019-05-13 | 2025-12-31 | 1733 | 8140 | 1712 | +756.9 | +110.1 | +2.823 | 40.1 | -6.0 | 4.697 |
| Live book + NYSE_CLOSE | 2019-05-13 | 2025-12-31 | 1733 | 8942 | 1712 | +822.1 | +119.5 | +2.959 | 44.9 | -7.0 | 5.160 |
| NYSE_CLOSE standalone | 2019-05-13 | 2025-12-31 | 1733 | 802 | 802 | +65.1 | +9.5 | +0.975 | 21.8 | -1.0 | 0.463 |

IS deltas from adding NYSE_CLOSE to the live 6-lane book:

- Annualized R delta: `+9.5`
- Honest Sharpe delta: `+0.136`
- Max drawdown delta: `+4.7`

## Candidate Diversification vs Live Book (IS)

| Live lane | Daily corr vs NYSE_CLOSE | Overlap days | Overlap % of NYSE_CLOSE days |
|---|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | -0.012 | 746 | 93.0% |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | -0.001 | 431 | 53.7% |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | +0.034 | 753 | 93.9% |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | +0.016 | 792 | 98.8% |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | +0.002 | 478 | 59.6% |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.015 | 714 | 89.0% |

- Corr to aggregate live-book daily R: `+0.023`
- NYSE_CLOSE trade days: `802`
- Candidate days overlapping any live lane: `802` (`100.0%`)
- Avg live lanes active on NYSE_CLOSE days: `4.880`

## 2026 OOS Monitor Only

This section is monitor-only and NOT used as selection proof.

| Portfolio | Start | End | BizDays | Trades | TradeDays | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Live 6-lane book | 2026-01-01 | 2026-04-10 | 72 | 386 | 70 | +54.5 | +190.7 | +3.973 | 8.4 | -6.0 | 5.361 |
| Live book + NYSE_CLOSE | 2026-01-01 | 2026-04-10 | 72 | 428 | 70 | +74.8 | +261.7 | +5.241 | 7.7 | -7.0 | 5.944 |
| NYSE_CLOSE standalone | 2026-01-01 | 2026-04-10 | 72 | 42 | 42 | +20.3 | +71.0 | +6.959 | 2.0 | -1.0 | 0.583 |

## Verdict

**Outcome:** `CONTINUE as allocator candidate`

Free slot exists, IS annual R and honest Sharpe improve when NYSE_CLOSE is added, and candidate/book daily correlation is low.

## Interpretation

- `ORB_G8` stays killed. This audit does not rescue a dead filter.
- The remaining question is role. Because the current MNQ profile has a free slot, the honest allocator test is additive first, not replacement-first.
- If additive inclusion helps the live-book math on the common IS window without meaningfully increasing co-movement, `NYSE_CLOSE` deserves continued allocator attention.
- If additive inclusion fails, the branch should fall back to standalone-monitor status or park, not more filter shopping.

## Reproduction

```bash
./.venv-wsl/bin/python research/mnq_nyse_close_rr10_role_audit.py
```

No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.
