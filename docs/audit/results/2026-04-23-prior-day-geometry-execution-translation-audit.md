# Prior-Day Geometry Execution Translation Audit

Date: 2026-04-23

## Scope

Resolve the only open Path 1 question left on the prior-day geometry branch:

> can the positive `MNQ US_DATA_1000 O5` prior-day geometry rows coexist honestly with the live `MNQ US_DATA_1000 O15` lane under the current runtime?

This is a runtime replay, not another discovery pass and not another additivity memo.

## Truth Surfaces

Canonical trade truth:

- `orb_outcomes`
- `daily_features`

Runtime / deployment context:

- `validated_setups` for exact live/candidate row parameters
- `trading_app/prop_profiles.py`
- `trading_app/risk_manager.py`
- `trading_app/execution_engine.py`

## MEASURED Runtime Facts Used

- Current live profile rows resolve with `max_contracts=1` from `build_portfolio_from_profile(...)` in `trading_app/portfolio.py`.
- Same-session different-aperture overlap triggers `suggested_contract_factor=0.5` in `RiskManager.can_enter(...)`.
- `ExecutionEngine` applies that factor with `max(1, int(...))`, so for 1-contract rows the half-size translation is a no-op.
- The current live book has no same-session duplicate session rows; this branch is a special runtime path, not the normal portfolio-builder path.

## Live Book Replayed Under Current Runtime Rules

Live lanes included in the replay:

- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
- `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base live book IS | 2019-05-13 | 2025-12-31 | +704.9 | +102.5 | +2.641 | 41.7 | -5.0 | 7740 | 4.466 |
| Base live book OOS monitor | 2026-01-01 | 2026-04-07 | +60.9 | +222.3 | +4.843 | 7.8 | -5.0 | 352 | 5.101 |

## Candidate Outcome Summary

| Candidate | Shared Days w/ O15 | Time Overlap Days | Same-Dir Overlap | Opp-Dir Overlap | Candidate Blocked | Incumbent Block Δ | Other Live Block Δ | Half-Size Suggested | Half-Size No-Op | Δ Annual R IS | Δ Sharpe IS | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | 190 | 94 | 94 | 0 | 26 | 0 | 2 | 85 | 85 | +6.6 | +0.126 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | 222 | 122 | 122 | 0 | 20 | 0 | 2 | 110 | 110 | +6.4 | +0.111 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | 329 | 170 | 170 | 0 | 36 | 0 | 3 | 154 | 154 | +8.3 | +0.135 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | 329 | 210 | 210 | 0 | 35 | 0 | 4 | 190 | 190 | +10.2 | +0.139 | `ARCHITECTURE_CHANGE_REQUIRED` |

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG

- Candidate filter: `PD_DISPLACE_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `190` shared trade days out of `215` candidate days.
- Time overlap days: `94` (same-direction `94`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `94` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `19.5` minutes.
- Runtime half-size suggestions: `85`; effective no-op due to 1-contract floor: `85`.
- Candidate blocked by runtime: `26`.
- Incremental incumbent blocks vs current live book: `0`.
- Incremental other-live blocks vs current live book: `2`.
- Max concurrent open positions in translated replay: `3` total, `2` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-20 | 2025-12-31 | +47.6 | +6.9 | +1.345 | 5.2 | -1.0 | 205 | 0.119 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-02 | +1.7 | +6.7 | +1.101 | 2.0 | -1.0 | 10 | 0.152 |
| Translated live+candidate IS | 2019-05-20 | 2025-12-31 | +744.0 | +108.5 | +2.750 | 40.0 | -6.0 | 7892 | 4.567 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-02 | +56.8 | +216.7 | +4.670 | 7.1 | -5.0 | 354 | 5.364 |

- IS delta vs base live book: annualized R `+6.6`, honest Sharpe `+0.126`.
- OOS monitor delta vs base: annualized R `+3.0`, honest Sharpe `+0.072`.
- Runtime block reasons: `circuit_breaker=2, hedging_guard=26`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG

- Candidate filter: `PD_CLEAR_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `222` shared trade days out of `245` candidate days.
- Time overlap days: `122` (same-direction `122`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `122` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `19.0` minutes.
- Runtime half-size suggestions: `110`; effective no-op due to 1-contract floor: `110`.
- Candidate blocked by runtime: `20`.
- Incremental incumbent blocks vs current live book: `0`.
- Incremental other-live blocks vs current live book: `2`.
- Max concurrent open positions in translated replay: `3` total, `2` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +50.0 | +7.3 | +1.334 | 8.0 | -1.0 | 232 | 0.134 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +0.7 | +2.6 | +0.424 | 3.1 | -1.0 | 11 | 0.159 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +748.1 | +109.0 | +2.754 | 42.2 | -5.0 | 7936 | 4.587 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +60.6 | +221.4 | +4.749 | 7.1 | -5.0 | 362 | 5.246 |

- IS delta vs base live book: annualized R `+6.4`, honest Sharpe `+0.111`.
- OOS monitor delta vs base: annualized R `-0.9`, honest Sharpe `-0.094`.
- Runtime block reasons: `circuit_breaker=1, hedging_guard=20, max_concurrent=1`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG

- Candidate filter: `PD_GO_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `329` shared trade days out of `365` candidate days.
- Time overlap days: `170` (same-direction `170`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `170` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `20.0` minutes.
- Runtime half-size suggestions: `154`; effective no-op due to 1-contract floor: `154`.
- Candidate blocked by runtime: `36`.
- Incremental incumbent blocks vs current live book: `0`.
- Incremental other-live blocks vs current live book: `3`.
- Max concurrent open positions in translated replay: `3` total, `2` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +63.4 | +9.2 | +1.375 | 10.2 | -1.0 | 350 | 0.202 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +2.7 | +9.7 | +1.444 | 3.0 | -1.0 | 13 | 0.188 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +761.2 | +110.9 | +2.778 | 41.3 | -6.0 | 8037 | 4.646 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +62.6 | +228.5 | +4.929 | 7.1 | -5.0 | 364 | 5.275 |

- IS delta vs base live book: annualized R `+8.3`, honest Sharpe `+0.135`.
- OOS monitor delta vs base: annualized R `+6.2`, honest Sharpe `+0.086`.
- Runtime block reasons: `circuit_breaker=2, hedging_guard=36, max_concurrent=1`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG

- Candidate filter: `PD_GO_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `329` shared trade days out of `362` candidate days.
- Time overlap days: `210` (same-direction `210`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `210` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `34.0` minutes.
- Runtime half-size suggestions: `190`; effective no-op due to 1-contract floor: `190`.
- Candidate blocked by runtime: `35`.
- Incremental incumbent blocks vs current live book: `0`.
- Incremental other-live blocks vs current live book: `4`.
- Max concurrent open positions in translated replay: `3` total, `2` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +72.1 | +10.5 | +1.224 | 8.9 | -1.0 | 347 | 0.201 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +1.7 | +6.1 | +0.715 | 4.1 | -1.0 | 13 | 0.188 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +774.6 | +112.8 | +2.782 | 39.0 | -6.0 | 8034 | 4.644 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +63.5 | +232.0 | +5.030 | 6.6 | -5.0 | 364 | 5.275 |

- IS delta vs base live book: annualized R `+10.2`, honest Sharpe `+0.139`.
- OOS monitor delta vs base: annualized R `+9.7`, honest Sharpe `+0.187`.
- Runtime block reasons: `circuit_breaker=3, hedging_guard=35, max_concurrent=1`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## Bottom Line

- `US_DATA_1000 O5` prior-day geometry is not blocked by same-session runtime rules in the same way as `COMEX_SETTLE O5`.
- But the current runtime does **not** implement a real same-session size-down for these rows: the half-size suggestion collapses to `1` contract because every live/candidate row is clamped to `max_contracts=1`.
- So any promotion here would be an explicit policy decision to allow full-size same-session duplicate exposure, not a quiet reuse of an already-safe translation path.
- `COMEX_SETTLE PD_CLEAR_LONG` remains outside this branch; same-aperture coexistence is still blocked and replacement remains negative.

## Limitations

- This is a runtime replay on canonical `orb_outcomes` timing plus current repo runtime rules; it is not a live shadow test.
- The translated replay measures current control-surface behavior, not a hypothetical improved architecture with fractional same-session sizing.
- Positive common-window IS/OOS deltas do not rescue a path that still requires architecture change to express honest reduced-size coexistence.
- The audit is intentionally narrow to `US_DATA_1000` cross-aperture coexistence. It does not reopen broader prior-day geometry discovery or same-aperture `COMEX_SETTLE` replacement questions.

## Reproduction

Runner:

- `research/prior_day_geometry_execution_translation_audit.py`

Command:

```bash
./.venv-wsl/bin/python research/prior_day_geometry_execution_translation_audit.py --output docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md
```

Outputs:

- result doc: `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`
- stage lock: `docs/runtime/stages/prior-day-geometry-execution-translation-audit.md`
- upstream narrowing note: `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-preaudit.md`
