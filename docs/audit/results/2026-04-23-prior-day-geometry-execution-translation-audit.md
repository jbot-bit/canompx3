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
- `ExecutionEngine` now fails closed when that reduction cannot be expressed on a 1-contract row, so same-session derisking is still unavailable under the current lane sizing.
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

| Candidate | Shared Days w/ O15 | Time Overlap Days | Same-Dir Overlap | Opp-Dir Overlap | Candidate Blocked | Incumbent Block Δ | Other Live Block Δ | Half-Size Suggested | Half-Size Unexpressible | Δ Annual R IS | Δ Sharpe IS | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | 190 | 94 | 94 | 0 | 26 | 85 | 0 | 85 | 85 | +6.7 | +0.154 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | 222 | 122 | 122 | 0 | 20 | 110 | 0 | 110 | 110 | +4.6 | +0.108 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | 329 | 170 | 170 | 0 | 36 | 154 | 0 | 154 | 154 | +6.6 | +0.149 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | 329 | 210 | 210 | 0 | 35 | 190 | 0 | 190 | 190 | +9.4 | +0.194 | `ARCHITECTURE_CHANGE_REQUIRED` |

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG

- Candidate filter: `PD_DISPLACE_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `190` shared trade days out of `215` candidate days.
- Time overlap days: `94` (same-direction `94`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `94` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `19.5` minutes.
- Runtime half-size suggestions: `85`; unexpressible on 1-contract lanes and therefore rejected: `85`.
- Candidate blocked by runtime: `26`.
- Incremental incumbent blocks vs current live book: `85`.
- Incremental other-live blocks vs current live book: `0`.
- Max concurrent open positions in translated replay: `3` total, `1` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-20 | 2025-12-31 | +47.6 | +6.9 | +1.345 | 5.2 | -1.0 | 205 | 0.119 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-02 | +1.7 | +6.7 | +1.101 | 2.0 | -1.0 | 10 | 0.152 |
| Translated live+candidate IS | 2019-05-20 | 2025-12-31 | +744.4 | +108.6 | +2.778 | 43.2 | -6.0 | 7820 | 4.525 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-02 | +58.8 | +224.4 | +4.822 | 7.1 | -5.0 | 352 | 5.333 |

- IS delta vs base live book: annualized R `+6.7`, honest Sharpe `+0.154`.
- OOS monitor delta vs base: annualized R `+10.6`, honest Sharpe `+0.224`.
- Runtime block reasons: `hedging_guard=26, unexpressible_contract_reduction=85`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG

- Candidate filter: `PD_CLEAR_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `222` shared trade days out of `245` candidate days.
- Time overlap days: `122` (same-direction `122`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `122` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `19.0` minutes.
- Runtime half-size suggestions: `110`; unexpressible on 1-contract lanes and therefore rejected: `110`.
- Candidate blocked by runtime: `20`.
- Incremental incumbent blocks vs current live book: `110`.
- Incremental other-live blocks vs current live book: `0`.
- Max concurrent open positions in translated replay: `3` total, `1` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +50.0 | +7.3 | +1.334 | 8.0 | -1.0 | 232 | 0.134 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +0.7 | +2.6 | +0.424 | 3.1 | -1.0 | 11 | 0.159 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +735.7 | +107.2 | +2.750 | 43.3 | -5.0 | 7839 | 4.531 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +63.5 | +232.1 | +4.896 | 7.1 | -5.0 | 359 | 5.203 |

- IS delta vs base live book: annualized R `+4.6`, honest Sharpe `+0.108`.
- OOS monitor delta vs base: annualized R `+9.8`, honest Sharpe `+0.053`.
- Runtime block reasons: `hedging_guard=20, unexpressible_contract_reduction=110`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG

- Candidate filter: `PD_GO_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `329` shared trade days out of `365` candidate days.
- Time overlap days: `170` (same-direction `170`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `170` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `20.0` minutes.
- Runtime half-size suggestions: `154`; unexpressible on 1-contract lanes and therefore rejected: `154`.
- Candidate blocked by runtime: `36`.
- Incremental incumbent blocks vs current live book: `154`.
- Incremental other-live blocks vs current live book: `0`.
- Max concurrent open positions in translated replay: `3` total, `1` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +63.4 | +9.2 | +1.375 | 10.2 | -1.0 | 350 | 0.202 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +2.7 | +9.7 | +1.444 | 3.0 | -1.0 | 13 | 0.188 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +749.8 | +109.2 | +2.792 | 43.8 | -6.0 | 7902 | 4.568 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +65.5 | +239.1 | +5.074 | 7.1 | -5.0 | 361 | 5.232 |

- IS delta vs base live book: annualized R `+6.6`, honest Sharpe `+0.149`.
- OOS monitor delta vs base: annualized R `+16.9`, honest Sharpe `+0.231`.
- Runtime block reasons: `hedging_guard=36, unexpressible_contract_reduction=154`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG

- Candidate filter: `PD_GO_LONG`
- Pairwise overlap with live `US_DATA_1000 O15` incumbent: `329` shared trade days out of `362` candidate days.
- Time overlap days: `210` (same-direction `210`, opposite-direction `0`).
- Candidate active when O15 tried to enter: `210` days.
- Median O15 minus O5 entry gap: `+10.0` minutes.
- Median overlapping hold time on overlap days: `34.0` minutes.
- Runtime half-size suggestions: `190`; unexpressible on 1-contract lanes and therefore rejected: `190`.
- Candidate blocked by runtime: `35`.
- Incremental incumbent blocks vs current live book: `190`.
- Incremental other-live blocks vs current live book: `0`.
- Max concurrent open positions in translated replay: `3` total, `1` on `US_DATA_1000`.

| Surface | Start | End | Total R | Annualized R | Sharpe Ann | MaxDD R | Worst Day R | Trades | Avg Trades/Day |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standalone candidate IS | 2019-05-16 | 2025-12-31 | +72.1 | +10.5 | +1.224 | 8.9 | -1.0 | 347 | 0.201 |
| Standalone candidate OOS monitor | 2026-01-01 | 2026-04-07 | +1.7 | +6.1 | +0.715 | 4.1 | -1.0 | 13 | 0.188 |
| Translated live+candidate IS | 2019-05-16 | 2025-12-31 | +768.6 | +112.0 | +2.837 | 41.5 | -6.0 | 7867 | 4.547 |
| Translated live+candidate OOS monitor | 2026-01-01 | 2026-04-07 | +63.6 | +232.3 | +5.112 | 6.6 | -5.0 | 359 | 5.203 |

- IS delta vs base live book: annualized R `+9.4`, honest Sharpe `+0.194`.
- OOS monitor delta vs base: annualized R `+10.0`, honest Sharpe `+0.269`.
- Runtime block reasons: `hedging_guard=35, unexpressible_contract_reduction=190`.
- Verdict: `ARCHITECTURE_CHANGE_REQUIRED`.
- Next step: Do not auto-route under current runtime; model a dedicated shadow translation because live blocking already appears in replay.

## Bottom Line

- `US_DATA_1000 O5` prior-day geometry is not blocked by same-session runtime rules in the same way as `COMEX_SETTLE O5`.
- But the current runtime still cannot express a real same-session size-down for these rows: the overlap path requests `0.5x` and 1-contract lanes cannot honor it.
- The live engine now fails closed instead of silently entering full-size, so any promotion here still needs architecture / policy work rather than a direct route claim.
- `COMEX_SETTLE PD_CLEAR_LONG` remains outside this branch; same-aperture coexistence is still blocked and replacement remains negative.

## Limitations

- This is a runtime replay on canonical `orb_outcomes` timing plus current repo runtime rules; it is not a live shadow test.
- The translated replay measures current control-surface behavior, not a hypothetical improved architecture with fractional same-session sizing.
- Positive common-window IS/OOS deltas do not rescue a path that still requires architecture or policy change to express honest reduced-size coexistence.
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
