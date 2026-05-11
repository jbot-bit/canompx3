# MNQ NYSE_OPEN RR1.5 COST_LT12 SR Alarm Review

**Date:** 2026-05-11
**Profile:** `topstep_50k_mnq_auto`
**Strategy:** `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`
**Decision:** `WATCH`

## Authority

- Criterion 12: `docs/institutional/pre_registered_criteria.md`
- SR method: `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`
- Runtime review registry: `trading_app/sr_review_registry.py`
- SR implementation: `trading_app/sr_monitor.py` and `trading_app/live/sr_monitor.py`
- Prior WATCH precedent: `docs/handoffs/archived/2026-04-23-root-handoff-archive.md`

## Finding

The SR alarm is real, not stale state. Replaying the monitor on canonical
forward outcomes produced:

| Metric | Value |
|---|---:|
| Monitored trades | 75 |
| Baseline | `validated_backtest` |
| Stream | `canonical_forward` |
| Expected R | +0.105 |
| Threshold | 31.96 |
| Alarm trade | 43 |
| Max SR | 552.36 |
| Final SR after recovery | 11.69 |
| Forward mean R | +0.048 |
| Wins / losses | 32 / 43 |

The alarm came from a clustered loss run around late February / early March
2026. This is exactly the path-dependent case where aggregate OOS alone is not
enough.

## Review Decision

Set code-backed review outcome to `watch`, not `pause`.

Reason: the lane clears the same review floors used for existing C12 WATCH
precedents:

| Evidence | Value |
|---|---:|
| WFE | 1.80 |
| OOS/IS | 61% |
| IS mean R | +0.099 |
| OOS mean R | +0.061 |
| Chordia verdict | `PASS_PROTOCOL_A` |

This does not make the lane institutional-proof or remove the SR concern. It
only converts the operational state from "blocked pending manual review" to
"reviewed WATCH" so a provisional allocation can be expressed and monitored.

## Recheck Trigger

Re-check after `N>=100` monitored trades. Retire if SR remains `ALARM` and
either WFE falls below `0.50` or OOS/IS falls below `0.40`, matching the
existing WATCH precedent.

## Reproduction

```bash
./.venv-wsl/bin/python -m trading_app.sr_monitor
./.venv-wsl/bin/python research/mnq_profile_candidate_proposal_2026_05_11.py --bootstrap-runtime-control
```
