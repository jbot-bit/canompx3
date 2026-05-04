# Garch A4 Portfolio-Ranking Allocator Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-a4-portfolio-ranking-allocator.yaml`
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x, max_slots=7, active=True)
**Purpose:** test routing-only scarce-slot allocation using a locked triple-mean vol-state score.
**Status:** operational stress test on the current research-provisional live book; not standalone edge proof and not a session-doctrine surface.

## Lane coverage

- Requested lanes: `6`
- Replayed lanes: `6`
- Skipped lanes: `0`

## Replay results

| Route | Per-acct total $ | 2-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X_SLOT_ORDER | +46,321.2 | +92,642.5 | +2.115 | -3,159.0 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 |
| TRIPLE_MEAN_SLOT_RANK | +46,321.2 | +92,642.5 | +2.115 | -3,159.0 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 |

## Delta vs base

| Candidate | Δ per-acct $ | Δ 2-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| TRIPLE_MEAN_SLOT_RANK | +0.0 | +0.0 | +0.000 | +0.0 | +0.0 | +0.0 | +0.000 | +0.000 |

## Routing diagnostics

| Metric | Value |
|---|---|
| Active days | 1789 |
| Collision days | 0 |
| Rerouted days | 0 |
| Pct days rerouted | 0.000 |
| Collision-day-only delta $ | +0.0 |
| Budget utilization rate | 0.596 |
| Non-collision days identical | True |
| Top abs lane-share of delta | 0.000 |
| Top abs session-share of delta | 0.000 |

## Top lane deltas

| Strategy | Δ$ |
|---|---|

## Top session deltas

| Session | Δ$ |
|---|---|

## Reading the replay

- This is routing-only. Every selected lane stays at 1x. No upsizing, downsizing, or fractional sizing is allowed in this stage.
- The candidate may differ from base only on collision days where eligible lanes exceed the profile slot budget.
- Baseline order is fixed by the profile lane order. Candidate order is the locked triple-mean score with deterministic tie-breaks.
- This stage tests deployment allocator utility, not standalone signal edge and not session doctrine.
