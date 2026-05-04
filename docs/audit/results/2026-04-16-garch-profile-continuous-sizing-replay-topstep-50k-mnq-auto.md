# Garch Profile Continuous Sizing Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-profile-continuous-sizing-replay.yaml`
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x, active=True)
**Purpose:** test A2 bounded continuous sizing translated into live-feasible integer contracts on the replayable lane set.
**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-level doctrine surface.

## Lane coverage

- Requested lanes: `6`
- Replayed lanes: `6`
- Skipped lanes: `0`

## Replay results

| Map | Per-acct total $ | 2-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass | Mean desired w | Mean contracts | Changed % | Zero % | Double % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +46,322.9 | +92,645.7 | +2.115 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 | 1.000 | 1.000 | 0.000% | 0.000% | 0.000% |
| LOW_CUT_ONLY | +43,598.0 | +87,196.0 | +2.114 | -2,598.7 | -1,522.2 | -2,373.8 | 1 | 0.935 | 0.935 | 1.003 | 0.796 | 20.364% | 20.364% | 0.000% |
| HIGH_BOOST_ONLY | +44,624.0 | +89,247.9 | +2.130 | -2,598.7 | -1,522.2 | -2,373.8 | 1 | 0.932 | 0.932 | 1.004 | 1.000 | 0.000% | 0.000% | 0.000% |
| SESSION_CLIPPED | +43,598.0 | +87,196.0 | +2.114 | -2,598.7 | -1,522.2 | -2,373.8 | 1 | 0.935 | 0.935 | 1.008 | 0.796 | 20.364% | 20.364% | 0.000% |
| SESSION_LINEAR | +39,647.5 | +79,295.0 | +1.727 | -4,372.2 | -3,302.1 | -4,019.7 | 1 | 0.909 | 0.909 | 1.006 | 0.807 | 22.244% | 20.749% | 1.495% |
| GLOBAL_LINEAR | +38,846.6 | +77,693.3 | +1.585 | -4,678.1 | -3,044.5 | -4,453.2 | 1 | 0.869 | 0.869 | 1.008 | 0.773 | 32.172% | 27.424% | 4.748% |

## Delta vs base

| Map | Δ per-acct $ | Δ 2-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| LOW_CUT_ONLY | -2,724.8 | -5,449.7 | -0.001 | +560.2 | +0.0 | +0.0 | +0.019 | +0.019 |
| HIGH_BOOST_ONLY | -1,698.9 | -3,397.8 | +0.015 | +560.2 | +0.0 | +0.0 | +0.015 | +0.015 |
| SESSION_CLIPPED | -2,724.8 | -5,449.7 | -0.001 | +560.2 | +0.0 | +0.0 | +0.019 | +0.019 |
| SESSION_LINEAR | -6,675.4 | -13,350.7 | -0.388 | -1,213.3 | -1,779.8 | -1,646.0 | -0.008 | -0.008 |
| GLOBAL_LINEAR | -7,476.2 | -14,952.4 | -0.530 | -1,519.3 | -1,522.2 | -2,079.5 | -0.047 | -0.047 |

## Reading the replay

- This stage tests bounded continuous desired weights only after translating them into real integer contracts.
- If a map improves the normalized desired-weight surface but collapses after translation, that is a valid negative result for A2.
- Session-level attribution is intentionally omitted here because that layer is not yet authoritative.
