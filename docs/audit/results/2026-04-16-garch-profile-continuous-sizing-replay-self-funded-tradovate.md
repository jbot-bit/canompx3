# Garch Profile Continuous Sizing Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-profile-continuous-sizing-replay.yaml`
**Profile:** `self_funded_tradovate` (`self_funded`, `30,000`, copies=1, stop=0.75x, active=False)
**Purpose:** test A2 bounded continuous sizing translated into live-feasible integer contracts on the replayable lane set.
**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-level doctrine surface.

## Lane coverage

- Requested lanes: `10`
- Replayed lanes: `10`
- Skipped lanes: `0`

## Replay results

| Map | Per-acct total $ | 1-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass | Mean desired w | Mean contracts | Changed % | Zero % | Double % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +33,689.6 | +33,689.6 | +2.254 | -3,228.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000% | 0.000% | 0.000% |
| LOW_CUT_ONLY | +33,372.9 | +33,372.9 | +2.282 | -3,660.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.002 | 0.874 | 12.638% | 12.638% | 0.000% |
| HIGH_BOOST_ONLY | +33,630.1 | +33,630.1 | +2.259 | -3,228.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.006 | 1.000 | 0.000% | 0.000% | 0.000% |
| SESSION_CLIPPED | +33,372.9 | +33,372.9 | +2.282 | -3,660.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.008 | 0.874 | 12.638% | 12.638% | 0.000% |
| SESSION_LINEAR | +32,655.1 | +32,655.1 | +2.285 | -3,137.7 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.007 | 0.827 | 17.318% | 17.318% | 0.000% |
| GLOBAL_LINEAR | +29,814.6 | +29,814.6 | +2.175 | -2,219.3 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 | 1.012 | 0.739 | 26.061% | 26.061% | 0.000% |

## Delta vs base

| Map | Δ per-acct $ | Δ 1-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| LOW_CUT_ONLY | -316.7 | -316.7 | +0.028 | -432.0 | +0.0 | -0.0 | +0.000 | +0.000 |
| HIGH_BOOST_ONLY | -59.5 | -59.5 | +0.005 | +0.0 | +0.0 | +0.0 | +0.000 | +0.000 |
| SESSION_CLIPPED | -316.7 | -316.7 | +0.028 | -432.0 | +0.0 | -0.0 | +0.000 | +0.000 |
| SESSION_LINEAR | -1,034.5 | -1,034.5 | +0.031 | +90.3 | +0.0 | +0.0 | +0.000 | +0.000 |
| GLOBAL_LINEAR | -3,875.0 | -3,875.0 | -0.079 | +1,008.7 | +0.0 | +0.0 | +0.000 | +0.000 |

## Reading the replay

- This stage tests bounded continuous desired weights only after translating them into real integer contracts.
- If a map improves the normalized desired-weight surface but collapses after translation, that is a valid negative result for A2.
- Session-level attribution is intentionally omitted here because that layer is not yet authoritative.
