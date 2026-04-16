# Garch Profile Production Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-profile-production-replay.yaml`
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x)
**Purpose:** convert regime maps into discrete live actions on the active profile under canonical account rules.
**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.

## Native validated session scaffolds

| Map | Session scaffold |
|---|---|
| GARCH_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(H), TOKYO_OPEN(M) |
| OVN_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(HL), NYSE_OPEN(H), TOKYO_OPEN(M) |
| GARCH_OVN_NATIVE_DISCRETE | COMEX_SETTLE(M), EUROPE_FLOW(HLM), TOKYO_OPEN(M) |

## Replay results

| Map | Per-acct total $ | 2-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +46,322.9 | +92,645.7 | +2.115 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 |
| GARCH_NATIVE_DISCRETE | +55,632.8 | +111,265.5 | +2.335 | -3,158.9 | -1,124.2 | -1,580.4 | 1 | 0.916 | 0.916 |
| OVN_NATIVE_DISCRETE | +54,493.0 | +108,986.0 | +1.978 | -5,423.4 | -1,028.5 | -2,735.9 | 1 | 0.823 | 0.823 |
| GARCH_OVN_NATIVE_DISCRETE | +49,627.0 | +99,253.9 | +2.233 | -3,158.9 | -910.2 | -1,785.5 | 1 | 0.928 | 0.928 |

## Delta vs base

| Map | Δ per-acct $ | Δ 2-copy $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| GARCH_NATIVE_DISCRETE | +9,309.9 | +18,619.8 | +0.220 | +0.0 | +398.0 | +793.4 | +0.000 | +0.000 |
| OVN_NATIVE_DISCRETE | +8,170.2 | +16,340.3 | -0.136 | -2,264.6 | +493.8 | -362.2 | -0.093 | -0.093 |
| GARCH_OVN_NATIVE_DISCRETE | +3,304.1 | +6,608.2 | +0.118 | +0.0 | +612.1 | +588.3 | +0.012 | +0.012 |

### Session delta: `GARCH_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +5,928.7 |
| EUROPE_FLOW | +3,695.3 |
| NYSE_OPEN | +2.0 |
| US_DATA_1000 | +1.1 |
| SINGAPORE_OPEN | +0.6 |
| TOKYO_OPEN | -1.0 |

### Session delta: `OVN_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| EUROPE_FLOW | +3,861.8 |
| COMEX_SETTLE | +3,820.5 |
| NYSE_OPEN | +878.0 |
| US_DATA_1000 | +1.1 |
| SINGAPORE_OPEN | +0.6 |
| TOKYO_OPEN | -1.0 |

### Session delta: `GARCH_OVN_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| EUROPE_FLOW | +4,002.5 |
| NYSE_OPEN | +2.0 |
| US_DATA_1000 | +1.1 |
| SINGAPORE_OPEN | +0.6 |
| COMEX_SETTLE | +0.2 |
| TOKYO_OPEN | -1.0 |

## Reading the replay

- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.
- Native maps use validated-scope discovered session support, not the broad common-scaffold shortcut.
- Discrete actions are `0` contracts in hostile low state, `1` in neutral, `2` in favorable high state.
- Survival metrics use canonical Topstep Criterion 11 path replay on the resulting daily scenarios.
