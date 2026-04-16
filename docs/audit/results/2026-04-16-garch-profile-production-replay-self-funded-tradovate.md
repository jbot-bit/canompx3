# Garch Profile Production Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-self-funded-production-replay.yaml`
**Profile:** `self_funded_tradovate` (`self_funded`, `30,000`, copies=1, stop=0.75x, active=False)
**Purpose:** convert regime maps into discrete live actions on a selected profile under canonical account rules.
**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.

## Lane coverage

- Requested lanes: `10`
- Replayed lanes: `10`
- Skipped lanes: `0`

## Native validated session scaffolds

| Map | Session scaffold |
|---|---|
| GARCH_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(H), TOKYO_OPEN(M) |
| OVN_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(HL), NYSE_OPEN(H), TOKYO_OPEN(M) |
| GARCH_OVN_NATIVE_DISCRETE | COMEX_SETTLE(M), EUROPE_FLOW(HLM), TOKYO_OPEN(M) |

## Replay results

| Map | Per-acct total $ | 1-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +33,689.6 | +33,689.6 | +2.254 | -3,228.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 |
| GARCH_NATIVE_DISCRETE | +39,149.1 | +39,149.1 | +2.291 | -3,343.3 | -1,411.5 | -1,677.3 | 3 | 0.831 | 0.831 |
| OVN_NATIVE_DISCRETE | +41,363.4 | +41,363.4 | +2.060 | -3,889.2 | -1,779.5 | -2,067.6 | 3 | 0.757 | 0.757 |
| GARCH_OVN_NATIVE_DISCRETE | +37,353.7 | +37,353.7 | +2.301 | -3,365.4 | -1,368.4 | -1,492.3 | 3 | 0.911 | 0.911 |

## Delta vs base

| Map | Δ per-acct $ | Δ 1-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| GARCH_NATIVE_DISCRETE | +5,459.5 | +5,459.5 | +0.037 | -115.3 | -308.4 | -166.8 | -0.168 | -0.168 |
| OVN_NATIVE_DISCRETE | +7,673.8 | +7,673.8 | -0.193 | -661.2 | -676.5 | -557.1 | -0.243 | -0.243 |
| GARCH_OVN_NATIVE_DISCRETE | +3,664.1 | +3,664.1 | +0.047 | -137.4 | -265.3 | +18.2 | -0.089 | -0.089 |

### Session delta: `GARCH_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +3,812.3 |
| EUROPE_FLOW | +1,907.2 |
| NYSE_OPEN | +1.3 |
| US_DATA_1000 | +1.1 |
| TOKYO_OPEN | +0.1 |
| CME_PRECLOSE | +0.0 |
| CME_REOPEN | -0.1 |
| SINGAPORE_OPEN | -0.1 |

### Session delta: `OVN_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| EUROPE_FLOW | +3,574.7 |
| NYSE_OPEN | +2,229.7 |
| COMEX_SETTLE | +1,496.8 |
| US_DATA_1000 | +1.1 |
| TOKYO_OPEN | +0.1 |
| CME_PRECLOSE | +0.0 |
| CME_REOPEN | -0.1 |
| SINGAPORE_OPEN | -0.1 |

### Session delta: `GARCH_OVN_NATIVE_DISCRETE`

| Session | Δ$ |
|---|---|
| EUROPE_FLOW | +3,484.6 |
| NYSE_OPEN | +1.3 |
| US_DATA_1000 | +1.1 |
| COMEX_SETTLE | +0.1 |
| TOKYO_OPEN | +0.1 |
| CME_PRECLOSE | +0.0 |
| CME_REOPEN | -0.1 |
| SINGAPORE_OPEN | -0.1 |

## Reading the replay

- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.
- Native maps use validated-scope discovered session support, not the broad common-scaffold shortcut.
- Discrete actions are `0` contracts in hostile low state, `1` in neutral, and `2` in favorable high state.
- Survival metrics use canonical account-survival style replay on the resulting daily scenarios.
- This is the first deployment-allocator slice, not the full continuous allocator architecture.
- If skipped lanes are non-zero, the replay is only for the replayable subset and must not be over-read as a full-book result.
