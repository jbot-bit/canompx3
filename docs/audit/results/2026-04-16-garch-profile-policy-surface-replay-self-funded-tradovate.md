# Garch Profile Policy Surface Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-self-funded-policy-surface-replay.yaml`
**Profile:** `self_funded_tradovate` (`self_funded`, `30,000`, copies=1, stop=0.75x, active=False)
**Purpose:** replay the expanded raw discrete garch policy surface on a selected profile under canonical account rules.
**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.

## Lane coverage

- Requested lanes: `10`
- Replayed lanes: `10`
- Skipped lanes: `0`

## Session directional support

| Session | High directional support | Low directional support |
|---|---|---|
| BRISBANE_1025 | Y | . |
| CME_PRECLOSE | Y | Y |
| CME_REOPEN | . | . |
| COMEX_SETTLE | Y | Y |
| EUROPE_FLOW | Y | Y |
| LONDON_METALS | Y | Y |
| NYSE_CLOSE | . | . |
| NYSE_OPEN | . | . |
| SINGAPORE_OPEN | Y | Y |
| TOKYO_OPEN | Y | Y |
| US_DATA_1000 | Y | . |
| US_DATA_830 | . | . |

## Replay results

| Policy | Per-acct total $ | 1-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +33,689.6 | +33,689.6 | +2.254 | -3,228.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 |
| SESSION_TAKE_HIGH_ONLY | +26,442.7 | +26,442.7 | +2.034 | -3,261.5 | -946.0 | -1,510.5 | 3 | 1.000 | 1.000 |
| GLOBAL_TAKE_HIGH_ONLY | +17,571.5 | +17,571.5 | +1.671 | -2,536.9 | -760.4 | -1,510.5 | 3 | 1.000 | 1.000 |
| SESSION_SKIP_LOW_ONLY | +33,432.4 | +33,432.4 | +2.277 | -3,660.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 |
| GLOBAL_SKIP_LOW_ONLY | +30,131.5 | +30,131.5 | +2.218 | -2,219.3 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 |
| SESSION_HIGH_2X_ONLY | +43,942.2 | +43,942.2 | +2.230 | -4,041.2 | -1,603.7 | -2,086.3 | 3 | 0.830 | 0.830 |
| GLOBAL_HIGH_2X_ONLY | +51,200.1 | +51,200.1 | +2.153 | -4,135.3 | -1,810.7 | -3,021.1 | 3 | 0.515 | 0.515 |
| SESSION_CLIPPED_0_1_2 | +43,685.0 | +43,685.0 | +2.240 | -4,041.2 | -1,603.7 | -2,086.3 | 3 | 0.830 | 0.830 |
| GLOBAL_CLIPPED_0_1_2 | +47,642.0 | +47,642.0 | +2.073 | -4,135.3 | -1,810.7 | -3,021.1 | 3 | 0.516 | 0.516 |

## Delta vs base

| Policy | Δ per-acct $ | Δ 1-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| SESSION_TAKE_HIGH_ONLY | -7,246.9 | -7,246.9 | -0.219 | -33.5 | +157.1 | +0.0 | +0.000 | +0.000 |
| GLOBAL_TAKE_HIGH_ONLY | -16,118.0 | -16,118.0 | -0.583 | +691.1 | +342.7 | +0.0 | +0.000 | +0.000 |
| SESSION_SKIP_LOW_ONLY | -257.2 | -257.2 | +0.023 | -432.0 | +0.0 | -0.0 | +0.000 | +0.000 |
| GLOBAL_SKIP_LOW_ONLY | -3,558.1 | -3,558.1 | -0.036 | +1,008.7 | +0.0 | +0.0 | +0.000 | +0.000 |
| SESSION_HIGH_2X_ONLY | +10,252.6 | +10,252.6 | -0.024 | -813.2 | -500.7 | -575.8 | -0.170 | -0.170 |
| GLOBAL_HIGH_2X_ONLY | +17,510.5 | +17,510.5 | -0.101 | -907.2 | -707.6 | -1,510.6 | -0.484 | -0.484 |
| SESSION_CLIPPED_0_1_2 | +9,995.4 | +9,995.4 | -0.014 | -813.2 | -500.7 | -575.8 | -0.170 | -0.170 |
| GLOBAL_CLIPPED_0_1_2 | +13,952.4 | +13,952.4 | -0.181 | -907.2 | -707.6 | -1,510.6 | -0.484 | -0.484 |

### Session delta: `SESSION_TAKE_HIGH_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +342.5 |
| NYSE_OPEN | +1.0 |
| CME_REOPEN | +0.1 |
| CME_PRECLOSE | -178.3 |
| TOKYO_OPEN | -583.2 |
| SINGAPORE_OPEN | -955.7 |
| EUROPE_FLOW | -2,239.2 |
| US_DATA_1000 | -3,631.8 |

### Session delta: `GLOBAL_TAKE_HIGH_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +342.5 |
| CME_PRECLOSE | -178.3 |
| TOKYO_OPEN | -583.2 |
| SINGAPORE_OPEN | -955.7 |
| CME_REOPEN | -2,090.6 |
| EUROPE_FLOW | -2,239.3 |
| US_DATA_1000 | -3,632.0 |
| NYSE_OPEN | -6,779.1 |

### Session delta: `SESSION_SKIP_LOW_ONLY`

| Session | Δ$ |
|---|---|
| CME_PRECLOSE | +503.2 |
| TOKYO_OPEN | +455.7 |
| SINGAPORE_OPEN | +83.7 |
| NYSE_OPEN | +1.1 |
| CME_REOPEN | +0.2 |
| US_DATA_1000 | +0.1 |
| COMEX_SETTLE | -259.8 |
| EUROPE_FLOW | -1,039.1 |

### Session delta: `GLOBAL_SKIP_LOW_ONLY`

| Session | Δ$ |
|---|---|
| CME_PRECLOSE | +503.2 |
| TOKYO_OPEN | +455.7 |
| SINGAPORE_OPEN | +83.7 |
| CME_REOPEN | -88.8 |
| COMEX_SETTLE | -259.8 |
| US_DATA_1000 | -372.1 |
| EUROPE_FLOW | -1,039.2 |
| NYSE_OPEN | -2,838.6 |

### Session delta: `SESSION_HIGH_2X_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +3,812.3 |
| CME_PRECLOSE | +2,654.8 |
| EUROPE_FLOW | +1,907.4 |
| TOKYO_OPEN | +1,641.1 |
| SINGAPORE_OPEN | +1,394.6 |
| NYSE_OPEN | +1.0 |
| CME_REOPEN | +0.1 |
| US_DATA_1000 | -1,156.5 |

### Session delta: `GLOBAL_HIGH_2X_ONLY`

| Session | Δ$ |
|---|---|
| CME_REOPEN | +4,372.4 |
| COMEX_SETTLE | +3,812.3 |
| NYSE_OPEN | +2,886.5 |
| CME_PRECLOSE | +2,654.8 |
| EUROPE_FLOW | +1,907.4 |
| TOKYO_OPEN | +1,641.1 |
| SINGAPORE_OPEN | +1,394.7 |
| US_DATA_1000 | -1,156.5 |

### Session delta: `SESSION_CLIPPED_0_1_2`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +3,552.5 |
| CME_PRECLOSE | +3,158.1 |
| TOKYO_OPEN | +2,096.9 |
| SINGAPORE_OPEN | +1,477.8 |
| EUROPE_FLOW | +868.0 |
| NYSE_OPEN | +0.9 |
| CME_REOPEN | +0.1 |
| US_DATA_1000 | -1,156.5 |

### Session delta: `GLOBAL_CLIPPED_0_1_2`

| Session | Δ$ |
|---|---|
| CME_REOPEN | +4,283.4 |
| COMEX_SETTLE | +3,552.5 |
| CME_PRECLOSE | +3,158.0 |
| TOKYO_OPEN | +2,096.9 |
| SINGAPORE_OPEN | +1,477.8 |
| EUROPE_FLOW | +867.9 |
| NYSE_OPEN | +46.8 |
| US_DATA_1000 | -1,528.6 |

## Reading the replay

- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.
- Policies replay the raw verified discrete action counts (`0`, `1`, `2`) rather than fractional sizing.
- Session-aware policies only act in sessions with raw directional support from the regime-family audit.
- This remains an operational stress test; profile/account geometry can still reorder raw row-level policy winners.
- If skipped lanes are non-zero, the replay is only for the replayable subset and must not be over-read as a full-book result.
