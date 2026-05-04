# Garch Profile Policy Surface Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-profile-policy-surface-replay.yaml`
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x, active=True)
**Purpose:** replay the expanded raw discrete garch policy surface on a selected profile under canonical account rules.
**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.

## Lane coverage

- Requested lanes: `6`
- Replayed lanes: `6`
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

| Policy | Per-acct total $ | 2-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +46,322.9 | +92,645.7 | +2.115 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 |
| SESSION_TAKE_HIGH_ONLY | +30,637.7 | +61,275.3 | +1.638 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.933 | 0.933 |
| GLOBAL_TAKE_HIGH_ONLY | +21,414.9 | +42,829.8 | +1.301 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.951 | 0.951 |
| SESSION_SKIP_LOW_ONLY | +45,296.9 | +90,593.8 | +2.098 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.922 | 0.922 |
| GLOBAL_SKIP_LOW_ONLY | +34,416.8 | +68,833.6 | +1.732 | -3,158.9 | -1,522.2 | -2,373.7 | 1 | 0.925 | 0.925 |
| SESSION_HIGH_2X_ONLY | +64,139.0 | +128,278.1 | +2.033 | -4,589.6 | -3,302.1 | -4,412.1 | 1 | 0.787 | 0.787 |
| GLOBAL_HIGH_2X_ONLY | +66,035.9 | +132,071.9 | +1.935 | -5,134.6 | -3,044.5 | -4,747.6 | 1 | 0.723 | 0.723 |
| SESSION_CLIPPED_0_1_2 | +63,113.1 | +126,226.2 | +2.014 | -4,589.6 | -3,302.1 | -4,412.1 | 1 | 0.787 | 0.787 |
| GLOBAL_CLIPPED_0_1_2 | +54,129.9 | +108,259.8 | +1.646 | -5,134.6 | -3,044.5 | -4,747.6 | 1 | 0.706 | 0.706 |

## Delta vs base

| Policy | Δ per-acct $ | Δ 2-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| SESSION_TAKE_HIGH_ONLY | -15,685.2 | -31,370.4 | -0.477 | +0.0 | +0.0 | +0.0 | +0.016 | +0.016 |
| GLOBAL_TAKE_HIGH_ONLY | -24,907.9 | -49,815.9 | -0.814 | +0.0 | +0.0 | +0.0 | +0.035 | +0.035 |
| SESSION_SKIP_LOW_ONLY | -1,026.0 | -2,051.9 | -0.017 | +0.0 | +0.0 | +0.0 | +0.005 | +0.005 |
| GLOBAL_SKIP_LOW_ONLY | -11,906.0 | -23,812.1 | -0.383 | +0.0 | +0.0 | +0.0 | +0.009 | +0.009 |
| SESSION_HIGH_2X_ONLY | +17,816.2 | +35,632.4 | -0.082 | -1,430.7 | -1,779.8 | -2,038.3 | -0.130 | -0.130 |
| GLOBAL_HIGH_2X_ONLY | +19,713.1 | +39,426.2 | -0.180 | -1,975.8 | -1,522.2 | -2,373.8 | -0.193 | -0.193 |
| SESSION_CLIPPED_0_1_2 | +16,790.2 | +33,580.5 | -0.101 | -1,430.7 | -1,779.8 | -2,038.3 | -0.130 | -0.130 |
| GLOBAL_CLIPPED_0_1_2 | +7,807.1 | +15,614.1 | -0.469 | -1,975.8 | -1,522.2 | -2,373.8 | -0.210 | -0.210 |

### Session delta: `SESSION_TAKE_HIGH_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +455.7 |
| NYSE_OPEN | +0.0 |
| TOKYO_OPEN | -1,096.2 |
| EUROPE_FLOW | -1,920.2 |
| SINGAPORE_OPEN | -3,423.0 |
| US_DATA_1000 | -9,699.9 |

### Session delta: `GLOBAL_TAKE_HIGH_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +455.7 |
| TOKYO_OPEN | -1,096.1 |
| EUROPE_FLOW | -1,920.1 |
| SINGAPORE_OPEN | -3,423.0 |
| NYSE_OPEN | -9,222.9 |
| US_DATA_1000 | -9,700.1 |

### Session delta: `SESSION_SKIP_LOW_ONLY`

| Session | Δ$ |
|---|---|
| TOKYO_OPEN | +432.3 |
| NYSE_OPEN | +0.0 |
| US_DATA_1000 | +0.0 |
| SINGAPORE_OPEN | -89.3 |
| COMEX_SETTLE | -314.6 |
| EUROPE_FLOW | -1,052.8 |

### Session delta: `GLOBAL_SKIP_LOW_ONLY`

| Session | Δ$ |
|---|---|
| TOKYO_OPEN | +432.4 |
| SINGAPORE_OPEN | -89.3 |
| COMEX_SETTLE | -314.7 |
| EUROPE_FLOW | -1,052.8 |
| NYSE_OPEN | -5,104.1 |
| US_DATA_1000 | -5,776.1 |

### Session delta: `SESSION_HIGH_2X_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +5,928.8 |
| TOKYO_OPEN | +3,702.2 |
| EUROPE_FLOW | +3,696.5 |
| SINGAPORE_OPEN | +3,460.7 |
| US_DATA_1000 | +1,029.5 |
| NYSE_OPEN | +0.0 |

### Session delta: `GLOBAL_HIGH_2X_ONLY`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +5,928.9 |
| TOKYO_OPEN | +3,702.2 |
| EUROPE_FLOW | +3,696.6 |
| SINGAPORE_OPEN | +3,460.6 |
| NYSE_OPEN | +1,897.2 |
| US_DATA_1000 | +1,029.3 |

### Session delta: `SESSION_CLIPPED_0_1_2`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +5,614.2 |
| TOKYO_OPEN | +4,134.5 |
| SINGAPORE_OPEN | +3,371.4 |
| EUROPE_FLOW | +2,642.1 |
| US_DATA_1000 | +1,029.5 |
| NYSE_OPEN | -0.0 |

### Session delta: `GLOBAL_CLIPPED_0_1_2`

| Session | Δ$ |
|---|---|
| COMEX_SETTLE | +5,614.2 |
| TOKYO_OPEN | +4,134.6 |
| SINGAPORE_OPEN | +3,371.3 |
| EUROPE_FLOW | +2,642.3 |
| NYSE_OPEN | -3,206.8 |
| US_DATA_1000 | -4,746.8 |

## Reading the replay

- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.
- Policies replay the raw verified discrete action counts (`0`, `1`, `2`) rather than fractional sizing.
- Session-aware policies only act in sessions with raw directional support from the regime-family audit.
- This remains an operational stress test; profile/account geometry can still reorder raw row-level policy winners.
- If skipped lanes are non-zero, the replay is only for the replayable subset and must not be over-read as a full-book result.
