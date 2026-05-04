# Garch A3 Confluence Allocator Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-a3-confluence-allocator-replay.yaml`
**Profile:** `topstep_50k_mnq_auto` (`topstep`, `50,000`, copies=2, stop=0.75x, active=True)
**Purpose:** compare simple confluence allocator maps built from garch, overnight, and ATR under canonical profile replay rules.
**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-attribution surface.

## Lane coverage

- Requested lanes: `6`
- Replayed lanes: `6`
- Skipped lanes: `0`

## Native validated scaffolds

| Map | Session scaffold |
|---|---|
| GARCH_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(H), TOKYO_OPEN(M) |
| OVN_NATIVE_DISCRETE | COMEX_SETTLE(HLM), EUROPE_FLOW(HL), NYSE_OPEN(H), TOKYO_OPEN(M) |
| GARCH_OVN_NATIVE_DISCRETE | COMEX_SETTLE(M), EUROPE_FLOW(HLM), TOKYO_OPEN(M) |
| GARCH_ATR_NATIVE_DISCRETE | COMEX_SETTLE(HM), TOKYO_OPEN(M) |
| TRIPLE_MEAN_NATIVE_DISCRETE | COMEX_SETTLE(M), EUROPE_FLOW(H), TOKYO_OPEN(M) |

## Replay results

| Map | Per-acct total $ | 2-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +46,322.9 | +92,645.7 | +2.115 | -3,158.9 | -1,522.2 | -2,373.8 | 1 | 0.916 | 0.916 |
| GARCH_NATIVE_DISCRETE | +55,632.8 | +111,265.5 | +2.335 | -3,158.9 | -1,124.2 | -1,580.4 | 1 | 0.916 | 0.916 |
| OVN_NATIVE_DISCRETE | +54,493.0 | +108,986.0 | +1.978 | -5,423.4 | -1,028.5 | -2,735.9 | 1 | 0.823 | 0.823 |
| GARCH_OVN_NATIVE_DISCRETE | +49,627.0 | +99,253.9 | +2.233 | -3,158.9 | -910.2 | -1,785.5 | 1 | 0.928 | 0.928 |
| GARCH_ATR_NATIVE_DISCRETE | +51,526.4 | +103,052.8 | +2.203 | -3,158.9 | -1,736.3 | -2,137.6 | 1 | 0.905 | 0.905 |
| TRIPLE_MEAN_NATIVE_DISCRETE | +50,131.3 | +100,262.6 | +2.243 | -3,158.9 | -910.2 | -1,785.5 | 1 | 0.929 | 0.929 |

## Delta vs base

| Map | Δ per-acct $ | Δ 2-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| GARCH_NATIVE_DISCRETE | +9,309.9 | +18,619.8 | +0.220 | +0.0 | +398.0 | +793.4 | +0.000 | +0.000 |
| OVN_NATIVE_DISCRETE | +8,170.2 | +16,340.3 | -0.136 | -2,264.6 | +493.8 | -362.2 | -0.093 | -0.093 |
| GARCH_OVN_NATIVE_DISCRETE | +3,304.1 | +6,608.2 | +0.118 | +0.0 | +612.1 | +588.3 | +0.012 | +0.012 |
| GARCH_ATR_NATIVE_DISCRETE | +5,203.5 | +10,407.1 | +0.089 | +0.0 | -214.1 | +236.1 | -0.011 | -0.011 |
| TRIPLE_MEAN_NATIVE_DISCRETE | +3,808.4 | +7,616.9 | +0.128 | +0.0 | +612.1 | +588.3 | +0.012 | +0.012 |

## Reading the replay

- This stage compares simple auditable confluence maps only. No tree model, no session-attribution claim, no forward tuning.
- Native scaffolds are earned separately by each map on validated populations before profile replay translation.
- Portfolio-ranking allocator work is explicitly deferred; this stage only answers whether simple confluence translation beats solo maps.
