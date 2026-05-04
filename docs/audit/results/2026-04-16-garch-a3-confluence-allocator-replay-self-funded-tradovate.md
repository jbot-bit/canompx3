# Garch A3 Confluence Allocator Replay

**Date:** 2026-04-15
**Pre-registration:** `docs/audit/hypotheses/2026-04-16-garch-a3-confluence-allocator-replay.yaml`
**Profile:** `self_funded_tradovate` (`self_funded`, `30,000`, copies=1, stop=0.75x, active=False)
**Purpose:** compare simple confluence allocator maps built from garch, overnight, and ATR under canonical profile replay rules.
**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-attribution surface.

## Lane coverage

- Requested lanes: `10`
- Replayed lanes: `10`
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

| Map | Per-acct total $ | 1-copy total $ | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |
|---|---|---|---|---|---|---|---|---|---|
| BASE_1X | +33,689.6 | +33,689.6 | +2.254 | -3,228.0 | -1,103.1 | -1,510.5 | 3 | 1.000 | 1.000 |
| GARCH_NATIVE_DISCRETE | +39,149.1 | +39,149.1 | +2.291 | -3,343.3 | -1,411.5 | -1,677.3 | 3 | 0.831 | 0.831 |
| OVN_NATIVE_DISCRETE | +41,363.4 | +41,363.4 | +2.060 | -3,889.2 | -1,779.5 | -2,067.6 | 3 | 0.757 | 0.757 |
| GARCH_OVN_NATIVE_DISCRETE | +37,353.7 | +37,353.7 | +2.301 | -3,365.4 | -1,368.4 | -1,492.3 | 3 | 0.911 | 0.911 |
| GARCH_ATR_NATIVE_DISCRETE | +36,620.0 | +36,620.0 | +2.325 | -3,228.0 | -1,240.5 | -1,660.7 | 3 | 1.000 | 1.000 |
| TRIPLE_MEAN_NATIVE_DISCRETE | +36,493.6 | +36,493.6 | +2.224 | -3,228.0 | -1,368.4 | -1,492.3 | 3 | 0.911 | 0.911 |

## Delta vs base

| Map | Δ per-acct $ | Δ 1-copy total $ | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |
|---|---|---|---|---|---|---|---|---|
| GARCH_NATIVE_DISCRETE | +5,459.5 | +5,459.5 | +0.037 | -115.3 | -308.4 | -166.8 | -0.168 | -0.168 |
| OVN_NATIVE_DISCRETE | +7,673.8 | +7,673.8 | -0.193 | -661.2 | -676.5 | -557.1 | -0.243 | -0.243 |
| GARCH_OVN_NATIVE_DISCRETE | +3,664.1 | +3,664.1 | +0.047 | -137.4 | -265.3 | +18.2 | -0.089 | -0.089 |
| GARCH_ATR_NATIVE_DISCRETE | +2,930.4 | +2,930.4 | +0.072 | +0.0 | -137.5 | -150.2 | +0.000 | +0.000 |
| TRIPLE_MEAN_NATIVE_DISCRETE | +2,804.0 | +2,804.0 | -0.029 | +0.0 | -265.3 | +18.2 | -0.089 | -0.089 |

## Reading the replay

- This stage compares simple auditable confluence maps only. No tree model, no session-attribution claim, no forward tuning.
- Native scaffolds are earned separately by each map on validated populations before profile replay translation.
- Portfolio-ranking allocator work is explicitly deferred; this stage only answers whether simple confluence translation beats solo maps.
