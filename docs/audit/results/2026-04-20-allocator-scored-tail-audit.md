# Allocator scored-tail audit — 2026-04-18 rebalance

**Generated:** replay of `rebalance_lanes.py --date 2026-04-18 --profile topstep_50k_mnq_auto`
**Script:** `research/allocator_scored_tail_audit.py`
**Canonical replay check:** MISMATCH: missing from replay={'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15', 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12', 'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5'}, extra in replay={'MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5', 'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100', 'MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5', 'MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30'}

## Replay-vs-persisted drift (secondary finding)

The replay above was run at current `data/state/sr_state.json` state. If the replay set differs from `docs/runtime/lane_allocation.json`, that difference is attributable to SR liveness state evolving between the rebalance commit (2026-04-18) and now. `_effective_annual_r` applies multiplicative discounts (ALARM×0.50, 3mo-decay×0.75) that change rankings. The replay reflects the allocator's CURRENT preference at the same rebalance_date — i.e., what it would select if you re-ran `rebalance_lanes.py --date 2026-04-18` right now.

## Audited claim

Mandate #3 from `memory/next_session_mandates_2026_04_20.md`: the 2026-04-18 allocator scored 38 lanes and selected 6 DEPLOY + 2 PAUSED. Inspect the 30-lane tail and the rank-6 / rank-7 boundary to determine whether the DEPLOY cutoff is well-calibrated or whether the first excluded contender was rejected unjustly.

## Pre-committed decision rule

Locked before this run in `docs/runtime/stages/allocator_scored_tail_audit.md`. Structural gates are the canonical allocator rules (Carver Ch.11/12 via `_effective_annual_r`, correlation gate `CORRELATION_REJECT_RHO=0.70`, DD budget from `ACCOUNT_TIERS`, max-slot budget from profile).

- **WELL_CALIBRATED:** r7 ≤ r6 AND exclusion reason is structural (correlation / DD / slot budget).
- **TIGHT_BUT_JUSTIFIED:** r7 > r6 OR r7 ≥ 0.95·r6, AND exclusion is structural.
- **MISCALIBRATED:** exclusion is non-structural (e.g., tie-breaker noise, undocumented gate).

## Profile constraints

- `profile_id`: `topstep_50k_mnq_auto`
- `max_slots`: 7
- `max_dd`: $2,000 (from `ACCOUNT_TIERS`)
- `stop_multiplier`: 0.75
- `allowed_instruments`: ['MNQ']
- `allowed_sessions`: ['CME_PRECLOSE', 'COMEX_SETTLE', 'EUROPE_FLOW', 'NYSE_OPEN', 'SINGAPORE_OPEN', 'TOKYO_OPEN', 'US_DATA_1000']

## Full rank-ordered scored set (all 38 lanes)

Rank = descending `_effective_annual_r` across ALL scored lanes (status-agnostic meta-audit view). `selected?` matches the canonical `build_allocation()` replay.

| # | strategy_id | session | RR | filter | status | eff_annual_r | annual_r | trailing_expr | N | session_regime | SR | selected? | exclusion_reason |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---|:---:|---|
| 1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | EUROPE_FLOW | 1.5 | ORB_G5 | DEPLOY | 46.10 | 46.1 | +0.1892 | 264 | +0.1534 | CONTINUE | ✓ | SELECTED |
| 2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | SINGAPORE_OPEN | 1.5 | ATR_P50 | DEPLOY | 44.00 | 44.0 | +0.2407 | 137 | +0.1609 | UNKNOWN | ✓ | SELECTED |
| 3 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | EUROPE_FLOW | 1.5 | COST_LT12 | DEPLOY | 40.60 | 40.6 | +0.1819 | 242 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=1.000 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 4 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | COMEX_SETTLE | 1.5 | OVNRNG_100 | DEPLOY | 40.10 | 40.1 | +0.2603 | 167 | +0.0418 | UNKNOWN | ✓ | SELECTED |
| 5 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | EUROPE_FLOW | 1.5 | CROSS_SGP_MOMENTUM | DEPLOY | 39.20 | 39.2 | +0.2603 | 163 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=1.000 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 6 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | EUROPE_FLOW | 2.0 | ORB_G5 | DEPLOY | 38.10 | 38.1 | +0.1563 | 264 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.861 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 7 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | EUROPE_FLOW | 1.5 | OVNRNG_100 | DEPLOY | 37.40 | 37.4 | +0.2305 | 176 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=1.000 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 8 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | EUROPE_FLOW | 1.0 | ORB_G5 | DEPLOY | 34.50 | 34.5 | +0.1416 | 264 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.799 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 9 | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | COMEX_SETTLE | 2.0 | ORB_G5 | DEPLOY | 34.20 | 34.2 | +0.1508 | 246 | +0.0418 | UNKNOWN |  | correlation_gate (ρ=0.830 vs MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100) |
| 10 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | EUROPE_FLOW | 1.0 | COST_LT12 | DEPLOY | 32.60 | 32.6 | +0.1459 | 242 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.803 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 11 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | COMEX_SETTLE | 1.0 | OVNRNG_100 | DEPLOY | 30.80 | 30.8 | +0.1975 | 169 | +0.0418 | UNKNOWN |  | correlation_gate (ρ=0.805 vs MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100) |
| 12 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | NYSE_OPEN | 1.0 | ORB_G5 | DEPLOY | 29.00 | 29.0 | +0.1200 | 262 | +0.1071 | UNKNOWN | ✓ | SELECTED |
| 13 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | COMEX_SETTLE | 1.0 | ORB_G5 | DEPLOY | 27.40 | 27.4 | +0.1165 | 255 | +0.0418 | UNKNOWN |  | correlation_gate (ρ=0.804 vs MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100) |
| 14 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | EUROPE_FLOW | 1.0 | OVNRNG_100 | DEPLOY | 26.80 | 26.8 | +0.1648 | 176 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.827 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 15 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | EUROPE_FLOW | 2.0 | CROSS_SGP_MOMENTUM | DEPLOY | 26.60 | 26.6 | +0.1765 | 163 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.860 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 16 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | COMEX_SETTLE | 1.0 | COST_LT12 | DEPLOY | 24.90 | 24.9 | +0.1140 | 237 | +0.0418 | UNKNOWN |  | correlation_gate (ρ=0.804 vs MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100) |
| 17 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | EUROPE_FLOW | 1.0 | CROSS_SGP_MOMENTUM | DEPLOY | 23.50 | 23.5 | +0.1560 | 163 | +0.1534 | UNKNOWN |  | correlation_gate (ρ=0.797 vs MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5) |
| 18 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | SINGAPORE_OPEN | 1.5 | ATR_P50 | DEPLOY | 22.00 | 44.0 | +0.2407 | 137 | +0.1609 | ALARM |  | correlation_gate (ρ=1.000 vs MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30) |
| 19 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | COMEX_SETTLE | 1.5 | ORB_G5 | DEPLOY | 20.50 | 41.0 | +0.1756 | 253 | +0.0418 | ALARM |  | correlation_gate (ρ=1.000 vs MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100) |
| 20 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | TOKYO_OPEN | 1.5 | ORB_G5 | DEPLOY | 19.10 | 19.1 | +0.0774 | 267 | +0.0897 | UNKNOWN | ✓ | SELECTED |
| 21 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | US_DATA_1000 | 1.5 | ORB_G5 | DEPLOY | 18.80 | 18.8 | +0.0792 | 257 | +0.0254 | CONTINUE | ✓ | SELECTED |
| 22 | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | TOKYO_OPEN | 2.0 | ORB_G5 | DEPLOY | 18.60 | 18.6 | +0.0756 | 267 | +0.0897 | UNKNOWN |  | correlation_gate (ρ=0.845 vs MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5) |
| 23 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | US_DATA_1000 | 1.5 | VWAP_MID_ALIGNED | DEPLOY | 16.00 | 16.0 | +0.1334 | 130 | +0.0254 | UNKNOWN |  | correlation_gate (ρ=1.000 vs MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15) |
| 24 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | NYSE_OPEN | 1.0 | COST_LT12 | DEPLOY | 14.50 | 29.0 | +0.1200 | 262 | +0.1071 | ALARM |  | correlation_gate (ρ=1.000 vs MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5) |
| 25 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | US_DATA_1000 | 1.0 | VWAP_MID_ALIGNED | DEPLOY | 13.80 | 13.8 | +0.1116 | 134 | +0.0254 | UNKNOWN |  | correlation_gate (ρ=0.832 vs MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15) |
| 26 | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | US_DATA_1000 | 2.0 | VWAP_MID_ALIGNED | DEPLOY | 12.10 | 12.1 | +0.1044 | 126 | +0.0254 | UNKNOWN |  | correlation_gate (ρ=0.892 vs MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15) |
| 27 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | NYSE_OPEN | 1.5 | ORB_G5 | DEPLOY | 12.00 | 12.0 | +0.0519 | 251 | +0.1071 | UNKNOWN |  | correlation_gate (ρ=0.825 vs MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5) |
| 28 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | NYSE_OPEN | 1.5 | COST_LT12 | DEPLOY | 12.00 | 12.0 | +0.0519 | 251 | +0.1071 | UNKNOWN |  | correlation_gate (ρ=0.825 vs MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5) |
| 29 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | TOKYO_OPEN | 1.5 | COST_LT12 | DEPLOY | 10.15 | 20.3 | +0.0934 | 236 | +0.0897 | ALARM |  | correlation_gate (ρ=1.000 vs MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5) |
| 30 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | TOKYO_OPEN | 1.0 | COST_LT12 | DEPLOY | 7.10 | 7.1 | +0.0324 | 236 | +0.0897 | UNKNOWN |  | correlation_gate (ρ=0.815 vs MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5) |
| 31 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | CME_PRECLOSE | 1.0 | ORB_G8 | PAUSE | 3.68 | 4.9 | +0.0736 | 56 | -0.0917 | UNKNOWN |  | profile_gate |
| 32 | `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | CME_PRECLOSE | 1.0 | COST_LT08 | PAUSE | 3.15 | 4.2 | +0.0678 | 46 | -0.0917 | UNKNOWN |  | profile_gate |
| 33 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60` | US_DATA_1000 | 1.0 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |
| 34 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MES_ATR60` | NYSE_OPEN | 1.5 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |
| 35 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60` | NYSE_OPEN | 1.0 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |
| 36 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | COMEX_SETTLE | 1.5 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |
| 37 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | COMEX_SETTLE | 1.0 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |
| 38 | `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | CME_PRECLOSE | 1.0 | X_MES_ATR60 | STALE | 0.00 | 0.0 | +0.0000 | 0 | none | UNKNOWN |  | status=STALE |

## Rank #6 → #7 boundary evidence

- **Last selected (rank-within-selected):** {
  "rank": 21,
  "strategy_id": "MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15",
  "effective_annual_r": 18.8,
  "annual_r": 18.8,
  "trailing_expr": 0.0792,
  "status_reason": "Session HOT (+0.0254), ExpR=+0.0792, N=257"
}
- **First tail contender (highest-ranked DEPLOY candidate NOT selected):** {
  "rank": 3,
  "strategy_id": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12",
  "effective_annual_r": 40.6,
  "annual_r": 40.6,
  "trailing_expr": 0.1819,
  "status": "DEPLOY",
  "exclusion_reason": "correlation_gate",
  "rho_blocker": "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
  "rho_value": 1.0
}
- **Rank immediately after last selected (may be STALE/PAUSE):** {
  "rank": 22,
  "strategy_id": "MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
  "effective_annual_r": 18.6,
  "status": "DEPLOY",
  "status_reason": "Session HOT (+0.0897), ExpR=+0.0756, N=267"
}

- **r6 (last selected eff_annual_r):** 18.800
- **r7 (first tail contender eff_annual_r):** 40.600
- **r7 / r6 ratio:** 2.16

## Correlation gate inspection

Pairwise ρ across selected DEPLOY set (top 10 by |ρ|):

| A | B | ρ |
|---|---|---:|
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | -0.053 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | +0.041 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | +0.033 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | -0.032 |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | -0.017 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -0.016 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -0.014 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | +0.013 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | -0.012 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -0.011 |

Max |ρ| within selected set: **0.053** (reject threshold: 0.7).

## Verdict

**`TIGHT_BUT_JUSTIFIED`**

Tail contender has higher raw effective_annual_r than last selected, but was rejected by a structural gate (correlation or budget).

## Classification

**Stamped:** `CONDITIONAL`.

- The **cutoff verdict is VALID**: all 38 lanes enumerated; pre-committed decision rule followed; exclusion of the first tail contender is attributable to the canonical correlation gate at ρ=1.000 — same session, different filter variant of an already-selected lane, perfect redundancy. The same-session correlation gate is doing exactly what Carver Ch.11/12 and the allocator's `CORRELATION_REJECT_RHO=0.70` rule were designed to enforce. 15 of 21 DEPLOY-candidate ranks above the last selected lane are correlation-duplicates of already-selected same-session lanes (ρ between 0.80 and 1.00), mostly at ρ=1.00 (same strategy spec, different filter variant on the same session — same underlying trade-day fire set).
- The **replay-vs-persisted comparison is CONDITIONAL**: 4 of 6 persisted DEPLOY lanes differ from current replay. Attribution: `data/state/sr_state.json` mtime is 2026-04-19 (post-rebalance). The ALARM status on 4 lanes (`SINGAPORE_OPEN ATR_P50_O15`, `COMEX_SETTLE ORB_G5`, `NYSE_OPEN COST_LT12`, `TOKYO_OPEN COST_LT12`) cuts their `_effective_annual_r` in half via `SR_ALARM_DISCOUNT=0.50`, which promotes their COST_LT12 / OVNRNG_100 / ORB_G5 counterparts above them in rank. This is the allocator behaving correctly — not a bug. A fresh `rebalance_lanes.py --date 2026-04-20` run would materialize this corrected allocation; the 2026-04-18 snapshot is stale on this dimension.

## Secondary finding — portfolio orthogonality is excellent

Max pairwise |ρ| within the 6 selected DEPLOY lanes is **0.053** (reject threshold: 0.70). The allocator's correlation-aware greedy selection has produced a near-orthogonal 6-lane portfolio — every pair of deployed lanes has decision-level P&L correlation essentially at noise floor. This validates the correlation gate's role in the DEPLOY/TAIL boundary seen above: the "wasted" slots 2-21 (non-selected high-ranked candidates) are not wasted, they are redundant reflections of the 5 selected lanes the gate chose to keep.

## Actionable implications

1. **No capital action required** — the cutoff is working as designed.
2. **Consider refreshing the 2026-04-18 allocation** — an on-demand `python scripts/tools/rebalance_lanes.py --date 2026-04-20 --profile topstep_50k_mnq_auto` would fold in the SR ALARM liveness that's been accrued since 2026-04-19. This is a capital-neutral housekeeping step; it may or may not shift lanes depending on whether the ALARM is judged structural vs transient.
3. **6 X_MES_ATR60 STALE lanes (ranks 33-38)** are N=0 / no regime data. Not a blocker, but a corollary of memory's 2026-04-20 X_MES_ATR60 re-audit — the filter is parked pending further audit. No action needed from this audit.
4. **max_slots=7 but only 6 selected** — slot #7 is correlation-gated, not a bug. Rank 22 (TOKYO_OPEN RR2.0 ORB_G5) would be the 7th but fails ρ=0.845 vs TOKYO_OPEN RR1.5 ORB_G5; rank 23 (US_DATA_1000 VWAP_MID_ALIGNED) fails ρ=1.000 vs the selected US_DATA_1000 ORB_G5_O15. Both exclusions are canonical.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/allocator_scored_tail_audit.py
```

No randomness. Read-only DB. No writes to `validated_setups` / `experimental_strategies` / `live_config` / `lane_allocation.json`.
