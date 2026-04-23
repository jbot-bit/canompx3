# Phase 2a — regime-gate empirical verification

- rebalance_date: `2026-04-18`
- profile: `topstep_50k_mnq_auto`
- regime window: `2025-10-01` -> `2026-04-18` (`6` months)
- lanes audited: `30`
- canonical deps: `_compute_session_regime` (UNFILT) + `filter_signal` (FILT)
- scope: Phase 2a of multi-phase audit roadmap / A2b-1 scope informing
- OOS consumption: zero (re-reads same window the allocator already consumed)

## Verdict counts

| code | count | meaning |
|---|---:|---|
| AGREE_SIGN | 30 | UNFILT and FILT_POOLED agree on sign — deployment verdict unchanged |
| SIGN_FLIP | 0 | UNFILT and FILT_POOLED disagree on sign — deployment verdict flips under patch |
| FILT_EMPTY | 0 | filter fires on 0 trades in window — FILT regime undefined |
| FILT_UNKNOWN | 0 | filter_type not in ALL_FILTERS |
| UNFILT_EMPTY | 0 | no pooled data in window |

## Bug materiality verdict

**BUG_COSMETIC** — all lanes agree on sign between UNFILT and FILT_POOLED. A2b-1 patch would not change deployment verdicts on the 2026-04-18 rebalance. Patch value is defensive / forward-looking, not immediate.

## Per-lane regime triplet

| strategy_id | inst | session | E/RR/CB/Omin | filter | UNFILT | FILT_POOLED (N) | FILT_LANE (N) | verdict |
|---|---|---|---|---|---:|---:|---:|---|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | E2/1.0/1/5 | `ORB_G5` | `+0.0418` | `+0.0418` (130) | `+0.0418` (130) | `AGREE_SIGN` |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | E2/1.0/1/5 | `COST_LT12` | `+0.0418` | `+0.0278` (120) | `+0.0278` (120) | `AGREE_SIGN` |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB` | MNQ | COMEX_SETTLE | E2/1.0/1/5 | `OVNRNG_100` | `+0.0418` | `+0.1212` (103) | `+0.1212` (103) | `AGREE_SIGN` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB` | MNQ | COMEX_SETTLE | E2/1.5/1/5 | `ORB_G5` | `+0.0418` | `+0.0418` (130) | `+0.0301` (128) | `AGREE_SIGN` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB` | MNQ | COMEX_SETTLE | E2/1.5/1/5 | `OVNRNG_100` | `+0.0418` | `+0.1212` (103) | `+0.1035` (101) | `AGREE_SIGN` |
| `MNQ_COMEX_SETTLE_E2_RR2.0_CB` | MNQ | COMEX_SETTLE | E2/2.0/1/5 | `ORB_G5` | `+0.0418` | `+0.0418` (130) | `+0.0514` (126) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1` | MNQ | EUROPE_FLOW | E2/2.0/1/5 | `CROSS_SGP_MOMENTUM` | `+0.1534` | `+0.2211` (83) | `+0.3711` (83) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | E2/1.0/1/5 | `OVNRNG_100` | `+0.1534` | `+0.2157` (107) | `+0.2157` (107) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | E2/1.5/1/5 | `COST_LT12` | `+0.1534` | `+0.1844` (118) | `+0.2242` (118) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | E2/1.5/1/5 | `ORB_G5` | `+0.1534` | `+0.1620` (135) | `+0.2285` (135) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | E2/1.5/1/5 | `OVNRNG_100` | `+0.1534` | `+0.2157` (107) | `+0.3015` (107) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR2.0_CB1` | MNQ | EUROPE_FLOW | E2/2.0/1/5 | `ORB_G5` | `+0.1534` | `+0.1620` (135) | `+0.2311` (135) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1` | MNQ | EUROPE_FLOW | E2/1.5/1/5 | `CROSS_SGP_MOMENTUM` | `+0.1534` | `+0.2211` (83) | `+0.3607` (83) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | E2/1.0/1/5 | `CROSS_SGP_MOMENTUM` | `+0.1534` | `+0.2211` (83) | `+0.2211` (83) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | E2/1.0/1/5 | `ORB_G5` | `+0.1534` | `+0.1620` (135) | `+0.1620` (135) | `AGREE_SIGN` |
| `MNQ_EUROPE_FLOW_E2_RR1.0_CB1` | MNQ | EUROPE_FLOW | E2/1.0/1/5 | `COST_LT12` | `+0.1534` | `+0.1844` (118) | `+0.1844` (118) | `AGREE_SIGN` |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_C` | MNQ | NYSE_OPEN | E2/1.5/1/5 | `COST_LT12` | `+0.1071` | `+0.1071` (133) | `+0.0320` (126) | `AGREE_SIGN` |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_O` | MNQ | NYSE_OPEN | E2/1.5/1/5 | `ORB_G5` | `+0.1071` | `+0.1071` (133) | `+0.0320` (126) | `AGREE_SIGN` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_O` | MNQ | NYSE_OPEN | E2/1.0/1/5 | `ORB_G5` | `+0.1071` | `+0.1071` (133) | `+0.1071` (133) | `AGREE_SIGN` |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_C` | MNQ | NYSE_OPEN | E2/1.0/1/5 | `COST_LT12` | `+0.1071` | `+0.1071` (133) | `+0.1071` (133) | `AGREE_SIGN` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_` | MNQ | SINGAPORE_OPEN | E2/1.5/1/15 | `ATR_P50` | `+0.1609` | `+0.2173` (104) | `+0.1566` (103) | `AGREE_SIGN` |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_` | MNQ | SINGAPORE_OPEN | E2/1.5/1/30 | `ATR_P50` | `+0.1609` | `+0.2173` (104) | `+0.1895` (101) | `AGREE_SIGN` |
| `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_` | MNQ | TOKYO_OPEN | E2/1.0/1/5 | `COST_LT12` | `+0.0897` | `+0.0949` (127) | `+0.0949` (127) | `AGREE_SIGN` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_` | MNQ | TOKYO_OPEN | E2/1.5/1/5 | `COST_LT12` | `+0.0897` | `+0.0949` (127) | `+0.2230` (127) | `AGREE_SIGN` |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_` | MNQ | TOKYO_OPEN | E2/1.5/1/5 | `ORB_G5` | `+0.0897` | `+0.0897` (137) | `+0.2113` (137) | `AGREE_SIGN` |
| `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_` | MNQ | TOKYO_OPEN | E2/2.0/1/5 | `ORB_G5` | `+0.0897` | `+0.0897` (137) | `+0.2701` (137) | `AGREE_SIGN` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB` | MNQ | US_DATA_1000 | E2/1.5/1/15 | `ORB_G5` | `+0.0254` | `+0.0254` (134) | `+0.1371` (112) | `AGREE_SIGN` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB` | MNQ | US_DATA_1000 | E2/1.0/1/15 | `VWAP_MID_ALIGNED` | `+0.0254` | `+0.0781` (63) | `+0.2877` (67) | `AGREE_SIGN` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB` | MNQ | US_DATA_1000 | E2/1.5/1/15 | `VWAP_MID_ALIGNED` | `+0.0254` | `+0.0781` (63) | `+0.3033` (62) | `AGREE_SIGN` |
| `MNQ_US_DATA_1000_E2_RR2.0_CB` | MNQ | US_DATA_1000 | E2/2.0/1/15 | `VWAP_MID_ALIGNED` | `+0.0254` | `+0.0781` (63) | `+0.2576` (56) | `AGREE_SIGN` |

## Self-consistency

UNFILT column was computed via canonical `_compute_session_regime` and matched `LaneScore.session_regime_expr` for every audited lane (harness would HALT otherwise).

## Next phase

Results feed Phase 2 Stage-1 scope doc `docs/audit/hypotheses/2026-04-18-a2b-1-regime-gate-filtered-patch-preregistered.md`:

- SIGN_FLIP count → whether A2b-1 is BUG_MATERIAL (high-priority) or BUG_COSMETIC (defensive).
- FILT_EMPTY count → whether the patch needs a fallback policy for lanes whose filter is rare in the 6mo window.
- FILT_POOLED vs FILT_LANE divergence → whether the patch should consume the baseline pool + filter (conservative) or the lane's own dims + filter (ground-truth).

