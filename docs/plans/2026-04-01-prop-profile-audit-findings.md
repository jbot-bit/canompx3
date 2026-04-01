# Prop Profile Audit Findings (2026-04-01)

## VERIFIED MISMATCHES (with primary sources)

### CRITICAL: Tradeify DD Values Wrong

| Tier | Our Value | Correct Value | Source |
|------|-----------|---------------|--------|
| 50K  | $2,000    | $2,000        | MATCH  |
| 100K | $4,000    | **$3,000**    | saveonpropfirms.com/blog/tradeify-select-guide |
| 150K | $6,000    | **$4,500**    | same   |

Our values appear to be from old Tradeify "Growth" plan. Select plan has lower DD.
Impact: TYPE-B profiles (tradeify_50k_type_b, tradeify_100k_type_b) had inflated DD
budget calculations. The tier analysis earlier this session used wrong numbers.

### MEDIUM: Close Times

| Firm | Our Value | Correct | Source |
|------|-----------|---------|--------|
| TopStep | 16:00 ET | **16:10 ET** (3:10 PM CT) | topstep.com/express-funded-account-rules |
| Tradeify | 16:00 ET | **16:59 ET** | help.tradeify.co (403, confirmed via review sites) |

### LOW: Apex Consistency (safe direction)

| Field | Our Value | Correct | Source |
|-------|-----------|---------|--------|
| consistency_rule | 0.30 (30%) | **0.50 (50%)** since Mar 2026 | apex support article 40463260337819 |

Our value is MORE conservative. Safe direction. Update for accuracy.

## CODE LOGIC AUDIT: ALL PASS

- _parse_strategy_id: 20/20 strategy IDs parse correctly
- compute_profit_split_factor: all edge cases correct
- get_lane_registry: shadow merge, suppression, fallback all correct
- validate_dd_budget: caps, _PV, inactive skip all correct
- DD_PER_CONTRACT constants: valid for current deployment

## LANE OPTIMALITY: 2 SUBOPTIMAL

- NYSE_CLOSE: VOL_RV20_N20 (0.263) vs best VOL_RV25_N20 (0.325) gap=+0.063
- COMEX_SETTLE: ATR70_VOL RR1.0 (0.215) vs best ORB_VOL_8K RR1.5 (0.264) gap=+0.049
