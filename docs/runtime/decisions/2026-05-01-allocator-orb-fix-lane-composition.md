# Operator Decision: Accept New 7-Lane Composition (Post-orb_minutes Fix)

**Date:** 2026-05-01
**Trigger:** PR #189 lands; live `lane_allocation.json` needs to be replaced with the corrected output
**Decision required:** Accept the 7-lane composition produced by the corrected allocator, OR partial-accept (keep old where unchanged, defer new entrants), OR hold all O15 paused (status quo from PR #188 interim) pending more review

---

## TL;DR for tomorrow morning

- Allocator was scoring O15 strategies against O5 data (fixed in PR #189)
- Rerun produces 7 lanes (was 6); all 7 are FDR-validated and active in `validated_setups`
- 3 of the 6 original lanes are dropped by the corrected allocator in favor of better candidates that became visible under correct math
- 1 brand-new aperture (O30) appears for the first time in the live composition
- Recommended: **accept the full 7-lane composition** (Option A); the new entrants all have lifetime N>400, positive lifetime ExpR, and the allocator is now using correct math to rank them

---

## Composition diff

### Pre-fix (buggy — 6 lanes, all keyed as O5 implicit)

| # | strategy_id | trailing_expr (O5 wrong) | Audit ExpR (O15 correct, unfiltered) |
|---|---|---|---|
| 1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 0.1892 | n/a (already O5) |
| 2 | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 0.2407 ⚠ wrong aperture | ~0.09 unfiltered |
| 3 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 0.1756 | n/a (already O5) |
| 4 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 0.1200 | n/a (already O5) |
| 5 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 0.0934 | n/a (already O5) |
| 6 | MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 0.0792 ⚠ wrong aperture | ~0.0708 unfiltered |

### Post-fix (corrected — 7 lanes, explicit `orb_minutes` per lane)

| # | strategy_id | aperture | trailing_expr (filtered, correct) | annual_r |
|---|---|---|---|---|
| 1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | O5 | 0.1892 | 46.1 |
| 2 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | O5 | 0.1756 | 41.0 |
| 3 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 ✱ | O5 | 0.1200 | 29.0 |
| 4 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 ✱ | O5 | 0.1877 | 28.1 |
| 5 | MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 ✱ | O15 | 0.2208 | 25.3 |
| 6 | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30 ✱ | O30 | 0.1773 | 32.9 |
| 7 | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 (corrected) | O15 | 0.1332 | 25.0 |

✱ = newly entering composition; ranks higher under corrected per-aperture math

### Mapping — what changed and why

| Change | Old | New | Reason |
|---|---|---|---|
| Filter swap | `MNQ_NYSE_OPEN_*_COST_LT12` | `MNQ_NYSE_OPEN_*_ORB_G5` | Same session, different filter — now ranks higher under correct math |
| Filter swap | `MNQ_TOKYO_OPEN_*_COST_LT12` | `MNQ_TOKYO_OPEN_*_COST_LT08` | Same session, tighter cost band — ExpR 0.0934 → 0.1877 |
| Filter swap | `MNQ_US_DATA_1000_*_ORB_G5_O15` | `MNQ_US_DATA_1000_*_VWAP_MID_ALIGNED_O15` | Different filter on same session+aperture; ExpR 0.0708 → 0.2208 (filter is doing real work) |
| New | — | `MNQ_SINGAPORE_OPEN_*_ATR_P50_O30` | Brand-new O30 lane fits budget under correct DD math (was being squeezed out by the wrong O5 P90 estimates) |
| Corrected | `MNQ_SINGAPORE_OPEN_*_ATR_P50_O15` (ExpR 0.2407 wrong) | (same id, ExpR 0.1332 correct) | Aperture corrected from O5 to actual O15 |
| Unchanged | `MNQ_EUROPE_FLOW_*_ORB_G5` (O5) | (same) | Was already O5; bug didn't affect |
| Unchanged | `MNQ_COMEX_SETTLE_*_ORB_G5` (O5) | (same) | Was already O5; bug didn't affect |

---

## Validation history of the 4 new entrants

All 4 are FDR-validated with `status=active` in `validated_setups` and have lifetime N ≥ 427 going back to 2019-05-06. Unfiltered baselines:

| strategy | aperture | N (lifetime) | unfiltered lifetime ExpR | unfiltered trailing 12mo ExpR | filtered allocator ExpR |
|---|---|---|---|---|---|
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | O5 | 1764 | +0.0830 | +0.1071 | +0.1200 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | O5 | 1794 | +0.0732 | +0.0565 | +0.1877 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | O15 | 1555 | +0.1094 | +0.0896 | +0.2208 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | O30 | 1779 | +0.0735 | +0.1443 | +0.1773 |

**Note:** The filter ExpR > unfiltered ExpR for all 4, so these filters are *not* vestigial in the same way the prior 6-lane filter-vestigialness audit (Target B, currently re-auditing per agent `a0dc702e2c12ad67c`) suggested for the previous cohort. Caveat: the filter-vestigialness audit was on a different lane set; not directly comparable.

---

## Decision options

### Option A — Accept full 7-lane composition (RECOMMENDED)

Replace the live JSON with the 4-corrected + 4-new + 3-dropped composition. Rationale:

- All 7 are FDR-validated (status=active, BH-FDR survived)
- Allocator is now using correct math
- The 4 newcomers all have lifetime N > 400 and positive lifetime ExpR
- 3 sessions retain SOME representation; lane swaps are filter-only on same session
- Aperture diversification (4 O5 / 2 O15 / 1 O30) reflects actual validated breadth

Risks:
- Real-money flip is still gated on OOS power floor (per `feedback_oos_power_floor.md`). Until OOS N grows, this is signal-only by policy. Real cap at risk: **none yet** (XFA not connected per memory `f1_xfa_active_correction.md`)
- The 3 lane swaps were not individually re-audited under the corrected allocator math; they ranked higher because the allocator can now SEE them ranked correctly relative to O15 lanes (which previously crowded budget under wrong p90 estimates)

### Option B — Partial accept (keep old where unchanged, hold new)

Keep the 2 unchanged O5 lanes (EUROPE_FLOW, COMEX_SETTLE) and the corrected SINGAPORE_OPEN_O15. Hold the 3 new entrants and the new O30 lane until per-lane re-audit. PR #188's interim O15 pauses can lift on SINGAPORE_OPEN_O15 (it's now correctly scored).

Use this if you want to limit changes to one category (correcting bug) without compounding into composition shifts.

### Option C — Status quo (PR #188 interim — 4 O5 lanes only)

Don't flip the JSON; keep the 4-lane O5-only composition that PR #188 produced. Wait for separate review of the new entrants.

Use this if any of the new entrants raises a concern (e.g. you want to read the validation lineage of VWAP_MID_ALIGNED_O15 first).

---

## Recommended action sequence (Option A)

1. Wait for PR #189 to merge (CI gate)
2. Pull main: `git checkout main && git pull`
3. Re-run allocator: `DUCKDB_PATH=$PWD/gold.db python scripts/tools/rebalance_lanes.py --date $(date +%Y-%m-%d)` (will compute as-of-today, slightly different from 2026-04-18 baseline above)
4. Inspect the output JSON; if composition matches expectations, commit:
   `git add docs/runtime/lane_allocation.json && git commit -m "ops(allocator): rebalance to corrected per-aperture composition"`
5. Allocator state is now correct; signal-only behavior continues until F-1 XFA gate flips and OOS power clears

## Open questions / follow-ups

- Per-lane OOS power calc for the 4 new entrants (defer to Target A's full audit completion)
- Re-audit of "filter vestigialness" claim (agent `a0dc702e2c12ad67c` running) — applies to the prior 6-lane cohort not directly to this composition, but findings may inform whether the new filter swaps are doing real work or just rearranging deck chairs
- Memory entry for "single-aperture-assumption bug class" (added: `feedback_allocator_orb_minutes_hardcode_2026_04_30.md`)
