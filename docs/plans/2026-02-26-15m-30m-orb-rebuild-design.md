# 15m/30m ORB Rebuild — Design Document

**Date:** 2026-02-26
**Goal:** Rebuild all 15m/30m ORB data from scratch using the standard pipeline with E1/E2/E3 entry models, producing validated strategies and edge families for comparison against the existing 5m portfolio.

---

## Context

After the E0-to-E2 entry model swap (Feb 26 2026), the 15m/30m ORB data is stale:
- **E2 outcomes at 15m/30m:** never built (0 rows)
- **E1/E3 outcomes at 15m/30m:** exist (~2.56M rows) but pre-date the session architecture overhaul and E2 swap
- **validated_setups at 15m/30m:** 0 rows (validator never completed or everything failed)
- **edge_families at 15m/30m:** 0 rows

Decision: purge all stale 15m/30m data and rebuild from scratch.

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline | Standard (`outcome_builder` / `strategy_discovery` / `strategy_validator`) | No new modules. Standard pipeline is parameterized for `orb_minutes`. |
| NOT nested | The `trading_app/nested/` research pipeline is not used | Nested uses 5m entry bars and separate tables. Standard uses 1m entry bars and production tables. Simpler. |
| Purge first | DELETE all stale 15m/30m from `orb_outcomes` and `experimental_strategies` | Clean slate removes doubt about pre-E2 data provenance. |
| Scope | 15m + 30m, all 4 instruments (MGC, MNQ, MES, M2K) | Let data speak. If 30m is garbage, validation kills it. |
| Entry models | E1, E2, E3 (all current) | Full coverage. |
| Portfolio integration | Build everything, compare separately, manual curation | Prevents auto-mixing timeframes (correlation risk, multiple testing). |

## Code Fixes Applied (Pre-Requisites)

### Fix 1: Edge family cross-duration contamination (BUG)
`scripts/tools/build_edge_families.py` hashed families as `{instrument}_{trade_day_hash}`. A 5m and 15m strategy sharing trade days would merge into one family. Fixed by including `orb_minutes` in the hash prefix: `{instrument}_{orb_minutes}m_{trade_day_hash}`.

### Fix 2: ATR query provenance comment
`trading_app/strategy_validator.py` line 608 hardcodes `orb_minutes = 5` in the ATR query for regime waivers. This is correct (ATR is a daily stat, same across all orb_minutes rows; filtering to 5 avoids 3x row inflation). Added provenance comment explaining the intentional hardcode.

## Execution Phases

### Phase 1: Purge stale data
- DELETE from `orb_outcomes` WHERE `orb_minutes IN (15, 30)` (~2.56M rows)
- DELETE from `experimental_strategies` WHERE `orb_minutes IN (15, 30)` (~21K rows)
- Verify validated_setups and edge_families have 0 affected rows

### Phase 2: Build outcomes
- `outcome_builder.py --orb-minutes 15` and `--orb-minutes 30` for each instrument
- 4 instruments x 2 durations = 8 runs
- Generates E1, E2, E3 outcomes at each duration

### Phase 3: Discovery
- `strategy_discovery.py --orb-minutes 15` and `--orb-minutes 30` for each instrument
- 8 discovery runs total

### Phase 4: Validation
- `strategy_validator.py` for each instrument (picks up 15m/30m from experimental_strategies)
- Flags: `--min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward`

### Phase 5: Edge families + trade days
- `build_edge_families.py --all` (picks up all orb_minutes from validated_setups)
- `backfill_strategy_trade_days.py --all`

### Phase 6: Compare
- `report_edge_portfolio.py --all --slots`
- `research_portfolio_assembly.py` with/without 15m/30m
- Document findings, update canonical docs

## Success Criteria
- All 15m/30m data rebuilt fresh with E1/E2/E3
- Validation honestly reports survivors (or zero)
- Edge families cluster 15m/30m separately from 5m (no cross-duration contamination)
- Comparison report documents whether 15m/30m adds portfolio value
- Canonical docs updated with results
