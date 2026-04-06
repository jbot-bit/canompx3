# Filter Eligibility Context — Design v2

**Date:** 2026-04-07
**Status:** APPROVED — implementation begun
**Author:** Claude (design iterated through v1 critique → v2)
**Scope:** Operational transparency for filter status across deployed lanes and validated opportunities. Closes the "silent filter" problem where overlays, composites, and per-instrument validity are invisible to the trader.

---

## Problem Statement

The codebase has 82 filters routed across 12 sessions × 3 instruments via `get_filters_for_grid()`. Strategies are discovered with one primary filter encoded in `strategy_id` (e.g., `COST_LT10`). On top of that, several overlays (calendar, ATR velocity, bull-day short avoidance) are applied at execution time without surfacing in any UI.

Result: when the trader looks at the dashboard signal strip or runs `/trade-book`, they cannot answer:
- Which filters gate this lane's trades?
- What are today's actual filter values (break_delay=3min, cost_ratio=8%, pit_range_atr=0.15)?
- Did any pre-session condition fail today, blocking the lane from firing?
- Was an overlay (calendar SKIP, ATR velocity AVOID) applied silently?
- Is the data fresh? Is the validation fresh?

Carver (Systematic Trading) and Chan (Algorithmic Trading) — *cited from training memory, not verified against local PDFs in resources/* — both identify operational transparency as the dividing line between live-tradeable systems and backtesting toys. The current system is the latter.

## Goal

Build an `EligibilityReport` data structure that, given a strategy and a trading day, returns the explicit status of every condition that gates that strategy's trades. Surface it in two consumer views:

- **View A — Dashboard signal strip + pre-session brief** (for DEPLOYED lanes from `prop_profiles.ACCOUNT_PROFILES`)
- **View B — Filter universe audit** (separate page in trade sheet HTML, shows all routed filters and which are deployed)

## Design Principles (anti-silence, anti-bias)

1. **Nine explicit statuses**, never silent defaults (PASS / FAIL / PENDING / DATA_MISSING / NOT_APPLICABLE_INSTRUMENT / NOT_APPLICABLE_SESSION / NOT_APPLICABLE_DIRECTION / RULES_NOT_LOADED / STALE_VALIDATION)
2. **Atomic decomposition** — composite filters explode into per-component conditions
3. **Compact-by-default UI** — drill-down on demand, prevents information overload
4. **Eligibility ≠ trade decision** — separate icon from fitness, both must be green
5. **Two views, not one** — deployed lanes vs filter universe are different consumers
6. **Look-ahead audit as a hard gate** — Phase 4 routing expansion blocks any session where feature data isn't available pre-session
7. **Equal visual weight** for all statuses — no green/red semaphore, anti-confirmation-bias
8. **Validation freshness surfaced** — STALE_VALIDATION when filter not retested in 6+ months
9. **Removal symmetry** — Phase 4 tests both EXPANSION and DEMOTION of routed filters

## Data Structures

### `EligibilityReport` (immutable)
- strategy_id, instrument, session, trading_day, as_of_timestamp
- freshness_status: FRESH | PRIOR_DAY | STALE | NO_DATA
- conditions: list of `ConditionRecord`
- overall_status: ELIGIBLE | INELIGIBLE | DATA_MISSING | NEEDS_LIVE_DATA
- data_provenance: dict mapping condition name → source query identifier

### `ConditionRecord` (immutable)
- name, category, status (one of nine)
- observed_value, threshold, comparator
- source_filter, validated_for, last_revalidated
- explanation (one-sentence plain English)

## Phases

| Phase | Description | Files | Stage |
|-------|-------------|-------|-------|
| 0 | Type definitions + decomposition registry | `trading_app/eligibility/types.py`, `decomposition.py` | this stage |
| 1 | Eligibility builder + tests | `trading_app/eligibility/builder.py`, `test_eligibility_*.py` | this stage |
| 2 | Trade sheet integration (View A + View B) | `scripts/tools/generate_trade_sheet.py` | next stage |
| 3 | Dashboard live integration | `bot_state.py`, `bot_dashboard.html`, `session_orchestrator.py`, `execution_engine.py` (event emission) | stage 3 |
| 4 | Filter routing audit (separate research task) | `scripts/research/filter_routing_audit.py` | deferred |

## Scope of THIS Stage (Phase 0 + Phase 1)

- New directory `trading_app/eligibility/` with type definitions, decomposition registry, and builder
- Decomposition coverage for filter types used by CURRENT deployed lanes (ORB_G*, COST_LT*, OVNRNG_*, PIT_MIN, GAP_*, PDR_*, X_*_ATR*, FAST5, FAST10, CONT, DIR_LONG) and the overlays applied to those lanes (calendar, ATR velocity)
- Fixture-based tests proving every status enum is producible
- **No production code modified.** This stage adds new files only.

Future stages will add Trade Sheet integration (Phase 2) and Dashboard integration (Phase 3). The decomposition registry will be expanded incrementally as new filters need coverage; a drift check will enforce 1:1 mapping between `ALL_FILTERS` and the registry once the full coverage is in place (Phase 2 or 3).

## Risks

1. **Decomposition registry incompleteness.** Initially covers only deployed-lane filters. Mitigation: Phase 2 will add coverage for all `validated_setups` filters; Phase 3 will add a drift check.
2. **Data provenance traceability cost.** Storing source query identifiers adds memory. Mitigation: only store query name strings, not full query text.
3. **Calendar rules.json detection.** Need a way to distinguish "rules say NEUTRAL" from "rules file missing." Mitigation: `calendar_overlay.py` already returns NEUTRAL on both — wrap with an explicit file existence check in the eligibility builder, return RULES_NOT_LOADED if missing.

## Out of Scope (this design)

- Modifying `get_filters_for_grid()` routing (Phase 4 only, deferred)
- Adding new filters to `ALL_FILTERS`
- ML-based eligibility scoring (ML is dead in this codebase)
- Real-time alerts for STALE_VALIDATION conditions (separate feature)
- Cross-strategy correlation context (use `/regime-check` separately)

## Literature Grounding

| Claim | Source | Status |
|-------|--------|--------|
| "Operational transparency = trader trust" | Carver Systematic Trading Ch.11/18, Chan Algorithmic Trading Ch.3 | Training memory, not verified against local PDFs |
| "Look-ahead bias proportional to predictive power of unavailable information" | Pardo Ch.4, Aronson Ch.6 | Already canonical in `trading_app/config.py:1538-1540` |
| "Confirmation bias from hiding the test universe" | Aronson Ch.6, Harvey-Liu Backtesting | Training memory, not verified |
| "Information overload causes second-guessing" | Carver Systematic Trading | Training memory, not verified |
| "Multiple testing requires showing all tests, not just hits" | Benjamini-Hochberg 1995 | Canonical, in `resources/` and `pipeline/check_drift.py` |

Optional pre-implementation hardening: extract Carver and Pardo passages from local PDFs to verify claims 1, 3, 4. Deferred unless requested.
