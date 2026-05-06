---
task: fix prop_profiles.parse_strategy_id silent fail-open by delegating to canonical parser
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - trading_app/paper_trade_logger.py
  - research/garch_profile_production_replay.py
  - tests/test_trading_app/test_prop_profiles.py
acceptance:
  - prop_profiles.parse_strategy_id raises ValueError on malformed input (matches canonical)
  - prop_profiles.parse_strategy_id is a thin delegator to trading_app.eligibility.builder.parse_strategy_id
  - paper_trade_logger and research/garch_profile_production_replay import from the canonical source
  - regression test covers malformed-input → ValueError
  - all existing tests pass; check_drift.py passes; behavioral audit 7/7
---

## Blast Radius

- trading_app/prop_profiles.py — body of `parse_strategy_id` (~40 lines) replaced with delegator. Function is imported by 2 external sites + called once internally at `get_profile_lane_definitions`. Behaviour change: malformed input now raises instead of returning default dict.
- trading_app/paper_trade_logger.py — single import-line change.
- research/garch_profile_production_replay.py — single import-line change.
- tests/test_trading_app/test_prop_profiles.py — add regression test for malformed → ValueError.
- Reads: gold.db (read-only via existing tests). Writes: none.
- Live data verified well-formed (10 paused MES + 1 active topstep_50k profile lane carry `_S<digits>` suffix; both parsers agree on those). Risk surface: any caller that fed malformed strings expecting silent defaults — none found in grep.

## Doctrine cited

- institutional-rigor.md § 4 — Delegate to canonical sources; never re-encode
- institutional-rigor.md § 6 — No silent failures
- integrity-guardian.md § 5 — Evidence over assertion (live runtime probe used to verify divergence)
- feedback_aperture_overlay_canonical_parser.md — explicit prior memo on this pattern
