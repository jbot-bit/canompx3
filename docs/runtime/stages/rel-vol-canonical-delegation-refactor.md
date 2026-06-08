---
task: "Resolve rel-vol canonical-delegation smell — extract shared core from strategy_discovery._compute_relative_volumes; strategy_fitness._enrich_relative_volumes delegates to it. Signature-preserving."
mode: IMPLEMENTATION
---

## Scope Lock

- trading_app/strategy_discovery.py
- trading_app/strategy_fitness.py
- tests/test_trading_app/test_strategy_discovery.py
- tests/test_trading_app/test_strategy_fitness.py
- docs/audit/2026-06-07-mutation-testing-capital-core.md

## Blast Radius

- `_compute_relative_volumes` is a canonical contract imported by 13 call sites (nested/discovery, regime/discovery, validation_provenance, 1 migration, 8 research scripts) — ALL positional `(con, features, instrument, orb_labels, all_filters)`. Public name + signature MUST be preserved. Refactor is internal-only: extract the per-(row, minute) median-baseline core into a shared private helper in strategy_discovery.py; the public fn becomes a thin wrapper. fitness `_enrich_relative_volumes` delegates to the same core.
- Behavior-neutral. Proven by a golden/characterization test pinning exact rel_vol output BEFORE extraction; must stay GREEN after.
- Reads: bars_1m (test fixtures write to tmp_path only — zero live-DB risk). Writes: none.
- pipeline/build_daily_features.py:1647 + init_db.py:191 hold "MUST match" parity comments — NOT touched, but the extraction must not change the computed value or it breaks parity with bars_1m rel_vol.

## Stage 1 ONLY
Refactor + equivalence test + suites. NO Tier-2 DB fixtures, NO re-mutation. Stop for review.
