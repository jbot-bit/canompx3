---
task: fix-allocator-orb-minutes-hardcode
mode: IMPLEMENTATION
phase: 1/1
spec: capital-class audit finding 2026-04-30 (evidence-auditor agent ab19fcc9e39814af2)
created: 2026-05-01
scope_lock:
  - trading_app/lane_allocator.py
  - tests/test_trading_app/test_lane_allocator.py
  - tests/test_tools/test_generate_profile_lanes.py
  - scripts/tools/generate_profile_lanes.py
  - research/garch_a4b_binding_budget_replay.py
  - docs/audit/results/2026-04-30-allocator-orb-minutes-hardcode-audit.md
---

# Stage: Allocator orb_minutes structural fix

## Blast Radius

- `trading_app/lane_allocator.py` — modifies 5 hardcoded `orb_minutes=5` sites (lines 116, 131, 388, 525, 786) plus dataclass and signatures.
  - `LaneScore` dataclass: add `orb_minutes: int` field
  - `_per_month_expr()`: add `orb_minutes` parameter; substitute into both queries
  - `compute_lane_scores()`: select `orb_minutes` from `validated_setups`; pass through
  - `compute_pairwise_correlation()`: use `s.orb_minutes` not hardcoded 5
  - `compute_orb_size_stats()`: GROUP BY now includes `orb_minutes`; key is 3-tuple
  - `_compute_session_regime()`: KEEP O5 as deliberate reference signal; add explicit comment
- Consumers of `orb_size_stats` updated for 3-tuple key (lines 635, 637, 848, 849)
- `save_allocation()`: include `orb_minutes` in lane dict
- Reads: `gold.db` read-only via `validated_setups`, `orb_outcomes`, `daily_features`. Writes: `docs/runtime/lane_allocation.json` only on rerun (separate stage step).
- No callers outside `lane_allocator.py` import `_per_month_expr` (private, leading underscore). `compute_lane_scores`, `compute_pairwise_correlation`, `compute_orb_size_stats`, `save_allocation` are public; called from allocator entry script — search confirmed.

## Acceptance

- All 5 hardcoded `orb_minutes = 5` sites either use a parameter OR have a comment justifying why O5 is the deliberate reference
- Existing tests for `lane_allocator` pass: `python -m pytest tests/test_trading_app/test_lane_allocator.py -v`
- `python pipeline/check_drift.py` passes
- Allocator dry-run on 2026-04-18 reproduces existing O5 lane stats EXACTLY (no regression on the 4 unaffected lanes)
- O15 lane stats CHANGE measurably (the bug fix)
- Self-review: simulate (a) O15 strategy with sparse data, (b) strategy_id missing from validated_setups (impossible but check), (c) pairwise correlation between O5 and O15 lanes
