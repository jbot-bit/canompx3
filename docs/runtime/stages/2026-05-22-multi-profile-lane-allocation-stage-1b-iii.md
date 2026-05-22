---
task: Stage 1b-iii — sweep 11 scripts/tools/* readers off direct lane_allocation.json literals onto the canonical resolver / loader; drain the grep-gate temporary allowlist
mode: CLOSED
closed_date: 2026-05-22
closed_note: |
  Acceptance: 0 'lane_allocation.json' literals across trading_app/* and
  scripts/tools/* outside the permanent allowlist (prop_profiles +
  lane_allocator + rebalance_lanes). _LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST
  is now an empty frozenset. 155 PASSED drift (844 pre-existing strict-IS
  carry-over unchanged). 662 passed on test_tools + grep-gate + parity
  slice. All 11 modules import cleanly. Grep-gate test fixtures rewired off
  live entries onto a synthetic test_only_synthetic_reader.py so the
  allowlist mechanism stays testable after the live set drained.
  Parent stage 1b sub-stage progress: 1b-i ✓, 1b-ii.a-1 ✓, 1b-ii.a-2 ✓,
  1b-ii.b ✓, 1b-iii ✓. Remaining: 1b-iv (open PR).
original_mode: IMPLEMENTATION
slug: 2026-05-22-multi-profile-lane-allocation-stage-1b-iii
parent_stage: docs/runtime/stages/2026-05-21-multi-profile-lane-allocation-stage-1b.md
parent_commit: 5370ed2f
doctrine_anchors:
  - .claude/rules/institutional-rigor.md § 4 (delegate to canonical sources, never re-encode)
  - feedback_fixture_vs_contract_drift_n1_2026_05_20.md (sweep tests/ fixtures alongside readers)
scope_lock:
  - scripts/tools/allocation_intel.py
  - scripts/tools/allocator_gate_audit.py
  - scripts/tools/deployable_shelf_gap.py
  - scripts/tools/deployment_coverage_audit.py
  - scripts/tools/fast_lane_status.py
  - scripts/tools/fast_lane_walk.py
  - scripts/tools/generate_trade_sheet.py
  - scripts/tools/go_portal.py
  - scripts/tools/live_readiness_report.py
  - scripts/tools/monitor_q4_band_shadow.py
  - scripts/tools/strategy_lab_mcp_server.py
  - pipeline/check_drift.py
  - tests/ fixture sweep (added to scope as discovered per fixture-vs-contract-drift watch)

## Blast Radius

- Three categories:
  - (A) Real path-resolution sites with hand-rolled `Path(...) / "docs" / "runtime" / "lane_allocation.json"` + `json.load`. Migrate to `resolve_allocation_json(profile_id)` where a profile_id is in scope; otherwise use the resolver with an empty/None profile_id only when the caller is profile-agnostic (e.g., docstring scans). For tools without a profile_id in scope, use `legacy_lane_allocation_path()` as the path source (no inlined literal). Files: allocation_intel, allocator_gate_audit, deployable_shelf_gap, generate_trade_sheet, live_readiness_report, monitor_q4_band_shadow, strategy_lab_mcp_server.
  - (B) Docstring-only mentions of the legacy filename (no path resolution code touched). Files: deployment_coverage_audit, fast_lane_status, fast_lane_walk. Reword "lane_allocation.json" → "allocation file" or "lane allocation file" in docstrings; behavior unchanged.
  - (C) Hybrid: error/log strings + docstring. Files: go_portal. Reword in-place.
- `live_readiness_report.py` has a `--allocation-path` argparse `help=` containing "Path to lane_allocation.json." — reword to "Path to lane allocation file." (user-facing help text; semantic preserved).
- `pipeline/check_drift.py` — drop all 11 entries from `_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST`. The frozenset will be empty after this sweep — the grep-gate's dead-allowlist-entry sub-check will then quietly do nothing, which is correct.
- `tests/` fixture sweep — per feedback_fixture_vs_contract_drift_n1_2026_05_20.md, any fixture that writes a literal `lane_allocation.json` file under the new dir or legacy path must be reviewed. Grep `tests/` at impl start; add to scope if found.
- Reads: gold.db read-only (unchanged). Writes: none.
- Adversarial-audit-gate: scripts/tools/* readers are NOT in the live-broker arming path. Per `.claude/rules/adversarial-audit-gate.md`, this stage is `[feature]`-tier (read-only audit tools + docstring rewords). evidence-auditor not required; self-review sufficient.

## Verification

1. `grep -rn "lane_allocation\.json" trading_app/ scripts/tools/` returns ONLY the 3 permanent-allowlist entries (prop_profiles.py / lane_allocator.py / rebalance_lanes.py).
2. `python pipeline/check_drift.py` reports 155 PASSED + 844 pre-existing strict-IS carry-over (baseline unchanged); grep-gate passes with empty temporary allowlist; dead-allowlist sub-check passes vacuously.
3. Spot-test each migrated tool runs `--help` cleanly (resolver imports resolve).
4. `pytest tests/test_tools/ tests/test_pipeline/test_check_drift_lane_allocation_*.py -q` passes.
5. Self-review walk on each Category-A migration: did the resolver call preserve profile-mismatch semantics? Did fail-closed branches stay fail-closed? Did paper-account fail-open stay fail-open?

## Acceptance criteria — Stage done when

- [ ] Zero `lane_allocation.json` literals across `scripts/tools/*.py` and `trading_app/*.py` outside permanent allowlist
- [ ] `_LANE_ALLOC_LITERAL_TEMPORARY_ALLOWLIST` is the empty frozenset
- [ ] Drift baseline preserved (155 PASSED + 844 pre-existing)
- [ ] Targeted tests pass
- [ ] Parent stage 1b sub-stage checklist ticks 1b-iii
- [ ] Commit pushed
- [ ] Open PR (1b-iv) — separate commit / step

## Notes

- 11 files; commit as ONE atomic migration since the migration is mechanically uniform (literal → resolver/helper). If a category-A file surfaces unusual semantics mid-sweep (e.g., compares two paths), split that file into its own follow-up commit.
- The grep-gate is substring-based — comment and docstring mentions count. This is the n=12+ instance of the canonical-inline-copy-parity bug class, per `feedback_canonical_inline_copy_parity_bug_class.md`.
