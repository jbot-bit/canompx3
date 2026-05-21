---
task: Multi-profile lane_allocation — Stage 1b (authority-inversion: canonical resolver + reader migration + drift gates)
mode: IMPLEMENTATION
slug: 2026-05-21-multi-profile-lane-allocation-stage-1b
parent_stage: docs/runtime/stages/2026-05-21-multi-profile-lane-allocation-stage-1a.md
parent_commit: a95b7a91
doctrine_anchors:
  - .claude/rules/institutional-rigor.md § 4 (delegate to canonical sources, never re-encode)
  - CLAUDE.md § Source-of-Truth Chain Rule
  - docs/specs/lane_allocation_schema.md § 4 (reader contract), § 5 (drift-check coverage)
---

## Task

Invert authority on `lane_allocation.json` reads. After Stage 1a, ~17 production sites each
hand-resolve the legacy path `docs/runtime/lane_allocation.json`. Stage 1b promotes a single
canonical resolver in `trading_app/prop_profiles.py`, migrates every production reader to call
it, and lands two new drift gates (parity + grep-gate) so Stage 1d becomes a delete-only operation
instead of a sweep across 17 sites that could silently grow back between now and 1d.

This is NOT a mechanical 17-site flip. The Stage-1a "Out of scope" estimate of ~15 readers was
right in shape but the EV is in the authority inversion: one owner of path semantics, runtime
ambiguity detection, and a grep gate that prevents regression.

## Scope Lock

Production code:
- trading_app/prop_profiles.py
- trading_app/opportunity_awareness.py
- trading_app/live/session_orchestrator.py
- trading_app/prop_portfolio.py
- trading_app/pre_session_check.py
- pipeline/check_drift.py
- scripts/tools/allocation_intel.py
- scripts/tools/allocator_gate_audit.py
- scripts/tools/deployable_shelf_gap.py
- scripts/tools/deployment_coverage_audit.py
- scripts/tools/generate_trade_sheet.py
- scripts/tools/go_portal.py
- scripts/tools/live_readiness_report.py
- scripts/tools/monitor_q4_band_shadow.py
- scripts/tools/rebalance_lanes.py
- scripts/tools/strategy_lab_mcp_server.py

Tests:
- tests/test_trading_app/test_prop_profiles.py
- tests/test_pipeline/test_check_drift_lane_allocation_parity.py (new)
- tests/test_pipeline/test_check_drift_lane_allocation_grep_gate.py (new)
- any tests/ fixture writing legacy-shape `lane_allocation.json` (sweep at impl-start; add to scope as discovered)

Schema:
- docs/specs/lane_allocation_schema.md

## Blast Radius

- trading_app/prop_profiles.py — promote `resolve_allocation_json(profile_id) -> AllocationRead`
  as the canonical resolver. NamedTuple `(data: dict | None, source: Literal["new_path","legacy","missing"], path: Path | None)`.
  Owns: new-path preference, legacy fallback, profile-mismatch fail-closed, multi-file ambiguity
  hard-fail (`raise RuntimeError` if `>=2` files match `<profile_id>` glob under the new dir),
  provenance. Existing `load_allocation_lanes` / `load_paused_strategy_ids` are refactored to call
  the resolver. Public API surface preserved (no caller-shape changes from 1a).
- trading_app/live/session_orchestrator.py — broker-arming path; FAIL-CLOSED branch + operator-runbook log strings preserved verbatim.
- 4 other trading_app/ readers + 11 scripts/tools/ readers — flip each from `Path(...) / "docs" / "runtime" / "lane_allocation.json"` + `json.load` to the resolver. Per-site fallback logic forbidden per institutional-rigor § 4.
- pipeline/check_drift.py — three existing legacy-only checks (chordia gate / c8 gate / displaced validity) rewritten to enumerate per-profile files via resolver discovery surface AND the legacy path. Plus TWO new checks:
  - parity check (pre-named in `lane_allocation_schema.md` § 5): for every profile present under the new path, body MUST byte-equal the legacy file when legacy `profile_id` matches.
  - grep-gate check: scan `trading_app/**/*.py pipeline/**/*.py scripts/**/*.py` for direct `lane_allocation.json` literals; allowlist `prop_profiles.py` (resolver) + `lane_allocator.py` (writer) + `rebalance_lanes.py` (`--output` argparse help text). Fail-closed on any other match. This is the gate that makes Stage 1c/1d safe.
- Reads: `gold.db` read-only (existing). Writes: none from this stage (1a's dual-write unchanged).
- Drift checks: existing legacy checks continue to pass because legacy is still written; new per-profile coverage is a strict superset. Grep gate is new — first run audits the codebase + my migration.
- Cross-cutting: `multi-terminal-shared-file-hygiene.md` three-check protocol applies on any commit touching `docs/runtime/lane_allocation/` (new dir is under `docs/runtime/` so the shared-state commit guard fires).

## Verification

1. `pytest tests/test_trading_app/ tests/test_pipeline/ tests/test_tools/ -q` — all pass; resolver tests cover {new only, legacy only, both equal, both differ → ambiguity-on-mismatch behavior, profile mismatch on legacy, multi-file ambiguity hard-fail, missing both → `source="missing"`}.
2. `python pipeline/check_drift.py` — passes including the new grep-gate (which auto-catches any reader I missed and any fixture mocking the legacy shape).
3. Provenance smoke: rebalance `topstep_50k_mnq_auto`; consume from each migrated reader; assert `source == "new_path"` for every read. This is the data Stage 1d needs to fire safely.
4. Live-rebalance equivalence: `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto` then `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto` — output byte-identical to pre-1b baseline (capture baseline at impl start).
5. Grep audit: `grep -rn "lane_allocation\.json" trading_app/ pipeline/ scripts/tools/` returns ONLY allowlisted sites. Anything else = the grep-gate would have caught it; this is belt-and-suspenders.
6. Self-review pass on the resolver's three branches + ambiguity-fail + provenance field; PREMISE→TRACE→EVIDENCE→CONCLUSION on the live-broker path test.

## Acceptance criteria — Stage done when

- [ ] Resolver `resolve_allocation_json` + `AllocationRead` land in `prop_profiles.py` with multi-file-ambiguity hard-fail + provenance
- [ ] All 17 production readers migrated to the resolver; zero per-site fallback logic
- [ ] `check_drift.py` parity check + grep-gate check land with injection tests
- [ ] Test-fixture sweep complete (grep `tests/` for `lane_allocation.json` writes; migrate or assert via resolver)
- [ ] Schema spec § 4a (resolver contract) + § 5 (parity-landed marker) updated
- [ ] Live-rebalance + live-readiness smoke equivalence vs baseline
- [ ] `pipeline/check_drift.py` passes (all new + existing checks)
- [ ] Commit + push + open PR on `session/joshd-multi-profile-lane-allocation`

## Sub-stage progress (resume-state hygiene — update on every commit)

Each entry is a landed commit on `session/joshd-multi-profile-lane-allocation`.
A fresh `/clear`'d session should read this block FIRST to know the next
concrete task. Tick boxes when commits land; keep the unticked entry at the
top of the unfinished list short enough to be the "next session" target.

- [x] 1b-i — drift-check bedrock (parity check + grep-gate + temp allowlist) — `6916dbd9`
- [x] 1b-ii.a-1 — opportunity_awareness.py migrated to resolver; legacy_lane_allocation_path + new_lane_allocation_dir public helpers promoted — `871f0496`
- [x] 1b-ii.a-2 — prop_portfolio.py docstring + pre_session_check.py real migration (check_lane_mismatch + check_allocation_staleness_gate reword) + deployability.py comment-only rewords; allowlist shrunk from 4 → 1 trading_app entries — `ba5d640b`
- [x] 1b-ii.b — `trading_app/live/session_orchestrator.py` migration. HIGH-severity (live-broker arming, kill-switch path). Resolver-swap + corruption-vs-mismatch disambiguation + grep-gate allowlist shrink. evidence-auditor pass surfaced 1 operator-confusion finding (profile_id-mismatch error message); fix applied in the same patch. Slug: `2026-05-22-multi-profile-lane-allocation-stage-1b-ii-b`.
- [x] 1b-iii — swept 11 `scripts/tools/*` readers. Three categories: (A) real readers migrated to `resolve_allocation_json(profile_id)` or `legacy_lane_allocation_path()` (allocation_intel, allocator_gate_audit, deployable_shelf_gap, generate_trade_sheet, live_readiness_report, monitor_q4_band_shadow, strategy_lab_mcp_server); (B) docstring-only rewords (deployment_coverage_audit, fast_lane_status, fast_lane_walk); (C) hybrid (go_portal). Allowlist drained to empty frozenset. Fixture sweep concluded: production-shape `lane_allocation.json` literals under `tests/` are all in drift-check tests for unrelated check classes or for the grep-gate itself — none required production-fixture migration. Slug: `2026-05-22-multi-profile-lane-allocation-stage-1b-iii`.
- [ ] 1b-iv — open PR on `session/joshd-multi-profile-lane-allocation` once 1b-iii lands. Acceptance criteria below must all be ticked first.

## Out of scope (subsequent stages)

- Research scripts under `scripts/research/` (Stage 1c)
- Removing the legacy writer + reader fallback (Stage 1d) — gated on provenance smoke showing zero `source="legacy"` reads in a stabilization window
- Profile→account_id broker routing (Stage 2)
- Eval profile entry + rebalance (Stage 3)

## Notes

- Authority-inversion framing per user direction in design loop: the EV is "Stage 1d becomes a delete-only operation," not "17 readers updated."
- Grep-gate doctrine grounding: institutional-rigor § 4 (never re-encode canonical logic) + n=10+ canonical-inline-copy-parity bug-class instances (`feedback_canonical_inline_copy_parity_bug_class.md`). This stage adds a 12th instance template specifically for path literals.
- Test-fixture sweep is the hidden-blast-radius callout per `feedback_fixture_vs_contract_drift_n1_2026_05_20.md`. Treat under-estimation as the prior, not the exception.
