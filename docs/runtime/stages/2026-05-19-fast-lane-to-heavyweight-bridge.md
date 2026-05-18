# Stage B — Fast-lane to heavyweight Chordia bridge generator + Check #161

task: Build scripts/research/fast_lane_to_heavyweight_bridge.py that translates a fast-lane v5.1 result MD + source YAML into a heavyweight Chordia strict-unlock prereg DRAFT under docs/audit/hypotheses/drafts/. Add Check #161 check_bridge_methodology_rules_parity. Append CANONICAL_INLINE_COPIES entry. Sibling-coverage injection tests.

mode: IMPLEMENTATION

scope_lock:
  - scripts/research/fast_lane_to_heavyweight_bridge.py
  - pipeline/check_drift.py
  - pipeline/canonical_inline_copies.py
  - tests/test_pipeline/test_check_drift_bridge_methodology_rules_parity.py
  - tests/test_research/test_fast_lane_to_heavyweight_bridge.py

## Blast Radius

- scripts/research/fast_lane_to_heavyweight_bridge.py — new file, zero callers
- pipeline/check_drift.py — adds one check function + one CHECKS registration; existing 160 checks untouched
- pipeline/canonical_inline_copies.py — appends one InlineCopyPair; existing two entries untouched
- tests/test_pipeline/test_check_drift_bridge_methodology_rules_parity.py — new file, sibling-coverage injection tests for METHODOLOGY_RULES_APPLIED
- tests/test_research/test_fast_lane_to_heavyweight_bridge.py — new file, draft YAML schema + scope inheritance + theory_citation absence
- Reads: docs/audit/results/*fast-lane*.md (read-only), docs/audit/hypotheses/*fast-lane*.yaml (read-only), .claude/rules/backtesting-methodology.md (read-only via Check #161)
- Writes: docs/audit/hypotheses/drafts/<date>-<slug>-chordia-heavyweight-v1.draft.yaml when invoked
- Capital-class? No. Drafts/ is quarantine zone; LHP validator does not walk it. No mutation of chordia_audit_log.yaml, validated_setups, lane_allocation.json, or trading_app/live/.

## Acceptance criteria

1. `python pipeline/check_drift.py` count rises 160 → 161; all checks PASS (modulo pre-existing MGC trade-window orthogonal).
2. `pytest tests/test_research/test_fast_lane_to_heavyweight_bridge.py -v` — all tests PASS.
3. `pytest tests/test_pipeline/test_check_drift_bridge_methodology_rules_parity.py -v` — all tests PASS including mutation injection (one per RULE in METHODOLOGY_RULES_APPLIED).
4. Bridge generates a draft for the QUEUED entry end-to-end; draft contains theory_grant=false, NO theory_citation key, methodology_rules_applied matches canonical doctrine.
5. Check #161 catches: (a) extra rule in METHODOLOGY_RULES_APPLIED not in doctrine, (b) missing canonical rule, (c) doctrine parse failure (fail-closed).
6. Layer 2 meta-check (Check #159) still passes — new InlineCopyPair has live parity_check + test_file + ≥1 test per gated_constant.
