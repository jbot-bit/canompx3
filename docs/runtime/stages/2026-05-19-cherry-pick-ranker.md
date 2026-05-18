# Stage A — Cherry-pick ranker + Check #160

task: Build scripts/research/cherry_pick_ranker.py that scores FAST_LANE PROMOTE survivors by heavyweight-Chordia-pass-probability. Add Check #160 check_cherry_pick_ranker_threshold_parity (T=3.79 from pre_registered_criteria.md Criterion 4). Append CANONICAL_INLINE_COPIES entry. Sibling-coverage injection tests.

mode: IMPLEMENTATION

scope_lock:
  - scripts/research/cherry_pick_ranker.py
  - pipeline/check_drift.py
  - pipeline/canonical_inline_copies.py
  - tests/test_pipeline/test_check_drift_cherry_pick_ranker_threshold_parity.py
  - tests/test_research/test_cherry_pick_ranker.py

## Blast Radius

- scripts/research/cherry_pick_ranker.py — new file, zero callers
- pipeline/check_drift.py — adds one check function + one CHECKS registration; existing 159 checks untouched
- pipeline/canonical_inline_copies.py — appends one InlineCopyPair to CANONICAL_INLINE_COPIES list; existing entry untouched
- tests/test_pipeline/test_check_drift_cherry_pick_ranker_threshold_parity.py — new file, ≥3 injection tests (mutation, missing-criterion, parse-failure)
- tests/test_research/test_cherry_pick_ranker.py — new file, score-ordering + score-formula + edge cases
- Reads: docs/runtime/promote_queue.yaml (read-only), docs/audit/results/*fast-lane*.md (read-only), docs/institutional/pre_registered_criteria.md (read-only via Check #160)
- Writes: docs/runtime/cherry_pick_ranking_<date>.csv (optional, --write flag only)
- Capital-class? No. Read-only descriptive output. No mutation of validated_setups, lane_allocation.json, chordia_audit_log.yaml, or trading_app/live/.

## Acceptance criteria

1. `python pipeline/check_drift.py` count rises 159 → 160; all checks PASS (modulo pre-existing MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 trade-window orthogonal).
2. `pytest tests/test_research/test_cherry_pick_ranker.py -v` — all tests PASS.
3. `pytest tests/test_pipeline/test_check_drift_cherry_pick_ranker_threshold_parity.py -v` — all tests PASS including mutation injection.
4. Ranker runs end-to-end on real promote_queue.yaml: ranks the 1 QUEUED entry MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30 with score in [0.0, 1.0]; REVOKED entry not scored.
5. Check #160 catches: (a) HEAVYWEIGHT_T_THRESHOLD mutation in ranker, (b) missing criterion 4 in doctrine, (c) Pyyaml/file parse failure (fail-closed).
6. Layer 2 meta-check (Check #159) still passes — new InlineCopyPair has live parity_check + test_file + ≥1 test per gated_constant.
