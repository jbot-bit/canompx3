---
task: Fast Lane V2 Phase 5 report-only research review; avoid readiness/deployment wording that could imply live or capital authority.
mode: CLOSED
closed_date: 2026-05-22
closed_note: |
  Implemented a stdout-only Fast Lane research review reporter, renamed active
  downstream tokens to capital review, added a Phase 5 boundary drift guard,
  rebuilt fast_lane_status.yaml, and verified full drift with 0 blocking
  violations. No capital-class or live runtime state was mutated.
scope_lock:
  - scripts/tools/fast_lane_research_review.py
  - scripts/tools/strategy_lab_mcp_server.py
  - scripts/tools/fast_lane_status.py
  - scripts/tools/fast_lane_walk.py
  - pipeline/check_drift.py
  - tests/test_tools/test_fast_lane_research_review.py
  - tests/test_tools/test_fast_lane_status.py
  - tests/test_tools/test_fast_lane_walk.py
  - tests/test_pipeline/test_check_drift_fast_lane_phase5_boundary.py
  - docs/specs/fast_lane_state_graph.md
  - docs/plans/2026-05-21-fast-lane-v2-institutional-design.md
  - docs/runtime/fast_lane_status.yaml
---

## Blast Radius

- Add a stdout-only Fast Lane research review reporter. It may read Fast Lane roll-up, cherry-pick journal, and per-strategy strategy-lab payloads; it must not write capital-class state.
- Rename active downstream token language from deployment decision to capital review so the chain does not imply live authorization.
- Add a drift guard for Phase 5 boundary wording and capital-class write attempts.
- Update Fast Lane spec/design wording for the Phase 5 report-only contract.
- Rebuild `docs/runtime/fast_lane_status.yaml` after token changes.
- No writes to `gold.db`, `validated_setups`, `docs/runtime/chordia_audit_log.yaml`, lane allocation JSON, broker state, account routing, or `trading_app/live/`.

## Acceptance

1. Phase 5 emits only research-review recommendations: `KILL`, `PARK`, `BULLPEN`, `RECOMMEND_RESEARCH_REVIEW`, or `ESCALATE_CAPITAL_REVIEW`.
2. Highest output means open a separate human capital review packet; it is not live authority.
3. Default report scope is Fast Lane lineage only. Direct heavyweight rows remain context and cannot exceed research-review recommendation.
4. Phase 5 output contains `REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY` and avoids banned active-surface phrases.
5. Drift check blocks reintroduction of Phase 5 capital-write or deployment-candidate wording.
6. Targeted tests, targeted lint, `fast_lane_walk --dry-run`, and `pipeline/check_drift.py` pass.

## Verification

- `./.venv-wsl/bin/python -m pytest tests/test_tools/test_fast_lane_research_review.py tests/test_tools/test_strategy_lab_mcp_server.py tests/test_tools/test_fast_lane_status.py tests/test_tools/test_fast_lane_walk.py tests/test_pipeline/test_check_drift_fast_lane_phase5_boundary.py` -> 70 passed.
- `./.venv-wsl/bin/python -m pytest tests/test_tools/test_fast_lane_research_review.py tests/test_pipeline/test_check_drift_fast_lane_phase5_boundary.py` -> 9 passed after final lint cleanup.
- `./.venv-wsl/bin/python -m ruff check scripts/tools/fast_lane_research_review.py scripts/tools/strategy_lab_mcp_server.py scripts/tools/fast_lane_status.py scripts/tools/fast_lane_walk.py pipeline/check_drift.py tests/test_tools/test_fast_lane_research_review.py tests/test_tools/test_strategy_lab_mcp_server.py tests/test_tools/test_fast_lane_status.py tests/test_tools/test_fast_lane_walk.py tests/test_pipeline/test_check_drift_fast_lane_phase5_boundary.py` -> all checks passed.
- `./.venv-wsl/bin/python scripts/tools/fast_lane_status.py --write` -> rebuilt `docs/runtime/fast_lane_status.yaml` with 46 entries.
- `./.venv-wsl/bin/python scripts/tools/fast_lane_research_review.py --strategy-id MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30` -> report-only output with `PARK` and `REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY`.
- `./.venv-wsl/bin/python scripts/tools/fast_lane_walk.py --dry-run` -> all four chain steps rc 0; direct heavyweight backlog now shows `operator_capital_review`.
- `./.venv-wsl/bin/python pipeline/check_drift.py` -> `NO DRIFT DETECTED: 157 checks passed [OK], 0 skipped (DB unavailable), 20 advisory`.
