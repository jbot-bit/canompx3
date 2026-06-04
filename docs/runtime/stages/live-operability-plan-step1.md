# Stage: live-operability-plan-step1

task: Land the 2026-06-04 live-app operability plan + execute its smallest-first diff (runbook tier wording + additive launch-tier report metadata). Additive/doc-only; no gate-logic, dashboard, or launch-behavior change.

mode: IMPLEMENTATION

scope_lock:
  - docs/plans/2026-06-04-live-app-operability-readiness-plan.md
  - docs/plans/2026-05-29-live-trading-readiness-runbook.md
  - scripts/tools/live_readiness_report.py
  - tests/test_tools/test_live_readiness_report.py
  - docs/runtime/stages/live-operability-plan-step1.md

## Blast Radius

- docs/plans/2026-06-04-live-app-operability-readiness-plan.md — new file, the ingested plan; zero code callers.
- docs/plans/2026-05-29-live-trading-readiness-runbook.md — prose-only section added (dashboard-open vs live-launch-allowed three-tier contract). No parser surface.
- scripts/tools/live_readiness_report.py — adds 5 ADDITIVE keys (`schema_version`, `startup_status`, `signal_status`, `live_status`, `launch_impact`) to the `report` dict at the assembly site. Derived from already-computed `strict_zero_warn["green"]`. NO gate logic, blocker semantics, or launch behavior changed. Consumers (`_render_text`, `_build_profile_proof_pack`, JSON emit) access by key — additive keys are non-breaking. Reads: none new. Writes: none.
- tests/test_tools/test_live_readiness_report.py — extends existing happy-path + blocked-path tests to assert the 5 new fields. No production behavior under test changed.
- Capital-adjacency: live_readiness_report.py is Tier B. Full stage-gate: drift green + targeted tests + self-review + adversarial pass before claim-of-done. If adding fields requires touching gate logic, STOP and surface.
