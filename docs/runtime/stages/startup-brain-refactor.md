---
task: startup-brain-refactor
mode: IMPLEMENTATION
agent: codex
updated: 2026-04-13T08:35:00+00:00
capsule: docs/runtime/capsules/startup-brain-refactor.md
scope_lock:
  - context/
  - pipeline/system_brief.py
  - pipeline/work_capsule.py
  - pipeline/startup_index.py
  - pipeline/runtime_snapshot.py
  - scripts/tools/context_resolver.py
  - scripts/tools/project_pulse.py
  - scripts/tools/session_preflight.py
  - scripts/tools/render_startup_index.py
  - scripts/tools/refresh_runtime_snapshot.py
  - tests/test_context/
  - tests/test_pipeline/test_system_brief.py
  - tests/test_pipeline/test_work_capsule.py
  - tests/test_tools/test_context_resolver.py
  - tests/test_tools/test_project_pulse.py
  - tests/test_tools/test_pulse_integration.py
  - tests/test_tools/test_session_preflight.py
---

# Stage Notes

Refactor startup orientation away from synchronous collector fan-out.

The critical path must become metadata-driven:

- file packet registry for startup-owned surfaces
- generated startup index for task and file routing
- materialized runtime snapshot for pulse-style freshness reporting
- fast startup tools read generated artifacts, not live repo/runtime scans

Latency budgets for this stage:

- `system_brief`: under 1 second
- `session_preflight`: under 1 second
- `project_pulse --fast`: under 3 seconds

Do not move trading or research truth into this layer. This workstream owns
startup routing/read-model behavior only.

# Outcome

Implemented:

- packet/index-driven `system_brief`
- durable `work_capsule` contract
- materialized `runtime_snapshot`
- snapshot-only `project_pulse --fast`
- explicit `--refresh` recomputation path

Measured current behavior:

- `system_brief`: about `0.18s`
- `project_pulse --fast --no-cache`: about `0.40s`
- `project_pulse --fast` with fresh snapshot: about `0.20s`
- `refresh_runtime_snapshot`: about `4.12s`
- `session_preflight`: about `2.24s`

Residual debt:

- `session_preflight` is still slower than the original stretch target because
  it still depends on `system_context` policy evaluation
- repo-level drift still has unrelated Check 91 Phase 4 DB integrity failures
