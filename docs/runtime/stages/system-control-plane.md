---
task: Unified project control plane — canonical system context and policy shell
mode: IMPLEMENTATION
scope_lock:
  - pipeline/system_context.py
  - scripts/tools/system_context.py
  - scripts/tools/session_preflight.py
  - scripts/tools/project_pulse.py
  - pipeline/system_authority.py
  - docs/governance/system_authority_map.md
  - tests/test_pipeline/test_system_context.py
  - tests/test_tools/test_session_preflight.py
  - tests/test_tools/test_project_pulse.py
  - HANDOFF.md
blast_radius:
  - pipeline/system_context.py (new canonical context snapshot + policy evaluation layer for repo/dev control-plane truth)
  - scripts/tools/system_context.py (human/tool CLI read model for the canonical context contract)
  - scripts/tools/session_preflight.py (consume shared policy decisions instead of local warning/blocker logic)
  - scripts/tools/project_pulse.py (consume shared context identity/control-plane summary instead of partial local rebuild)
  - pipeline/system_authority.py + docs/governance/system_authority_map.md (register the new canonical context surface in project authority docs)
  - tests (lock expected mutating-session behavior and pulse identity behavior to the new shared contract)
updated: 2026-04-12T00:00:00Z
agent: codex
---
