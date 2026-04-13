+++
capsule_id = "startup-brain-refactor"
title = "startup-brain-refactor"
status = "handoff"
branch = "wt-codex-startup-brain-refactor"
worktree_name = "startup-brain-refactor"
tool = "codex"
task_id = "system_orientation"
route_id = "system_orientation"
briefing_level = "mutating"
purpose = "Refactor startup orientation to metadata-driven packets and snapshots; remove expensive collector fan-out from fast path"
summary = "Refactor startup orientation so fast startup reads generated metadata and snapshots instead of recomputing repo/runtime state live."
objective = "Deliver a packet-driven startup index plus runtime snapshot split, then rewire system_brief, session_preflight, and project_pulse fast mode onto those contracts with verification and code review."
next_step = "Commit the isolated worktree branch, then treat the unrelated Phase 4 SHA drift as a separate data-integrity task."
created_at = "2026-04-13T05:27:36.754714+00:00"
updated_at = "2026-04-13T08:35:00+00:00"
stage_path = "docs/runtime/stages/startup-brain-refactor.md"
task_domains = ["repo_governance"]
authorities = ["AGENTS.md", "HANDOFF.md", "CLAUDE.md", "CODEX.md", "docs/governance/system_authority_map.md"]
scope_paths = [
  "context/",
  "pipeline/system_brief.py",
  "pipeline/work_capsule.py",
  "pipeline/startup_index.py",
  "pipeline/runtime_snapshot.py",
  "scripts/tools/context_resolver.py",
  "scripts/tools/project_pulse.py",
  "scripts/tools/session_preflight.py",
  "scripts/tools/render_startup_index.py",
  "scripts/tools/refresh_runtime_snapshot.py",
  "tests/test_context/",
  "tests/test_pipeline/test_system_brief.py",
  "tests/test_pipeline/test_work_capsule.py",
  "tests/test_tools/test_context_resolver.py",
  "tests/test_tools/test_project_pulse.py",
  "tests/test_tools/test_pulse_integration.py",
  "tests/test_tools/test_session_preflight.py",
]
out_of_scope = [".claude/", "trading logic beyond this workstream unless explicitly added"]
verification_commands = [
  "./.venv-wsl/bin/python -m ruff check context pipeline/startup_index.py pipeline/runtime_snapshot.py pipeline/system_brief.py pipeline/work_capsule.py scripts/tools/context_resolver.py scripts/tools/project_pulse.py scripts/tools/session_preflight.py scripts/tools/render_startup_index.py scripts/tools/refresh_runtime_snapshot.py tests/test_context tests/test_pipeline/test_system_brief.py tests/test_pipeline/test_work_capsule.py tests/test_tools/test_context_resolver.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py tests/test_tools/test_session_preflight.py",
  "./.venv-wsl/bin/python -m pytest tests/test_context tests/test_pipeline/test_system_brief.py tests/test_pipeline/test_work_capsule.py tests/test_tools/test_context_resolver.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py tests/test_tools/test_session_preflight.py -q",
  "./.venv-wsl/bin/python scripts/tools/render_startup_index.py",
  "./.venv-wsl/bin/python pipeline/check_drift.py",
]
acceptance_criteria = [
  "system_brief resolves route and owners from generated startup metadata rather than live collector fan-out",
  "session_preflight remains an orientation tool and reports snapshot freshness instead of recomputing runtime state",
  "project_pulse fast mode reads a materialized runtime snapshot rather than recomputing expensive collectors inline",
  "stale runtime state is explicit and timestamped",
  "fast startup no longer depends on broad git/worktree scans or DB-backed runtime diagnostics by default",
]
risks = [
  "duplicated truth between file packets, task routes, and runtime snapshot contracts",
  "live collectors leaking back into fast startup through convenience helpers",
  "accidentally moving trading/runtime authority into startup metadata",
]
references = [
  "docs/plans/2026-04-13-startup-orientation-refactor.md",
  "docs/governance/system_authority_map.md",
  "scripts/tools/project_pulse.py",
  "scripts/tools/session_preflight.py",
  "scripts/tools/context_resolver.py",
  "pipeline/system_context.py",
]
decision_refs = ["docs/runtime/decision-ledger.md#current"]
debt_refs = ["docs/runtime/debt-ledger.md#open-debt"]
history_window = "current"
+++

# Context

The current startup path is too slow and too eager. Fast orientation is still
doing live collector fan-out across repo, git, worktree, and runtime surfaces.
That makes startup latency unpredictable and conflates orientation with runtime
diagnostics.

# Design

Split startup into two explicit planes:

- startup knowledge plane: generated packets and index for file/task routing
- runtime snapshot plane: materialized freshness and control-state summary

Default startup tools consume generated artifacts. Expensive refresh moves to
explicit refresh/deep commands.

# Decision Ledger

- 2026-04-13: This refactor will be implemented from the clean worktree branch
  state, not by porting the earlier uncommitted startup-brain experiment from
  the dirty root checkout.
- 2026-04-13: Fast startup is redefined as artifact reads, not collector
  recomputation.

# Verification Ledger

- `./.venv-wsl/bin/python scripts/tools/session_preflight.py --context codex-wsl`
  - PASS; current path measures about 2.24s and prints the packet-driven system brief
- `./.venv-wsl/bin/python scripts/tools/system_brief.py --format json`
  - PASS; current path measures about 0.18s
- `./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --no-cache --format json`
  - PASS; current path measures about 0.40s and returns startup packet only
- `./.venv-wsl/bin/python scripts/tools/refresh_runtime_snapshot.py`
  - PASS; writes `.canompx3-runtime/runtime_snapshot.json` in about 4.12s
- `./.venv-wsl/bin/python -m pytest tests/test_context/test_registry.py tests/test_pipeline/test_system_brief.py tests/test_pipeline/test_work_capsule.py tests/test_tools/test_context_resolver.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py tests/test_tools/test_session_preflight.py -q`
  - PASS; `114 passed`
- `./.venv-wsl/bin/python pipeline/check_drift.py`
  - startup-related drift checks pass; pre-existing DB-backed Check 91 still fails on orphaned hypothesis SHAs

# Open Questions

- Whether `session_preflight` should now be split away from the remaining
  `system_context` policy evaluation cost in a follow-up latency pass.
