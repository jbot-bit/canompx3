# Startup Orientation Refactor

Date: 2026-04-13
Owner: Codex workstream `startup-brain-refactor`
Branch: `wt-codex-startup-brain-refactor`

## Problem

The current startup path is architecturally wrong for interactive use.

Observed failure mode:

- `project_pulse --fast --no-cache` is taking roughly 30 seconds in real usage
- the fast path still performs multiple live collectors and repeated repo scans
- orientation is mixing two concerns:
  - startup knowledge routing
  - runtime freshness diagnostics

This produces slow, duplicated, and misleading startup behavior.

## Design Goals

1. Startup must read the smallest honest context packet for the current task or touched files.
2. Fast orientation must not depend on live repo scans, DB reads, or worktree conflict scans.
3. Runtime freshness diagnostics remain available, but they must be moved off the fast path.
4. Generated artifacts must be explicit and stale-aware rather than silently recomputed.
5. Ownership must remain honest:
   - startup metadata is startup metadata
   - trading/runtime truth stays with its canonical owners

## Architectural Split

### 1. Startup Knowledge Plane

This plane answers:

- which file or task is being worked on
- which doctrine applies
- which canonical companions matter
- which tests and verification profile apply
- which expansion triggers justify deeper context

This plane is generated and cheap to read.

Planned surfaces:

- `context/file_packets.py`
- `context/registry.py`
- `pipeline/startup_index.py`
- `scripts/tools/render_startup_index.py`
- `.canompx3-runtime/startup_index.json`

### 2. Runtime Snapshot Plane

This plane answers:

- what the last known runtime/control state was
- when it was refreshed
- whether the snapshot is stale

This plane is refreshed explicitly or lazily, but not as part of the default startup path.

Planned surfaces:

- `pipeline/runtime_snapshot.py`
- `scripts/tools/refresh_runtime_snapshot.py`
- `.canompx3-runtime/runtime_snapshot.json`

## Contracts

### File Packet

```python
@dataclass(frozen=True)
class FilePacket:
    path: str
    role: str
    doctrine: tuple[str, ...]
    canonical_companions: tuple[str, ...]
    test_files: tuple[str, ...]
    verification_profile: str
    published_surfaces: tuple[str, ...]
    expansion_triggers: tuple[str, ...]
    tags: tuple[str, ...]
```

### Startup Index

Generated read-model containing:

- file packet map
- task route map
- doctrine/test reverse indexes
- verification profile map

### Runtime Snapshot

Materialized operator summary containing:

- branch/head
- snapshot timestamps
- last known lifecycle/deployment/worktree rollups
- stale markers and degradation notes

## Tool Changes

### `system_brief`

New role:

- resolve route from task id, touched files, or capsule
- read startup index
- emit bounded orientation packet

Must not do by default:

- git status fan-out
- worktree conflict scans
- DB-backed lifecycle/deployment reads
- repeated ledger parsing

### `session_preflight`

New role:

- validate interpreter/context
- read `system_brief`
- show snapshot freshness
- warn if runtime snapshot is stale

Must not perform broad runtime discovery on startup.

### `project_pulse`

New modes:

- `--fast`: snapshot-read only
- `--refresh`: recompute runtime snapshot
- optional `--deep`: explicit expensive reads

The current assumption that “cheap collectors are always fresh” is removed.

## Migration Phases

### Phase 1

Introduce packet registry and startup index renderer.

### Phase 2

Refactor `context_resolver` and `system_brief` to consume the startup index.

### Phase 3

Introduce runtime snapshot builder and snapshot contract.

### Phase 4

Refactor `project_pulse --fast` to read the snapshot only.

### Phase 5

Trim `session_preflight` to packet-based orientation plus snapshot freshness.

### Phase 6

Add tests, latency assertions, and drift coverage for the new contracts.

## Acceptance Criteria

1. `system_brief` reads generated startup metadata instead of recomputing live repo state.
2. `session_preflight` no longer blocks on broad collector fan-out.
3. `project_pulse --fast --no-cache` is replaced or redefined so the fast path is still fast.
4. Snapshot freshness is explicit and timestamped.
5. The fast path succeeds without requiring DB or worktree scans unless the task explicitly expands into them.
6. Existing behavioral coverage remains green after the refactor.

## Verification

Targeted verification surface:

- `tests/test_context/test_registry.py`
- `tests/test_tools/test_context_resolver.py`
- `tests/test_pipeline/test_system_brief.py`
- `tests/test_pipeline/test_work_capsule.py`
- `tests/test_tools/test_project_pulse.py`
- `tests/test_tools/test_pulse_integration.py`
- `tests/test_tools/test_session_preflight.py`

Required gates before sign-off:

- targeted `ruff check`
- targeted pytest surface
- generated artifacts rendered
- `pipeline/check_drift.py`
- code review focused on startup honesty and duplicated truth

## Review Risks

1. Reintroducing live collectors into the fast path through helper reuse.
2. Duplicating truth between file packets, task routes, and runtime snapshot.
3. Accidentally moving trading/runtime authority into startup metadata.
4. Making stale state look fresh by auto-refreshing too aggressively.
