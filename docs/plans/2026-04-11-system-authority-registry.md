# System Authority Registry And Pulse Identity Backbone

Date: 2026-04-11

## Problem

The repo had started to explain itself better, but the explanation still lived
in too many places:

- `docs/governance/system_authority_map.md` was prose-only
- `pipeline/check_drift.py` duplicated the authority-map expectations
- `scripts/tools/project_pulse.py` reported health, but not the repo's own
  identity or truth surfaces

That still left the project vulnerable to the same old failure mode:
stale folklore copied across docs, audits, and entrypoints.

## Decision

Promote whole-project authority into a code-backed registry:

1. Canonical authority registry
   - `pipeline/system_authority.py`
2. Generated authority-map doc
   - `docs/governance/system_authority_map.md`
   - rendered by `scripts/tools/render_system_authority_map.py`
3. Entrypoint identity surface
   - `scripts/tools/project_pulse.py` exposes repo identity from linked
     sources instead of hardcoded prose
4. Drift enforcement
   - `pipeline/check_drift.py` verifies:
     - authority map stays generated from the registry
     - project pulse keeps reading canonical identity surfaces

## Why this shape

This is stronger than hand-maintained docs or one-off helpers:

- Published Interface:
  shared semantics should have an explicit surface other code can depend on
- CQRS/read-model discipline, used narrowly:
  `project_pulse` is a read model, so it should project identity from
  canonical sources instead of rebuilding meanings locally
- Bounded-context discipline:
  doctrine, registries, command paths, read models, and audits should be
  distinct categories with clear ownership

The important repo-specific rule is still:

**Linked truth, not copied truth.**

## Canonical surfaces

- `pipeline/system_authority.py`
  - code-backed authority registry
- `docs/governance/system_authority_map.md`
  - generated project-level authority map
- `scripts/tools/project_pulse.py`
  - operational/orientation entrypoint that now exposes:
    - canonical repo root
    - canonical DB path
    - active ORB instruments
    - active runtime profiles
    - published shelf relations
    - doctrine docs
    - backbone modules

## Rules

1. Do not hand-edit `docs/governance/system_authority_map.md`; re-render it.
2. If a whole-project truth surface changes, update `pipeline/system_authority.py`
   and regenerate the doc in the same workstream.
3. Entrypoint/orientation tools should consume the registry, not restate the
   repo's structure from memory.
4. Drift should fail if docs or entrypoints drift away from the registry.

## Blast radius

Low:
- generated governance docs
- project orientation/status surfaces
- drift hardening

Medium:
- future consumers that should move onto the authority registry
- future decomposition of `scripts/tools/` by role

Not done here:
- full `scripts/tools/` decomposition
- full repo-wide authority/classification pass for every reader/writer
- schema-level splits between research and deployable validated rows

## Sources

- Martin Fowler, `Published Interface`
  - https://martinfowler.com/bliki/PublishedInterface.html
- Martin Fowler, `CQRS`
  - https://martinfowler.com/bliki/CQRS.html
- Martin Fowler, `Bounded Context`
  - https://martinfowler.com/bliki/BoundedContext.html
- DuckDB docs, `CREATE VIEW`
  - https://duckdb.org/docs/stable/sql/statements/create_view.html
