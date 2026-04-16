# Self-Documenting Context Routing

Date: 2026-04-13

## Goal

Make cold-start agent orientation deterministic, generated, and auditable.

The repo should answer:

- what doctrine applies to task X
- which code/data files are canonical for task X
- which live views should be queried first
- which surfaces are explicitly not live truth
- which verification profile applies before claiming done

## Problem

The previous shape had useful authority docs and live state readers, but no
single deterministic router:

- `CLAUDE.md` remained a practical monolith
- `.claude/rules/*` encoded useful routing, but only for Claude-oriented flows
- live state existed (`system_context`, `project_pulse`) without task scoping
- docs could still duplicate or rephrase changing facts

The main architectural risk was creating a second stale registry while trying
to fix the first one.

## Design decision

Use one code-backed routing registry:

- `context/registry.py`

It owns:

- domain definitions
- task manifests
- verification step definitions
- verification profiles
- live-view definitions

Tasks do **not** own raw file lists directly. They reference domain IDs and
verification profile IDs. The resolver expands those IDs into concrete files
and commands.

Verification profiles do **not** own raw command strings directly. They
reference verification step IDs so command ownership also lives once.

This avoids a second drift surface where every task manifest repeats paths,
doctrine files, and verification rules.

## Anti-drift invariants

1. Generated docs in `docs/context/` come only from `context/registry.py`.
2. The resolver reads the registry; it does not maintain parallel mappings.
3. Tasks reference domain/profile IDs only; shared file ownership lives once.
4. Verification profiles reference step IDs only; shared command ownership
   lives once.
5. Natural-language routing must fail closed on ambiguity instead of guessing.
6. Task examples must continue to resolve to their declared manifest IDs.
7. Generated task views must preserve strict truth-class boundaries:
   - `canonical_state`
   - `live_operational_state`
   - `non_authoritative_context`
8. Generated task views must stay compressed:
   - no freeform recommendations in canonical/live sections
   - no mixed baton/history inside canonical sections
   - no repeated prose padding that restates owned sources
9. `pipeline/check_drift.py` must fail if:
   - the registry references missing files or unknown IDs
   - generated docs drift from the registry
   - `AGENTS.md` stops pointing cold-start agents to the resolver
   - task-view truth boundaries are violated
10. Doctrine remains human-written and binding. Generated docs remain
   orientation-only.

## Migration path

### Phase A

Add:

- routing registry
- resolver CLI
- generated context docs
- drift enforcement

Do not yet refactor `CLAUDE.md` or `.claude/rules/*` heavily.

### Phase B

Refactor startup/orientation docs to point at the new resolver first and move
more task-specific routing out of prose.

### Phase C

Promote more generated context views and shrink stale reference docs.

## Non-goals for this first slice

- no embeddings or vector search
- no attempt to solve every task taxonomy immediately
- no replacement of canonical doctrine docs
- no weakening of existing drift/hook workflow
