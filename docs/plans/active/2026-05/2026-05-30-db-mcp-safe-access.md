---
status: active
owner: Codex
last_reviewed: 2026-05-30
superseded_by: ""
---

# DB MCP Safe Access Plan

## Decision

`gold-db` remains the canonical database MCP for agents. It is local, stdio,
and read-only by default. Remote, phone-driven, and GitHub-adjacent workflows
must use MCP health/policy/snapshot surfaces instead of direct live `gold.db`
access.

## Implemented V1

- `get_db_health` reports the resolved DB path, read-only open status,
  mtime/size, table horizons, and the explicit access policy.
- `get_db_freshness` reports row counts and latest timestamps/dates for
  approved read-model tables.
- `get_db_snapshot_manifest` lists only valid stamped snapshot manifests.
- `get_db_access_policy` makes the no-write/no-live-GitHub policy explicit.
- `scripts/tools/export_gold_db_snapshot.py` exports approved tables to Parquet
  under the approved snapshot root with a JSON manifest.

## Write Broker Boundary

No V1 MCP tool may write to `gold.db`, switch databases, route orders, mutate
allocation, or write `paper_trades`.

If write automation becomes necessary, build it as a separate single-writer
broker with:

- named jobs only;
- one writer lease;
- dry-run before execution;
- idempotency key;
- audit log;
- lock timeout;
- rollback/restore instructions.

Forbidden future broker jobs: raw SQL, allocation edits, live order routing,
`paper_trades` mutation, and live state writes.

## Operational Rule

GitHub Actions and remote review jobs consume stamped snapshots or summaries.
They do not receive live `gold.db` access.
