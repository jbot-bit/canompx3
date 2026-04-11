# Published DB Contracts For Validated Shelf

Date: 2026-04-11

## Problem

The repo had improved deployable-shelf semantics, but the contract still lived
mostly as caller knowledge:

- helpers in `trading_app/validated_shelf.py`
- repeated `WHERE` predicates in readers
- no first-class published relation that cross-layer code could depend on

That left the system vulnerable to the same failure mode in new clothes:
policy drift through scattered SQL and "everyone knows what active means."

## Decision

Promote validated-shelf semantics into a published cross-layer DB contract:

1. Canonical DB views
   - `active_validated_setups`
   - `deployable_validated_setups`
2. Canonical neutral contract module
   - `pipeline/db_contracts.py`
3. Runtime helpers return relations, not just predicates
   - readers can `FROM` the published shelf directly
4. Pipeline code consumes the published DB contract, not `trading_app/`
   runtime helpers

## Why this shape

This follows established architecture practice more closely than ad hoc helper
reuse:

- Martin Fowler's `Repository` / `Gateway` guidance: data-access semantics
  should live behind a dedicated boundary, not be reconstructed by every
  caller.
- Fowler's `Published Interface` idea: shared contracts need an explicit
  surface other parts of the system can depend on.
- DuckDB gives the correct storage primitive for that contract:
  `CREATE OR REPLACE VIEW`.

This is lower blast radius than splitting tables while still giving the codebase
an actual box to put the cards in.

## Canonical surfaces

- `pipeline/db_contracts.py`
  - neutral, cross-layer DB contract names and relation helpers
- `trading_app/validated_shelf.py`
  - trading-app lifecycle semantics and re-export of the published shelf
    contract
- `trading_app/db_manager.py`
  - responsible for publishing / refreshing the canonical views

## Rules

1. Pipeline code may read published DB contracts, but may not import runtime
   helpers from `trading_app/`.
2. New live-adjacent readers should prefer relation helpers over rebuilding
   validated-shelf predicates.
3. Schema initialization must publish shelf views only after all
   `validated_setups` migrations complete.
4. Force rebuilds must drop dependent views before dropping underlying tables.

## Blast radius

Low:
- `pipeline/db_contracts.py`
- published view creation / refresh
- generic read helpers
- drift/test hardening

Medium:
- readers moving from raw table + `WHERE` predicate to canonical relation

Not done here:
- full repo-wide migration of every historical / research reader
- table split between deployable and research validated rows
- DB constraint layer beyond published views

## Verification bar

- targeted schema/helper/reader tests
- drift
- self-review for:
  - stale view publication after later `ALTER TABLE`s
  - force rebuild ordering with dependent views
  - one-way dependency preservation (`pipeline/` must not import `trading_app/`)

## Sources

- Martin Fowler, `Repository`
  - https://martinfowler.com/eaaCatalog/repository.html
- Martin Fowler, `Gateway`
  - https://martinfowler.com/articles/gateway-pattern.html
- Martin Fowler, `Published Interface`
  - https://martinfowler.com/bliki/PublishedInterface.html
- DuckDB docs, `CREATE VIEW`
  - https://duckdb.org/docs/stable/sql/statements/create_view.html
