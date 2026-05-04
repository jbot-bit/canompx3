# Documentation Index (Active Entrypoints)

Use these as the only daily entrypoints. Everything else in `docs/` is supporting detail, historical context, or generated output.

1. [`docs/governance/document_authority.md`](./governance/document_authority.md) — binding authority model and document lifecycle policy.
2. [`CLAUDE.md`](../CLAUDE.md) — canonical repository operating contract and guardrails.
3. [`TRADING_RULES.md`](../TRADING_RULES.md) — live trading doctrine and deployment semantics.
4. [`RESEARCH_RULES.md`](../RESEARCH_RULES.md) — research methodology and statistical standards.
5. [`HANDOFF.md`](../HANDOFF.md) — current cross-tool baton and immediate session state.
6. [`docs/runtime/decision-ledger.md`](./runtime/decision-ledger.md) — durable runtime decisions.
7. [`docs/runtime/action-queue.yaml`](./runtime/action-queue.yaml) — active operational queue and priorities.
8. [`docs/plans/active/`](./plans/active/) — date-bounded active design plans only.
9. [`docs/governance/system_authority_map.md`](./governance/system_authority_map.md) — full authority/context map.

## Usage rules

- Treat this index as the front door for day-to-day work.
- New routine workflows should link back to one of the entrypoints above instead of creating new top-level authority docs.
- Keep `docs/plans/active/` intentionally small; archive completed or inactive items under `docs/plans/archive/`.

## Claude Code daily flow (project-specific)

1. Read `HANDOFF.md`, then `CLAUDE.md`.
2. Use this index to route to the right canonical surface.
3. Work from `docs/plans/active/` only for current design execution.
4. Run `python3 scripts/tools/list_stale_active_docs.py --threshold-days 14` during routine maintenance.
