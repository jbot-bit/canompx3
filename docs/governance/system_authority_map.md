# System Authority Map

<!-- Auto-generated from pipeline/system_authority.py via scripts/tools/render_system_authority_map.py -->

## Purpose

This file is the project-level map of where truth lives.

The point is simple: the repo should explain itself without relying on stale
folklore or someone remembering how "Josh's weird machine" works.

If a value or rule changes often, it should be linked to a canonical code or
DB surface, not copied into scattered docs and audits.

Generated from `scripts/tools/render_system_authority_map.py` and
`pipeline/system_authority.py`.

## Design Rule

**Linked truth, not copied truth.**

- Doctrine may live in prose.
- Frequently changing truth must live in code, data, or published read models.
- Audits must verify the linked source, not restate their own local version of the rule.
- Plans and handoffs may explain decisions, but they do not become runtime truth.

## Surface Taxonomy

| Category | Purpose | Canonical examples | Mutation rule |
|---|---|---|---|
| Doctrine | Human-facing binding rules | `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, `docs/institutional/pre_registered_criteria.md`, `docs/governance/document_authority.md` | Update only when policy or workflow changes |
| Canonical registries | Stable code truth for changing facts and rules | `pipeline/system_authority.py`, `pipeline/asset_configs.py`, `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/config.py`, `trading_app/holdout_policy.py`, `trading_app/prop_profiles.py` | One owned source per concept; no duplicate literals downstream |
| Command writers | The only places allowed to mutate durable state | `pipeline/init_db.py`, `trading_app/db_manager.py`, `trading_app/strategy_validator.py`, `trading_app/edge_families.py`, `scripts/migrations/` | Mutations must go through owned command paths |
| Published read models / contracts | Stable query surfaces for operational consumers | `pipeline/db_contracts.py`, `trading_app/validated_shelf.py`, `DB views active_validated_setups and deployable_validated_setups`, `scripts/tools/project_pulse.py` | Readers consume these instead of rebuilding semantics ad hoc |
| Derived operational state | Runtime snapshots and envelopes derived from canonical truth | `trading_app/lifecycle_state.py`, `trading_app/derived_state.py`, `data/state/sr_state.json`, `Criterion 11 survival reports` | Must validate envelope/fingerprint before trust |
| Audit / verification | Checks that linked truth and downstream consumers stay aligned | `pipeline/check_drift.py`, `scripts/audits/`, `scripts/tools/audit_integrity.py`, `scripts/tools/audit_behavioral.py` | Audits must import canonical truth where possible |
| Plans / history / baton | Decision history and in-flight context | `docs/plans/`, `HANDOFF.md`, `ROADMAP.md`, `docs/postmortems/` | Never cited as live runtime truth |
| Reference / generated docs | Orientation aids and generated inventory | `docs/ARCHITECTURE.md`, `docs/MONOREPO_ARCHITECTURE.md`, `REPO_MAP.md`, `docs/governance/system_authority_map.md` | Must be marked non-authoritative and kept linked/generated |

## Canonical Truth Map

| Question | Canonical source |
|---|---|
| Which instruments are active or dead? | `pipeline/asset_configs.py` |
| What are the live session definitions and DOW alignment rules? | `pipeline/dst.py` |
| What are the cost specs? | `pipeline/cost_model.py` |
| What is the sacred holdout policy? | `trading_app/holdout_policy.py` |
| What filters, entry models, and routing rules exist? | `trading_app/config.py` |
| What is deployable on the validated shelf? | `pipeline/db_contracts.py + deployable_validated_setups` |
| What are the active execution lanes? | `trading_app/prop_profiles.py` |
| What is the unified operational block/allow state? | `trading_app/lifecycle_state.py` |
| What is planning vs current implementation? | `ROADMAP.md is planning only; code/DB decide current implementation` |

## Enforcement Rules

1. New mutable workflow surfaces must declare their category here or in docs/governance/document_authority.md in the same change.
2. If a consumer needs deployable shelf semantics, it should read deployable_validated_setups or deployable_validated_relation(...), not validated_setups WHERE status='active'.
3. If a rule changes frequently with data, profiles, or runtime state, do not hardcode it in prose. Link the source or expose a published contract.
4. Audits should fail when they read deprecated truth surfaces after a newer canonical surface exists.
5. Reference docs must say what they are not authoritative for.
