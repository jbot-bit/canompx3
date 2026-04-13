# Task Routes

Generated canonical task routes for cold-start orientation.

## `completion_claim`

- Purpose: Verify whether a change is actually done before closing it.
- Verification profile: `done`
- Briefing contract: `mutating_briefing`
- Packs: `coding_runtime_pack`
- Doctrine: `CLAUDE.md`, `docs/governance/system_authority_map.md`
- Canonical owners: `pipeline/check_drift.py`, `scripts/tools/session_preflight.py`, `scripts/tools/project_pulse.py`
- Live views: `verification_context`, `system_brief`

## `docs_drift_audit`

- Purpose: Audit whether docs and generated context contracts are stale against code-backed truth.
- Verification profile: `investigation`
- Briefing contract: `orientation_briefing`
- Packs: `coding_runtime_pack`, `project_orientation_pack`
- Doctrine: `CLAUDE.md`, `docs/governance/document_authority.md`, `docs/governance/system_authority_map.md`
- Canonical owners: `pipeline/system_authority.py`, `scripts/tools/render_context_catalog.py`, `pipeline/check_drift.py`
- Live views: `verification_context`, `system_brief`

## `live_trading_status`

- Purpose: Understand the current runtime/deployment state without loading research doctrine.
- Verification profile: `runtime_status`
- Briefing contract: `orientation_briefing`
- Packs: `coding_runtime_pack`, `trading_runtime_pack`
- Doctrine: `TRADING_RULES.md`, `CLAUDE.md`
- Canonical owners: `trading_app/prop_profiles.py`, `trading_app/lifecycle_state.py`, `pipeline/db_contracts.py`
- Live views: `trading_context`, `system_brief`

## `research_investigation`

- Purpose: Investigate performance changes, edge behavior, or discovery questions using canonical research truth.
- Verification profile: `investigation`
- Briefing contract: `investigation_briefing`
- Packs: `coding_runtime_pack`, `trading_runtime_pack`, `research_methodology_pack`
- Doctrine: `RESEARCH_RULES.md`, `TRADING_RULES.md`, `docs/STRATEGY_BLUEPRINT.md`, `docs/institutional/pre_registered_criteria.md`
- Canonical owners: `trading_app/holdout_policy.py`, `pipeline/asset_configs.py`, `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/strategy_fitness.py`, `pipeline/db_contracts.py`
- Live views: `gold_db_mcp`, `research_context`, `recent_performance_context`, `system_brief`

## `system_orientation`

- Purpose: Return the smallest complete startup model for the repo or current workstream.
- Verification profile: `orientation`
- Briefing contract: `orientation_briefing`
- Packs: `coding_runtime_pack`, `trading_runtime_pack`, `project_orientation_pack`
- Doctrine: `AGENTS.md`, `HANDOFF.md`, `CLAUDE.md`, `CODEX.md`, `docs/governance/document_authority.md`, `docs/governance/system_authority_map.md`
- Canonical owners: `pipeline/system_authority.py`, `pipeline/system_context.py`, `pipeline/work_capsule.py`, `pipeline/system_brief.py`, `context/registry.py`, `context/institutional.py`
- Live views: `system_brief`

## Fallback Read Set

- `AGENTS.md`
- `HANDOFF.md`
- `CLAUDE.md`
- `CODEX.md`
- `docs/governance/document_authority.md`
- `docs/governance/system_authority_map.md`