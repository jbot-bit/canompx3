# Specs & Audit Prompts

## Feature Specs — ALWAYS Check First
Before implementing ANY feature or building anything new, check `docs/specs/` for an existing spec.
If a spec exists for the work you're about to do, **follow it exactly** — it defines scope, files touched, and what NOT to change. Do not deviate.

## System Audit — Periodic Full-System Check
The system audit (`docs/prompts/SYSTEM_AUDIT.md`) is a comprehensive 11-phase integrity check covering docs, config, database, build chain, live trading, tests, research, and git/CI.

**When to suggest or run the system audit:**
- User asks for "health check," "system audit," "full check," or "is everything in sync?"
- More than 2 weeks since the last audit (check `research/output/` for `HIGH_LEVEL_AUDIT_*.md`)
- A multi-file refactor or schema change just completed
- User says something feels "off," "broken," or "out of sync"
- Before a major release or deployment change

**Run modes:**
- **Quick** (~30 min): Phase 0B triage → Phase 1 automated → Phase 6 build chain → Phase 3A numbers only
- **Standard** (~2-3 hours): All phases, focused on recently changed files
- **Deep** (~half day): All phases plus ENTRY_MODEL_GUARDIAN + PIPELINE_DATA_GUARDIAN

**How to run:** Read `docs/prompts/SYSTEM_AUDIT.md` and execute phases in order. Phase 0 self-identifies existing checks to avoid duplication.

## Guardian Prompts — Before Significant Changes
Before making significant changes to production logic, read the relevant guardian prompt:

- **Entry model changes** (outcome_builder, strategy_discovery, strategy_validator, config.py entry model enums, drift checks referencing entry models):
  → Read `docs/prompts/ENTRY_MODEL_GUARDIAN.md` and run Pass 1 Discovery before modifying code

- **Pipeline data changes** (ingest_dbn, build_bars_5m, build_daily_features, init_db schema, dst.py session logic, strategy_validator batch writes, build_edge_families):
  → Read `docs/prompts/PIPELINE_DATA_GUARDIAN.md` and run Pass 1 Discovery before modifying code

- **Any change touching both** (e.g., adding a new entry model that requires schema changes):
  → Read BOTH guardian prompts. Run both Pass 1 discoveries.

## Escalation Between Prompts
- System audit finds entry model issues → run ENTRY_MODEL_GUARDIAN
- System audit finds pipeline data issues → run PIPELINE_DATA_GUARDIAN
- Guardian prompts can trigger system audit if systemic drift is suspected

## What Counts as "Significant"
- Adding, removing, or modifying entry models or their parameters
- Changing database schema (tables, columns, constraints)
- Modifying pipeline data flow (ingestion, aggregation, feature computation)
- Changing strategy lifecycle logic (discovery, validation, purging)
- Altering drift check definitions or thresholds
- Any change that affects how data moves from raw bars to validated strategies

## When NOT to Use Audit Prompts
- Bug fixes that don't change definitions or data flow
- Documentation-only changes
- Test additions that don't modify production logic
- Running existing scripts with existing parameters
