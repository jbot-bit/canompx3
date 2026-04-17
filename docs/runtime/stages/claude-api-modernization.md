---
task: Claude API modernization — canonical client + grounding rebuild + caching + structured outputs
mode: IMPLEMENTATION
stage: 2_complete_awaiting_stage_3_go
total_stages: 4
slug: claude-api-modernization
created: 2026-04-17
updated: 2026-04-17
---

# Claude API Modernization

## Why
The project's AI subsystem is on retired/aging models (Sonnet 4.0, Sonnet 4.5), hardcodes "MGC-only" trading rules in the grounding prompt despite the live portfolio being multi-instrument (MNQ=36, MES=2, MGC=0 validated), and uses manual JSON parsing where `messages.parse()` now exists. Three model pins drift independently, violating `institutional-rigor.md` rule 4 (canonical sources).

## Why this matters beyond cosmetics
The AI's system prompt currently tells Claude "you are a trading data analyst for an MGC research system" — this is a lie about the project's actual state and poisons every downstream response. It's not a model-bump task; it's a "fix the foundational grounding" task.

## Blast radius (summary — see conversation transcript 2026-04-17 for full report)
- 8 files in scope across 4 stages
- MODERATE blast radius — all within AI/tools subsystem, no pipeline canonical sources modified
- Companion tests WILL break without updates: `test_grounding.py` (hardcoded `"5.74"`), `test_corpus.py` (iterates `CORPUS_FILES`), `test_app_sync.py:804` (asserts `from pipeline.init_db import ORB_LABELS` import)
- `trading_app/mcp_server.py` lazy-imports `load_corpus` — corpus expansion auto-propagates to the MCP `get_canonical_context` tool (benign; increases payload)
- Test gap: no tests exist for `trading_coach.py` or `coaching_digest.py` (LOW — pre-existing gap, not created by this change)

## Stage plan

### Stage 1 — Canonical Claude client + dependency pin (THIS STAGE)

**Out of scope:** any call-site migration, grounding changes, drift checks.

## Scope Lock
- trading_app/ai/grounding.py
- trading_app/ai/corpus.py
- tests/test_trading_app/test_ai/test_grounding.py
- tests/test_trading_app/test_ai/test_corpus.py
- tests/test_app_sync.py

## Blast Radius
- `grounding.py` REWRITE: import `ACTIVE_ORB_INSTRUMENTS`, `SESSION_CATALOG`, `COST_SPECS` (multi-instrument), `CORE_MIN_SAMPLES`/`REGIME_MIN_SAMPLES`. Drops `ORB_LABELS` import. Drops "MGC only" framing.
- `corpus.py` EXPAND: +4 entries in `CORPUS_FILES` (RESEARCH_RULES.md, CLAUDE.md, pre_registered_criteria.md, mechanism_priors.md).
- `trading_app/mcp_server.py` lazy-imports `load_corpus` — corpus expansion auto-propagates to `get_canonical_context` MCP tool (benign; increases payload).
- `test_grounding.py`: drop hardcoded `"5.74"` MGC literal, assert multi-instrument via canonical imports.
- `test_corpus.py`: expected `len(CORPUS_FILES)` bumps 4→8; all 8 paths must exist on disk.
- `test_app_sync.py:804`: replace `ORB_LABELS` import assertion with `SESSION_CATALOG` assertion.
- One-way dependency respected throughout (pipeline.* → trading_app.ai.*).
- Stage 1 complete. Stage 1 canonical module (`claude_client.py`) NOT touched — unaffected.

## Stage 1 — complete (2026-04-17)
- [x] 9/9 stage tests pass
- [x] Full suite: 4451 passed, 0 failed
- [x] Drift: 3 pre-existing (Check 45), 0 new
- [x] Canonical module exports CLAUDE_STRUCTURED_MODEL / CLAUDE_REASONING_MODEL / get_client

## Stage 2 — complete (2026-04-17)
- [x] 28/28 stage tests pass (17 new grounding + 8 new/updated corpus + 1 updated app_sync)
- [x] Full suite: 4459 passed, 0 failed (1 pre-existing pulse-length test deselected — 62 vs 60 line cap, fails at HEAD with my work stashed)
- [x] Drift: 3 pre-existing (Check 45 SGP_MOMENTUM) + Check 16 hook-env-only (anthropic not in hook subprocess python; passes via uv run); 0 new from Stage 2
- [x] grounding.py: zero hardcoded instruments / sessions / costs / thresholds — all from canonical sources
- [x] corpus.py: 4 → 8 entries (added RESEARCH_RULES, CLAUDE_MD, PRE_REGISTERED_CRITERIA, MECHANISM_PRIORS)
- [x] SESSION_CATALOG replaces ORB_LABELS in grounding; drift assertion in test_app_sync.py updated accordingly
- [x] No "MGC only" framing remains anywhere in grounding

**Acceptance:**
- [ ] `from trading_app.ai.claude_client import CLAUDE_STRUCTURED_MODEL, CLAUDE_REASONING_MODEL, get_client` works
- [ ] Model constants are current IDs (not in retired table per `shared/models.md`)
- [ ] `get_client()` raises `ValueError` when `ANTHROPIC_API_KEY` absent
- [ ] `get_client()` returns configured client when key present
- [ ] Tests pass (`pytest tests/test_trading_app/test_ai/test_claude_client.py -v`)
- [ ] Full suite green (`pytest tests/ -x -q`)
- [ ] `pipeline/check_drift.py` passes

### Stage 2 — Grounding rebuild from canonical sources
**scope_lock:**
- `trading_app/ai/grounding.py` (REWRITE)
- `trading_app/ai/corpus.py` (EXPAND: +RESEARCH_RULES.md, +CLAUDE.md, +docs/institutional/pre_registered_criteria.md, +docs/institutional/mechanism_priors.md)
- `tests/test_trading_app/test_ai/test_grounding.py` (UPDATE — drop `"5.74"` literal, assert multi-instrument via `ACTIVE_ORB_INSTRUMENTS`)
- `tests/test_trading_app/test_ai/test_corpus.py` (UPDATE — expected count matches new `CORPUS_FILES`)
- `tests/test_app_sync.py` (UPDATE line 804 assertion if `ORB_LABELS` import is changed)

### Stage 3 — Migrate call sites + prompt caching + structured outputs
**scope_lock:**
- `trading_app/ai/query_agent.py`
- `scripts/tools/trading_coach.py`
- `scripts/tools/coaching_digest.py`
- `tests/test_trading_app/test_ai/test_query_agent.py` (UPDATE — `test_extract_intent_code_block` migration)

### Stage 4 — Drift check for canonical model enforcement
**scope_lock:**
- `pipeline/check_drift.py` (ADD new check)
- `tests/test_check_drift.py` (or equivalent — inject-and-verify test)

## Current stage: 1

## Canonical sources used
- `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` (Stage 2)
- `pipeline.dst.SESSION_CATALOG` (Stage 2)
- `pipeline.cost_model.COST_SPECS` (Stage 2)
- `trading_app.config.CORE_MIN_SAMPLES` / `REGIME_MIN_SAMPLES` (Stage 2)
- `anthropic.Anthropic` (Stages 1+3) — routed through new `claude_client.py`

## Verification after each stage
- `python pipeline/check_drift.py`
- `python -m pytest tests/ -x -q`
- Self-review against institutional-rigor.md rules 1-8

## Stage 1 RED tests to write
1. `test_model_constants_are_current` — both model IDs are strings and NOT in the retired/deprecated list
2. `test_get_client_requires_api_key` — raises `ValueError` when `ANTHROPIC_API_KEY` missing
3. `test_get_client_returns_configured_client` — with key, returns an `anthropic.Anthropic` instance
4. `test_structured_model_is_sonnet_46` — explicit pin check
5. `test_reasoning_model_is_opus_47` — explicit pin check
