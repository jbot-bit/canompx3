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
The AI subsystem was on retired/aging models (Sonnet 4.0, Sonnet 4.5), hardcoded "MGC-only" trading rules in the grounding prompt despite the live portfolio being multi-instrument (MNQ=36, MES=2), and used manual JSON parsing where `messages.parse()` now exists. Three model pins drifted independently, violating `institutional-rigor.md` rule 4 (canonical sources).

## Why this matters beyond cosmetics
The AI's system prompt told Claude "you are a trading data analyst for an MGC research system" — a lie about the project's actual state that poisoned every downstream response. Not a model-bump task; a foundational-grounding fix.

## Canonical sources used
- `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` (Stage 2)
- `pipeline.dst.SESSION_CATALOG` (Stage 2)
- `pipeline.cost_model.COST_SPECS` (Stage 2)
- `trading_app.config.CORE_MIN_SAMPLES` / `REGIME_MIN_SAMPLES` (Stage 2)
- `anthropic.Anthropic` — routed through `trading_app/ai/claude_client.py` (Stages 1+3)

## Verification after each stage
- `uv run python pipeline/check_drift.py`
- `uv run python -m pytest tests/ -q`
- Self-review against institutional-rigor.md rules 1-8

---

## Stage 1 — complete (2026-04-17, commit f219ef15)
- [x] 9/9 stage tests pass
- [x] Full suite: 4451 passed, 0 failed
- [x] Drift: 3 pre-existing (Check 45 SGP_MOMENTUM trade-window staleness), 0 new
- [x] Canonical module exports CLAUDE_STRUCTURED_MODEL / CLAUDE_REASONING_MODEL / get_client

## Stage 2 — complete (2026-04-17)
- [x] 28/28 stage tests pass (17 new grounding + 8 new/updated corpus + 1 updated app_sync)
- [x] Full suite: 4459 passed, 0 failed (1 pre-existing pulse-length test deselected — 62 vs 60 line cap, fails at HEAD with Stage 2 work stashed, verified pre-existing)
- [x] Drift: 3 pre-existing (Check 45 SGP_MOMENTUM) + Check 16 hook-env-only (anthropic not in hook subprocess python; passes via uv run); 0 new from Stage 2
- [x] `grounding.py`: zero hardcoded instruments / sessions / costs / thresholds — all from canonical sources
- [x] `corpus.py`: 4 → 8 entries (added RESEARCH_RULES, CLAUDE_MD, PRE_REGISTERED_CRITERIA, MECHANISM_PRIORS)
- [x] SESSION_CATALOG replaces ORB_LABELS in grounding; drift assertion in `test_app_sync.py:800-812` updated accordingly
- [x] No "MGC only" framing remains anywhere in grounding

---

## Stage 3 — planned (not yet executed)

### scope_lock
- trading_app/ai/query_agent.py
- scripts/tools/trading_coach.py
- scripts/tools/coaching_digest.py
- tests/test_trading_app/test_ai/test_query_agent.py

### Blast radius
- Three call sites migrate to canonical `claude_client` (Stage 1). Removes hardcoded model strings + direct `anthropic.Anthropic()` constructions + manual ANTHROPIC_API_KEY checks.
- `query_agent.py`: Pass 1 → `messages.parse(output_format=QueryIntentSchema)` on Sonnet 4.6 (drops `temperature=0.0` + markdown-fence recovery). Pass 2 → `messages.create(thinking=adaptive)` on Opus 4.7 with cache_control on grounding prompt.
- `coaching_digest.py`: `messages.parse(output_format=DigestResponseSchema)` on Opus 4.7 + adaptive thinking. Removes `parse_digest_response` regex markdown-fence stripping. Replaces bare `except Exception` with typed Anthropic errors.
- `trading_coach.py`: two call sites migrated to canonical client on Opus 4.7. Adaptive thinking OFF by default (interactive chat latency) — hybrid per user decision.
- Deployment-state files (validated_setups, prop_profiles, live_config) NOT touched.

---

## Stage 4 — planned

### scope_lock
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift.py

### Blast radius
- New drift check enforces `trading_app/ai/claude_client.py` as the sole source of:
  1. Hardcoded Claude model-ID strings (`claude-(opus|sonnet|haiku)-\d` pattern)
  2. Direct `anthropic.Anthropic(` client constructions
- All other .py files under `pipeline/`, `trading_app/`, `scripts/`, `research/` must NOT contain either pattern.
- Injection-and-verify test follows `test_canonical_orb_utc_window_source` pattern.
