task: Wire Claude Opus 4.8 model bump + honest per-pass effort calibration into trading_app/ai
mode: IMPLEMENTATION

## Scope Lock
- trading_app/ai/claude_client.py
- trading_app/ai/provider_registry.py
- trading_app/ai/query_agent.py
- tests/test_trading_app/test_ai/test_claude_client.py
- tests/test_trading_app/test_ai/test_query_agent.py

## Blast Radius
- claude_client.py — single-source model pin; 4 downstream consumers (query_agent, trading_coach, coaching_digest, lhp/llm_client) inherit via import. Reads: none; Writes: none.
- provider_registry.py — declares reasoning_effort per profile; query_agent reads it via get_profile(). No capital path.
- query_agent.py — Pass 1 (parse, Sonnet, low effort) + Pass 2 (create, Opus, high effort). Read-only research/coaching AI surface; no pipeline/ or trading_app/live/.
- 2 test files — model-pin assertions, stale-id guard, SDK bind-fixtures, new effort-passthrough tests.
- Reads: gold.db (read-only at runtime via QueryAgent). Writes: none. No schema, no capital.
