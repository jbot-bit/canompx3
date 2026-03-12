## Iteration: 32
## Phase: implement
## Target: trading_app/mcp_server.py:213 + lines 54-55
## Finding: MCP1 — hardcoded "735 FDR-validated" and instrument data years in MCP instructions (volatile data violation). MCP2 — unused _CORE_MIN/_REGIME_MIN aliases.
## Decision: implement (batch — same file)
## Rationale: Volatile data rule violation (CLAUDE.md). Strategy count goes stale after every rebuild. Unused aliases are dead code.
## Blast Radius:
  - Callers: _build_server called only from __main__. _CORE_MIN/_REGIME_MIN called by nobody.
  - Callees: ACTIVE_ORB_INSTRUMENTS (39 files already import — no new dependency)
  - Tests: tests/test_trading_app/test_mcp_server.py (17 tests — no assertions on instructions string)
  - Drift checks: none reference mcp_server
## Invariants (MUST NOT change):
  1. MCP tool signatures (list_available_queries, query_trading_db, get_strategy_fitness, get_canonical_context) must remain identical
  2. _query_trading_db guardrails (G1-G5) must remain intact
  3. _generate_warnings must still delegate to config.generate_strategy_warnings
## Diff estimate: ~8 lines changed (remove 2 unused aliases, rewrite instructions string with canonical import)
