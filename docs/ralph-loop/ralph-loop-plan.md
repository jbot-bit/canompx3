## Iteration: 158
## Target: trading_app/mcp_server.py:136-185 + :253
## Finding 1: Silent-failure — _get_strategy_fitness returns {"strategy_count": 0} with no error for invalid/dead instruments (e.g., M2K), instead of a clear error dict. ACTIVE_ORB_INSTRUMENTS is imported but not used for validation in this path.
## Finding 2 (batch): Stale docstring at line 253 — "entry_model: One of: E1, E2, E3" but E3 is soft-retired (SKIP_ENTRY_MODELS).
## Classification: [judgment] (Finding 1 — new guard); [mechanical] (Finding 2 — doc fix). Combined commit tagged [judgment].
## Blast Radius: 1 production file (mcp_server.py), 1 test file (test_mcp_server.py)
## Invariants:
##   1. Valid instrument calls must behave identically to before
##   2. Error return format must be {"error": "..."} — consistent with _query_trading_db
##   3. No change to compute_fitness or compute_portfolio_fitness signatures
## Diff estimate: ~5 lines production + ~8 lines test
