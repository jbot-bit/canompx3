## Iteration: 155
## Target: trading_app/ai/sql_adapter.py:58
## Finding: VALID_ENTRY_MODELS = {"E1", "E2", "E3"} hardcoded canonical list — should derive from trading_app.config.ENTRY_MODELS
## Classification: [mechanical]
## Blast Radius: 1 file (sql_adapter.py). Callers (grounding.py, query_agent.py, mcp_server.py) import SQLAdapter/templates not VALID_ENTRY_MODELS. phase_4_config_sync.py and check_drift.py import VALID_ENTRY_MODELS but only to compare it with ENTRY_MODELS — after fix, they will always be equal (trivially passing).
## Invariants:
##   1. VALID_ENTRY_MODELS must remain a set (callers use `in` operator)
##   2. _validate_entry_model() behavior must not change for valid inputs
##   3. check_drift.py check_entry_models_sync() must continue to pass
## Diff estimate: 3 lines (add import, change 1 assignment)
## Fix: Import ENTRY_MODELS from trading_app.config; replace hardcoded set with set(ENTRY_MODELS)
