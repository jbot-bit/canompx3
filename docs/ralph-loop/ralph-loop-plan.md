## Iteration: 148
## Target: trading_app/live/rithmic/auth.py:54,202
## Finding: _ensure_connected() and refresh_if_needed() only gate on _connected flag, ignoring _auth_healthy=False state after bridge timeout — reconnect path is bypassed when connection object exists but is functionally broken
## Classification: [judgment]
## Blast Radius: 1 file (auth.py), 3 callers in session_orchestrator.py (no change to callers)
## Invariants:
##   [1] Fast path on fully healthy state (_connected=True AND _auth_healthy=True) must still short-circuit immediately
##   [2] disconnect() behavior unchanged
##   [3] No new imports
## Diff estimate: 2 lines changed
