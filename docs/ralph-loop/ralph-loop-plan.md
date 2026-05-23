## Iteration: 204
## Target: trading_app/live/bot_state.py:135-149 (write_live_health)
## Finding: write_live_health uses json.dumps(default=str) without _sanitize_for_state, inconsistent with write_state's strict contamination guard
## Classification: [judgment]
## Blast Radius: 1 production file (bot_state.py), 1 test file (test_bot_state_strict_types.py), test_bot_dashboard.py verified to still pass
## Invariants:
##   1. write_live_health must remain fail-open for disk errors (operator surface only)
##   2. Existing round-trip tests (test_bot_dashboard.py) must still pass
##   3. snapshot_ts_utc added BEFORE sanitization must be included
## Diff estimate: 8 lines
## Doctrine cited: integrity-guardian.md § 3 (Fail-Closed), § 6 (No silent failures); ALERT-CONTAM-N2 class pattern (n=2 canonical fix applied to sibling function)
