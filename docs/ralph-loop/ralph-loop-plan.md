## Iteration: 122
## Target: trading_app/live/projectx/order_router.py:110
## Finding: cancel() does not check response body for success=False — ProjectX API can return {"success": false, "errorMessage": "..."} on HTTP 200, silently appearing to succeed (unlike submit() which checks data.get("success"))
## Classification: [judgment]
## Blast Radius: 1 production file, 1 test file; session_orchestrator._cancel_brackets() wraps cancel() in exception handler — raising RuntimeError on success=False will be caught and logged as "Bracket cancel failed"
## Invariants:
##   1. raise_for_status() must remain as the primary HTTP error guard
##   2. cancel() returns None — signature must not change
##   3. Must raise RuntimeError on success=False (same pattern as submit())
##   4. Happy path (success=True or missing success field) must not change behaviour
## Diff estimate: 5 lines (1 variable assignment + 2-line guard + 1 log)
