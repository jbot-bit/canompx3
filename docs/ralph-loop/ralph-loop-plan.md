## Iteration: 88
## Target: scripts/run_live_session.py:50
## Finding: Hardcoded checks_total = 5 in _run_preflight() — if a 6th check is added without updating this, checks_passed == checks_total returns False even when all checks pass, silently blocking live sessions.
## Classification: [mechanical]
## Blast Radius: 1 file (run_live_session.py). _run_preflight is private, called once at line 237. No test file covers it.
## Invariants:
## 1. Preflight display format [N/checks_total] must remain unchanged
## 2. Pass/fail logic (checks_passed == checks_total) must remain identical
## 3. Current 5-check structure must continue to work correctly
## Diff estimate: 2 lines (add # NOTE comment to checks_total line)
