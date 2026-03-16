## Iteration: 121
## Target: trading_app/live/projectx/auth.py:84
## Finding: `except Exception:` in `_validate_or_login` is broader than intended — should be `except requests.RequestException:` to catch only HTTP/network errors, not programming errors that indicate real bugs
## Classification: [judgment]
## Blast Radius: 1 file changed; 3 callers (broker_factory.py, order_router.py, fetch_broker_fills.py) unaffected — they call higher-level get_token()/headers(); 2 test files (test_projectx_auth.py, test_broker_factory.py)
## Invariants:
##   1. Fallback to _login() must still occur on genuine network/HTTP failures
##   2. Programming errors (AttributeError, TypeError from JSON parse) must NOT be silently swallowed
##   3. Exception is still narrower than BaseException — KeyboardInterrupt etc. are unaffected
## Diff estimate: 1 line changed (except clause)
