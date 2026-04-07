## Iteration: 161
## Target: trading_app/live/tradovate/contracts.py:64
## Finding: resolve_front_month() returns empty string "" when API response lacks both "name" and "contractSymbol" fields — silent empty symbol passed to order router
## Classification: [judgment]
## Blast Radius: 1 production file (contracts.py), 1 test file (test_tradovate.py)
## Invariants:
##   1. resolve_front_month() must raise RuntimeError (not return silently) on empty/bad symbol
##   2. Happy path (API returns "name" key) must be unchanged
##   3. Existing no-contracts RuntimeError path (line 59-60) must be preserved
## Diff estimate: 4 lines production, 15 lines test
