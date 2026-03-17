## Iteration: 127
## Target: trading_app/live/broker_factory.py:89-90
## Finding: VALID_BROKERS is defined as "canonical source for dispatcher" but used only in the ValueError message string — the actual dispatch guard is the if/elif chain. These can drift (e.g. adding broker to one but not the other). Fix: add early guard using VALID_BROKERS before if/elif; remove redundant else clause.
## Classification: [mechanical] — coherence fix, no behavior change (same ValueError, same message)
## Blast Radius: 1 production file, 1 test file (test_broker_factory.py). 3 callers unaffected (no signature change).
## Invariants:
## 1. create_broker_components raises ValueError for unknown brokers (fail-closed preserved)
## 2. Auth is an instance; feed/router/contracts/positions are classes — dict shape unchanged
## 3. Valid broker strings "projectx" / "tradovate" behave identically
## Diff estimate: ~5 lines (add 2, remove 3)
