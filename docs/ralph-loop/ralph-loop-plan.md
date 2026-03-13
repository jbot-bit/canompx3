## Iteration: 45
## Target: trading_app/execution_engine.py:410-412
## Finding: ARMED/CONFIRMING trades silently discarded at session_end — no log entry, invisible to diagnostics (DF-02)
## Blast Radius: 1 file (execution_engine.py); 0 callers affected — pure logging addition
## Invariants:
1. No TradeEvent emitted for ARMED/CONFIRMING (they never entered — correct behavior preserved)
2. Trade state set to EXITED and appended to completed_trades — unchanged
3. 43 execution engine tests must continue to pass
## Diff estimate: 5 lines added (logger.debug call)
