## Iteration: 166
## Target: trading_app/risk_manager.py:321-328
## Finding: RiskManager._warnings accumulated in can_enter() but never logged — silently discarded on daily_reset(). Drawdown and chop warnings provide zero operational value.
## Classification: [mechanical]
## Blast Radius: 1 file changed (risk_manager.py); session_orchestrator.py and paper_trader.py unchanged (no API change)
## Invariants: [1] No behavior change to can_enter() logic; [2] warnings list still maintained for callers that read it; [3] existing tests must pass
## Diff estimate: 4 lines (import logging, log = getLogger, 2 log.warning calls)
