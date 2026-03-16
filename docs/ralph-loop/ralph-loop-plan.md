## Iteration: 124
## Target: tests/test_trading_app/test_session_orchestrator.py:729,1015,1048,1074,1100
## Finding: 5 test feed stubs missing `on_stale` keyword arg — TypeError at test runtime since d4fe8cb added on_stale to SessionOrchestrator.run()
## Classification: [mechanical]
## Blast Radius: 1 file (test only), 0 production callers affected
## Invariants: [1] production session_orchestrator.py NOT touched; [2] stub run()/was_stopped logic unchanged; [3] only __init__ signatures extended with on_stale=None
## Diff estimate: 5 lines
