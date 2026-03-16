## Iteration: 123
## Target: trading_app/live/projectx/data_feed.py:298-302
## Finding: _drain_bar_queue silently crashes on on_bar exception — kills bar delivery in signalrcore path with no log or recovery
## Classification: [judgment]
## Blast Radius: 1 file changed, 2 test files reference, broker_factory.py imports only
## Invariants: [1] Drain loop must continue after single bar failure; [2] Each failure must be logged; [3] CancelledError must propagate (not swallowed) to allow clean task shutdown
## Diff estimate: 4 lines
