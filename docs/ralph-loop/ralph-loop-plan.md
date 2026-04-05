## Iteration: 152
## Target: trading_app/live/trade_journal.py:255-288
## Finding: `incomplete_trades(trading_day=None)` allows calling without a day filter, returning ALL history incomplete trades — the docstring explicitly warns this "would incorrectly restore stale incomplete records from previous days as active positions". The no-filter code path is reachable and dangerous.
## Classification: [judgment]
## Blast Radius: 2 files (trade_journal.py production + test_trade_journal.py tests). All production callers in session_orchestrator.py already pass `trading_day=`. 2 test calls need updating.
## Invariants:
##   1. get_strategy_ids_for_day() remains fail-closed (raises on _con=None) — do not touch
##   2. incomplete_trades() must still raise RuntimeError when _con is None (fail-closed)
##   3. Production callers in session_orchestrator.py already pass trading_day — they must not change
## Diff estimate: 7 lines production (remove else branch + update signature); 2 test lines
## Fix: Remove the `trading_day=None` default and the else branch. Make `trading_day: date` required. Update 2 test calls to pass a date.
