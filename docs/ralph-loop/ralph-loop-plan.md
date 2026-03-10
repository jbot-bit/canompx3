## Iteration: 13
## Phase: implement
## Target: trading_app/live_config.py:330-332
## Finding: N1 — Dollar gate fail-open when median_risk_points is NULL. Strategy enters live portfolio without cost-adequacy screening. Returns True silently instead of failing closed.
## Decision: implement
## Rationale: Single file, clear fail-closed fix (return False + logger.warning). The dollar gate was added specifically for MNQ TOKYO/BRISBANE thin-margin cases. NULL risk points = unknown cost adequacy = should BLOCK, not PASS.
## Blast Radius: _check_dollar_gate() called by build_live_portfolio(). Changing return True→False means strategies with NULL median_risk_points will be EXCLUDED instead of silently included. This is the correct behavior — unknown cost adequacy should block.
## Estimated diff: ~3 lines changed

## Implementation Report
- Lines changed: 7 insertions, 2 deletions (live_config.py) + 3 lines test update
- Added `import logging` + `log = logging.getLogger(__name__)` at module top
- Changed `return True, "dollar gate skipped"` → `return False, "dollar gate blocked"` with `log.warning()`
- Updated companion test: `test_none_guard_passes` → `test_none_guard_blocks` (assert False, "blocked")

## Verification Verdict: ACCEPT
- Gate 1 (drift): PASS — 71 checks
- Gate 2 (behavioral): PASS — 6 checks
- Gate 3 (tests): PASS — 20/20 live_config
- Gate 4 (lint): PASS (auto-fixed import order)
- Gate 5 (blast radius): PASS — 2 callers (lines 456, 576) both handle `passes=False` correctly
- Gate 6 (regression): PASS — warning logged, fail-closed confirmed
