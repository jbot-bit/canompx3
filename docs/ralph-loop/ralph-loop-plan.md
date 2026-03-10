## Iteration: 16
## Phase: implement
## Target: scripts/tools/generate_trade_sheet.py:134,140
## Finding: Dollar gate `_passes_dollar_gate` returns True (fail-open) on missing data and cost spec exceptions — diverges from live_config.py which correctly returns False (fail-closed). Trade sheet could show phantom trades.
## Decision: implement
## Rationale: Aligns trade sheet with live_config dollar gate (fixed in iter 13). Both gates must agree — the trade sheet is what the user reads before trading. Dormant today (0 NULL median_risk_points) but wrong code. Fix is 2 lines, same file, no external callers.
## Blast Radius: `_passes_dollar_gate` → `collect_trades` (same file only). No callers outside generate_trade_sheet.py.
## Diff estimate: 4 lines changed (2 return values flipped True→False)
