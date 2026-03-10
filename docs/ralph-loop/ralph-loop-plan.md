## Iteration: 18
## Phase: implement (batch — code review findings T5 + T6)
## Target: scripts/tools/generate_trade_sheet.py:121 (_exp_dollars_from_row) + :216 (ORDER BY)
## Finding: T5 (IMPORTANT): _exp_dollars_from_row adds spec.total_friction to 1R base, inflating Exp$ and making dollar gate comparison diverge from live_config. T6 (cosmetic): Missing NULLS LAST in ORDER BY.
## Decision: implement
## Rationale: T5 is a formula error — 1R should be stop distance only (median_risk_pts * point_value), not stop + friction. live_config.py:381 is correct. Inflated Exp$ misleads the user and could let strategies pass the dollar gate in the trade sheet that would fail in live_config. T6 is cosmetic alignment.
## Blast Radius: _exp_dollars_from_row → _passes_dollar_gate + collect_trades (same file only).
## Diff estimate: 2 lines changed
