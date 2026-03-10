## Iteration: 20
## Phase: implement
## Target: trading_app/strategy_discovery.py:630,634
## Finding: SD1 — `median_risk_dollars` and `avg_risk_dollars` include `total_friction`, inflating stored values. Same error class as trade sheet T5 (iter 18).
## Decision: implement
## Rationale: Informational columns only — no gate uses them — but conceptually wrong and could mislead. Same friction-inflation pattern we fixed in generate_trade_sheet.py. Minimal risk (2 lines, no callers use these stored values for decisions).
## Blast Radius: compute_metrics() → all_strategies list → batch_df → experimental_strategies table. Downstream consumers read these columns for display only (MCP templates, reports). No gate decision depends on median_risk_dollars or avg_risk_dollars.
## Diff estimate: 2 lines changed
