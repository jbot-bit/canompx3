## Iteration: 22
## Phase: implement (batch — 3 mechanical fixes)
## Target: contract_resolver.py:40, strategy_fitness.py:124, portfolio.py:953
## Finding: CR1: account ID `or` falsy-zero (same class as OR1). F3a: Sharpe decay threshold -0.1 not extracted. F3b: trade frequency 0.4 not extracted. Also closes iter 9 PRODUCT_MAP finding (has fallback, not a gate).
## Decision: implement
## Rationale: CR1 is same antipattern as iter 21 (OR1) — mechanical `is None` fix. F3a/F3b are named constant extractions. All 3 changes are 1-2 lines, no logic change, zero blast radius overlap.
## Blast Radius: CR1: resolve_account_id() → session_orchestrator (caller). F3a: classify_fitness() used by compute_fitness (same file) → live_config, portfolio, trade sheet. F3b: estimate_daily_capital() → portfolio dashboard. All callers unaffected by constant extraction.
## Diff estimate: ~8 lines changed
