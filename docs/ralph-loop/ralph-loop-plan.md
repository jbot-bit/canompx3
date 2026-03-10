## Iteration: 17
## Phase: implement (batch — T2 + T3, same function, same blast radius)
## Target: scripts/tools/generate_trade_sheet.py:200-226 (_load_best_by_expr query)
## Finding: T2: LEFT JOIN family_rr_locks with IS NULL fallback diverges from live_config's INNER JOIN — could show unlocked RR variants. T3: query missing vs.orb_minutes, aperture parsed from strategy_id string instead.
## Decision: implement (batch — both touch same query, zero blast radius overlap)
## Rationale: Aligns trade sheet query with live_config pattern. T2 dormant (all families have locks) but wrong code. T3 eliminates fragile string parsing. Both changes in _load_best_by_expr, called only from collect_trades (same file).
## Blast Radius: _load_best_by_expr → collect_trades → generate_html (same file). _parse_aperture becomes dead code after T3.
## Diff estimate: ~8 lines changed in query + 3 lines in collect_trades to use orb_minutes
