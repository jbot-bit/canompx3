## Iteration: 38
## Target: trading_app/market_state.py:20 + docstring:10
## Finding: MS1+MS2 — Dead `PROJECT_ROOT` assignment (defined but never referenced anywhere in file) — Orphan Risk; module docstring usage example uses relative `Path("gold.db")` instead of canonical `GOLD_DB_PATH` — Canonical violation; identical pattern to CT1 fixed iter 37
## Blast Radius: 2 files (paper_trader.py lazy-import, test_market_state.py) — neither references PROJECT_ROOT; no callee changes
## Invariants:
##   1. MarketState class API unchanged
##   2. ORB_LABELS import + SESSION_ORDER dict unchanged
##   3. No logic changes — delete dead assignment + update docstring only
## Diff estimate: 3 lines (delete PROJECT_ROOT line; update docstring usage example)
