## Iteration: 118
## Target: trading_app/portfolio.py:699
## Finding: Hardcoded _COMPRESSION_SESSIONS inside loop body duplicates canonical COMPRESSION_SESSIONS from pipeline.build_daily_features (canonical violation + orphan risk)
## Classification: [mechanical]
## Blast Radius: 1 file (portfolio.py), no signature changes, test file exercises code path
## Invariants: SQL column list must remain ["CME_REOPEN","TOKYO_OPEN","LONDON_METALS"] equivalent; no behavior change
## Diff estimate: 3 lines changed
