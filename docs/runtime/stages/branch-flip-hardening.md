# Branch-flip hardening

mode: TRIVIAL
task: Add branch-flip guard hook + check_referenced_paths tool (session hardening)

## Scope Lock

- scripts/tools/check_referenced_paths.py

## Blast Radius

New standalone tool script — no imports from pipeline or trading_app. No production logic touched.
