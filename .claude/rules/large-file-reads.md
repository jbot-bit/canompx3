# Large File Reading Protocol

Read tool limit: 2000 lines/call — files above this are silently truncated.

**Rule:** Before reading files >1800 lines: `wc -l <file>`, then use `offset`+`limit`. Use Grep to find the section first.

**Large files (always chunk-read):** `pipeline/check_drift.py`, `trading_app/config.py`, `trading_app/strategy_discovery.py`, `trading_app/strategy_validator.py`, large test files.
