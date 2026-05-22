## Iteration: 191
## Target: trading_app/strategy_validator.py:2308
## Finding: Hardcoded ["E1", "E2"] list for DSR V[SR] computation instead of deriving from ENTRY_MODELS - SKIP_ENTRY_MODELS
## Classification: [mechanical]
## Blast Radius: 1 file, 0 callers (DSR block is internal to run_validation)
## Invariants: [DSR computation logic unchanged; only the entry model iteration list source changes; behavior identical with current config]
## Diff estimate: 3 lines (1 import line expanded by 2, 1 list literal replaced)
## Doctrine cited: integrity-guardian.md § 2 / institutional-rigor.md § 10 — canonical sources, never inline lists
