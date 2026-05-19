## Iteration: 186
## Target: trading_app/strategy_discovery.py:1155
## Finding: `run_discovery(instrument: str = "MGC", ...)` hardcodes canonical instrument as default
## Classification: [mechanical]
## Blast Radius: 1 production file; 8+ test callers all pass instrument= explicitly — no behavior change
## Invariants: All current callers pass instrument= explicitly; CLI default at L1784 untouched; no SQL changes
## Diff estimate: 4 lines
## Doctrine cited: integrity-guardian.md § 2 (institutional-rigor.md § 10) — never hardcode canonical sources as defaults
