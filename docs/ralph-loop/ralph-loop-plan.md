## Iteration: 40
## Target: trading_app/execution_engine.py:21,23
## Finding: EE1 — Dead `from pathlib import Path` import and `PROJECT_ROOT` assignment — never referenced anywhere in the file (same orphan pattern as CT1/MS1/RM1 in iters 37-39)
## Blast Radius: 0 callers affected — PROJECT_ROOT is module-level dead code only in this file; other files define their own PROJECT_ROOT independently
## Invariants:
- No functional logic changes — only remove two dead lines (import + assignment)
- Path must not be used anywhere else in the file after removal
- All imports from pathlib remain in other files that actually use them
## Diff estimate: 2 lines removed
