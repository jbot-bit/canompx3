## Iteration: 104
## Target: research/research_mgc_mnq_correlation.py:158,309,433
## Finding: 3 ruff violations — F541 (f-string no placeholders) + 2x B905 (zip without strict=)
## Classification: [mechanical]
## Blast Radius: 1 file, 0 external callers (standalone research script)
## Invariants: behavior unchanged; zip semantics same (strict=False matches current same-length behavior); f-string removal is cosmetic only
## Diff estimate: 3 lines
