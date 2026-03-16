## Iteration: 103
## Target: research/research_zt_cpi_nfp.py:101
## Finding: zip() without strict= parameter (ruff B905) — consecutive-diff list comprehension
## Classification: [mechanical]
## Blast Radius: 1 file, 0 external callers
## Invariants: zip semantics unchanged (strict=False matches current behavior); no logic change; lists are deliberately unequal length (uniq vs uniq[1:])
## Diff estimate: 1 line
