---
mode: IMPLEMENTATION
slug: fix-drift-check-96-ast-aware
stage: 1/1
started: 2026-04-19
task: "Drift check 96 (shared profile fingerprint canonical) must use ast-based import check, not literal string match"
---

# Stage 1 — Make check 96 syntax-aware

## Context (grounded)
After CI green-pass on ruff format + lint, drift check 96 fires:

  trading_app/account_survival.py must import build_profile_fingerprint
  from trading_app.derived_state

The file DOES import it (account_survival.py:31-38 contains a multi-line
`from trading_app.derived_state import (...)` block including
`build_profile_fingerprint`). But check 96 at
`pipeline/check_drift.py:5108` does literal-string match:

  if "from trading_app.derived_state import build_profile_fingerprint" not in account_text:

The literal pattern doesn't match the multi-line form even though it's
semantically equivalent. Ruff's I001 auto-fix consolidated the
previously-separate single-line import into the existing multi-line
block during PR #9's sweep — exposing the brittleness.

## Why this matters (project value)
The check's PURPOSE is to enforce canonical-source delegation per
`integrity-guardian.md` rule 2: "delegate to canonical sources, never
re-encode." A brittle string-match check that breaks on style
normalization undermines its own purpose — once it false-fires once,
people start ignoring it. Per `institutional-rigor.md`: "no silent
failures" and "delegate to canonical sources, never re-encode" — the
check must be as robust as the principle it enforces.

## Approach (proper, not band-aid)
Rewrite the import detection to parse Python imports via `ast`:

  import ast
  tree = ast.parse(account_text)
  imports = [
      name.name for node in ast.walk(tree)
      if isinstance(node, ast.ImportFrom)
      and node.module == "trading_app.derived_state"
      for name in node.names
  ]
  if "build_profile_fingerprint" not in imports:
      violations.append(...)

This handles ALL valid Python import forms:
  - `from X import Y`
  - `from X import (Y,)`
  - `from X import (Y, Z, W)`  (multi-line)
  - `from X import Y as Z` (aliased — could be added if relevant)

Rejected alternatives:
  - Restore single-line import form: ruff will re-consolidate on next
    --fix run; treadmill problem.
  - Add `# noqa: I001` to file: hides ruff complaint but doesn't fix the
    underlying brittleness in the check.
  - Skip check 96 (continue-on-error): violates "no silent failures".

## Scope Lock
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift_ws2.py

## Acceptance criteria
1. RED test: assert check 96 PASSES on a multi-line import form (currently fails).
2. GREEN: ast-based detection passes both single-line and multi-line.
3. Full pytest suite passes (no regressions to other check_drift tests).
4. `python pipeline/check_drift.py` locally → check 96 reports PASSED.
5. CI on PR #9 reaches green (this is the only remaining real failure).

## Blast Radius
- pipeline/check_drift.py: edit one function (check_shared_profile_fingerprint_canonical, ~10 lines).
- No callers — drift check is a standalone runner.
- Adds dependency on stdlib `ast` (already imported in check_drift.py? need to verify).
- Test coverage: add 2-3 tests covering single-line, multi-line, and missing-import cases.
