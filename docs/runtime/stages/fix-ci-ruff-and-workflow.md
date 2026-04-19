---
mode: IMPLEMENTATION
slug: fix-ci-ruff-and-workflow
stage: 1/1
started: 2026-04-19
task: "Make CI ruff format + lint actually pass — workflow ui/ fix + 84 lint errors cleared. Honest gate, not band-aid."
---

# Stage 1 — CI integrity restoration

## Context
PRs #10, #11, #8 all CI-fail because:
1. Workflow runs `uv run ruff format --check pipeline/ trading_app/ ui/ scripts/ tests/`
   but `ui/` directory does not exist → CI exits 1 unconditionally
2. Even after format passes (PR #9 sweep), ruff lint check at workflow line 31
   fails with 84 errors. Pre-existing on main.

PR #9 ("ruff format sweep — unblocks CI") only fixes formatting. Workflow
ui/ reference and lint debt are unaddressed.

## Why fix all on this branch (PR #9 expansion)
PR #9's stated purpose is "unblocks CI". A format-only sweep doesn't
achieve that — workflow ui/ + lint also block. Fixing all three on
PR #9 makes the PR atomically self-sufficient with its stated goal.

Per institutional-rigor.md: "Do not band-aid. Do not skip. Take the
proper long-term institutional-grounded fix." Adding `continue-on-error`
to lint hides the debt, which is the band-aid this rule forbids.

Per CLAUDE.md "no silent failures": a CI gate that's always red is
worse than no gate — it teaches everyone to ignore failures. The 2
F821s in account_survival.py would normally be "real bugs" but
nobody saw them because lint was always red.

## Approach
1. **Workflow fix** — remove phantom `ui/` from both ruff invocations.
2. **Auto-fix 60 errors** via `ruff check --fix` (mechanical, safe):
   - 40 F541 f-string-missing-placeholders
   - 15 I001 unsorted-imports
   - 3 F401 unused-import (and 2 hidden in --fix)
   - 1 UP035 deprecated-import
3. **Manual fix 24 errors:**
   - 2 F821 undefined-name `AccountProfile` in account_survival.py — add import
   - 8 E712 true-false comparison — apply `--unsafe-fixes` after spot-check
   - 6 B905 zip without explicit strict — add `strict=False` (preserves behavior)
   - 5 F841 unused variables — auto-fixable, in the 60
   - 4 E741 ambiguous variable name (l, I, O) — rename per call site
4. **Verify** — `ruff check pipeline/ trading_app/ scripts/` exits 0;
   `ruff format --check pipeline/ trading_app/ scripts/ tests/` exits 0;
   pytest passes (catch any regressions from --unsafe-fixes / renames).

## Scope Lock
- .github/workflows/ci.yml
- trading_app/account_survival.py
- trading_app/lane_correlation.py
- scripts/migrations/backfill_validated_trade_windows.py
- scripts/research/gc_proxy_validity.py
- scripts/research/gc_to_mgc_cross_validation.py
- scripts/research/portfolio_correlation_audit.py
- scripts/research/wave4_feature_audit.py
- scripts/research/wave4_presession_t2t8.py
- scripts/research/wave4_presession_t2t8_v2.py
- scripts/tmp/directional_context_alignment.py
- scripts/tmp/lane_analysis.py

## Acceptance criteria
1. `uv run ruff format --check pipeline/ trading_app/ scripts/ tests/` → exit 0
2. `uv run ruff check pipeline/ trading_app/ scripts/` → exit 0
3. `uv run pytest tests/ -q` → no new failures
4. All Edits within current ruff scope only — no production logic changes
5. PR #9 title/description updated to reflect expanded scope

## Blast Radius
- workflow change: trivial (1 path removed)
- format/lint auto-fixes: cosmetic (whitespace, import order, f-string prefixes)
- F821 fix: adds 1 import, no behavior change (annotations are strings under
  `from __future__ import annotations`)
- E712 unsafe-fix: `x == True` → `x is True` — semantic equivalent for normal
  bool, theoretically affects `__eq__`-overriding mocks (none in trading code)
- B905 strict=False: adds explicit kwarg matching existing implicit behavior
- E741 rename: 4 single-letter variables → descriptive names; trace impact
  per file
