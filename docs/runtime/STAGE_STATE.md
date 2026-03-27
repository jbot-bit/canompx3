---
mode: IMPLEMENTATION
task: "Deep audit: lint, type errors, code smells, deprecations"
stage_purpose: "Fix all ruff errors, pyright errors, deprecation warnings, bare excepts, and code smells across entire codebase"
scope_lock:
  - pipeline/
  - trading_app/
  - ui/
  - scripts/
  - tests/
  - docs/runtime/STAGE_STATE.md
acceptance:
  - "ruff check returns 0 errors across pipeline/ trading_app/ ui/ scripts/ tests/"
  - "All tests pass (batched to avoid OOM)"
  - "75/75 drift checks pass"
  - "Zero bare except (E722) in production code"
  - "Zero fail-open patterns in production code"
  - "build_live_portfolio deprecation warnings eliminated"
updated: 2026-03-27T22:30:00+10:00
prior_stage: "ML Phase 1 (stashed: git stash list)"
terminal: main
---
