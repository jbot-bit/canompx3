---
slug: pass-three-magic-number-drift-check
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-26
updated: 2026-04-27
task: v6.1 Phase 5 Pass Three — magic-number rationale drift check on trading_app/live/. Closes the deferred-from-PR-#121 item by adding the IMPLEMENTATION stage that the guard requires for pipeline/check_drift.py edits. (Re-extracted from abandoned PR #124 onto fresh main 2026-04-27.)
---

# Stage: Pass Three — magic-number rationale drift check

mode: IMPLEMENTATION
date: 2026-04-27 (re-extraction)

scope_lock:
  - pipeline/check_drift.py
  - docs/runtime/stages/pass-three-magic-number-drift-check.md
  - trading_app/live/alert_engine.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/instance_lock.py
  - trading_app/live/session_orchestrator.py
  - trading_app/live/webhook_server.py

## Why

v6.1 Phase 5 listed Pass Three as the fourth tail-end item to land alongside M4/M5/S2 in PR #121. The other three landed; Pass Three was held back because no IMPLEMENTATION stage permitted `pipeline/check_drift.py` at the time. This stage closes that gap.

Burying Pass Three in deferred-status leaves the rationale-discipline rule unenforced: any new magic number in `trading_app/live/` could land without a `Rationale:` comment and the codebase loses the institutional-grade auditability Robert Carver, *Systematic Trading*, Ch. 4 requires.

## History

Original implementation: PR #124 (`chore/pass-three-drift`, commit `6c279810`, 2026-04-26). Abandoned by parallel terminal; CI failed on a base-staleness test fix landed by PR #126 in main; rebase onto main developed 4 production-code conflicts (`pipeline/check_drift.py`, `trading_app/live/alert_engine.py`, `trading_app/live/webhook_server.py`, `tests/test_pipeline/test_work_queue.py`). Re-extracted on fresh `chore/pass-three-drift-v2` from current `origin/main` 2026-04-27 — strips CRLF noise + base-staleness, applies real content delta only.

## Blast Radius

- **Files modified:** `pipeline/check_drift.py` adds one new function `check_magic_number_rationale(trading_app_dir)` and one new entry to the `CHECKS` registry. 5 trading_app/live/ files retag 9 existing magic-number constants with `Rationale:` comments (no behavior change).
- **Drift count:** 119 → 120.
- **Production behavior:** zero impact. Drift checks are CI/pre-commit-only static analysis; do not run at trading-app runtime. Comment-only edits in trading_app/live/ change no executable code.
- **Pre-commit hook impact:** AST-walk over ~30 .py files in `trading_app/live/`. Expected <0.3s, not in `SLOW_CHECK_LABELS`.
- **Reversibility:** single squash commit; revert via `git revert`.

## Rule

For every UPPER_SNAKE_CASE assignment (class-body or module-level) in `trading_app/live/` whose value is a numeric literal with `abs(value) > RATIONALE_THRESHOLD` (default 10), require either:
  (a) a comment containing "Rationale:" or "rationale" (case-insensitive) within ±10 lines of the assignment, OR
  (b) the constant name appears in `RATIONALE_WHITELIST` (initially empty — names added only with explicit justification).

## Adversarial-audit-gate

Per `.claude/rules/adversarial-audit-gate.md`: this touches `pipeline/check_drift.py` (production tooling). Dispatch evidence-auditor BEFORE merge.

## Re-trigger

If a future audit finds the rationale-discipline check too permissive (e.g., a magic number landed without rationale and the check missed it), tighten the regex from `\brationale\b` to require the literal `Rationale:` prefix. Track in `deferred-findings.md`.
