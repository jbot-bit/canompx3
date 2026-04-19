---
mode: IMPLEMENTATION
slug: fix-drift-ci-skip
stage: 1/1
started: 2026-04-19
task: "Drift checks skip silently when CI=true (no DB by design); local stays fail-closed"
---

# Stage 1 — CI-aware DB-skip helper for 5 drift checks

## Context
PR #9 CI fails because 5 drift checks (37, 61, 80, 92, 95) return
violations when gold.db is missing — by design for local fail-closed.
CI legitimately has no DB per CLAUDE.md "local disk, no cloud sync".

Previous attempt: `continue-on-error: true` on the workflow step. That
hid the gate. User pushed back: "we cant leav ethis shit lyting aroundd.
need to plan and implement."

## Approach (proper, not band-aid)
Add `_skip_db_check_for_ci(msg)` helper that returns `[]` (silent skip,
counted as skip_count) when `CI=true` (auto-set by GitHub Actions) or
`SKIP_DB_CHECKS=1`. Otherwise returns `[msg]` (preserves local
fail-closed). Same check function, env-controlled response.

Apply to 5 checks where DB-missing is currently violation:
- check 37 (gold.db existence)
- check 61 (family_rr_locks coverage)
- check 80 (holdout contamination)
- check 92 (Phase 4 SHA integrity)
- check 95 (prop_profiles alignment)

Then drop `continue-on-error: true` from drift step in ci.yml — drift
becomes a real hard gate again on CI.

## Scope Lock
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift_ws2.py
- .github/workflows/ci.yml

## Acceptance criteria
1. New `_skip_db_check_for_ci` helper unit-tested both branches.
2. 5 checks updated to use helper.
3. `python pipeline/check_drift.py` locally without DB → still fails (canonical).
4. `CI=true python pipeline/check_drift.py` without DB → all 5 SKIPPED (counted in skip_count).
5. `continue-on-error: true` removed from drift step in ci.yml.
6. CI green on PR #9.

## Blast Radius
Affected files:
- pipeline/check_drift.py: 1 new helper + 5 check rewrites (37, 61, 80, 92, 95)
- tests/test_pipeline/test_check_drift_ws2.py: 2 new tests for helper + update existing 37 tests if needed
- .github/workflows/ci.yml: drop `continue-on-error: true` from drift step

Downstream consumers:
- pipeline/check_drift.py runner (single caller, internal)
- CI workflow (`.github/workflows/ci.yml`) — drift step
- Local dev workflow (`python pipeline/check_drift.py` direct invocation)
- Pre-commit hook (`.githooks/pre-commit` — runs check_drift)

Behavior change boundaries:
- Local without DB: NO change (still fails — canonical fail-closed)
- Local with DB: NO change (full enforcement)
- CI=true env: 5 checks now skip silently (was: violate)
- SKIP_DB_CHECKS=1: same as CI=true (manual override for non-CI contexts)
