# Stage 1 — Single Source of Preflight Truth (Design)

**Date:** 2026-06-09
**Status:** APPROVED-PENDING-GO (design audited; awaiting operator go before code)
**Parent:** App Overhaul plan (5 stages). This is Stage 1, the upstream cohesion seam.

## Purpose

The operator reported preflight running **three times** on one START_BOT live launch.
Traced against live code, that is the four-way preflight divergence made physical:

- **Run 1** — `START_BOT.bat:137`: blocking `run_live_session --preflight --strict-zero-warn` subprocess.
- **Run 2** — `START_BOT.bat:160`: spawns the orchestrator (`run_live_session --live`, no `--preflight`),
  which since the 2026-06-05 mandatory-inline-gate fix runs the full 15-gate sweep again at boot
  (pinned by `test_live_launch_runs_preflight_inline_without_preflight_flag`).
- **Run 3** — dashboard LIVE press: `bot_dashboard.py:753` `_run_preflight_subprocess` spawns a third
  `run_live_session --preflight` and text-parses its stdout.

Stage 1 collapses runs 1 and 3 onto ONE in-process engine; Stage 5 later removes run 2's spawn.
Stage 1 is **behavior-preserving** — no gate verdict changes, only where the gate logic lives and how
it is consumed.

## Blast radius (verified)

- No production code `import`s the preflight building blocks from the launcher — only the launcher's
  own entry function, the test file, the dashboard subprocess path, and the drift parser.
- `grep -rn run_live_session trading_app/ pipeline/` → docstring mentions + the `-m` subprocess string +
  `pipeline/check_drift.py:978` static parser. No import-level coupling.
- Baseline: `tests/test_scripts/test_run_live_session_preflight.py` → **71 passed** (cold run 2026-06-09).

## Audit findings absorbed (institutional audit, pre-code)

1. **[HIGH] Drift surface keys on file residence, not import.**
   `check_preflight_status_token_parity` → `_extract_preflight_emitted_tokens(rls)` ASTs
   `run_live_session.py` for `CheckResult(...)` call sites. Moving the gates to `preflight.py` empties
   the token set → fail-closed red drift on a capital-path guard.
   **Fix:** re-point the extractor to parse `trading_app/live/preflight.py` (the new, honest residence).
   Proven by known-bad-token injection still failing closed.
2. **[MEDIUM] Dashboard in-process call changes the failure surface.**
   The subprocess path catches timeout/non-zero and still returns a structured cache entry. The
   in-process call must be wrapped fail-closed so a gate exception yields a `fail` dict, not a raised
   dashboard request.
3. **[LOW] `canonical_inline_copies.py:78`** mentions `run_live_session.py` in a scope description —
   verify the extraction does not orphan an inline-copy parity pairing (cheap check at implementation).

## Approach (recommended — approach 2 of 3)

Move the gate building blocks into `preflight.py`; keep the run-and-print entry function physically in
the launcher, re-importing every moved name back into the launcher namespace so test swaps and the
entry function keep resolving. Add a new structured entry point in the module returning a per-gate
report (identifier, title, status, message, remediation) + overall verdict + strict-block flag. The
dashboard calls that entry point in-process; a thin fail-closed adapter shapes it into the exact
dictionary the arm-guard already consumes.

Rejected: (1) move the runner too → breaks monkeypatch namespace + source-grep tests; (3) refactor the
arm-guard to read the report directly → reaches the capital arm-gate, deferred to a later stage.

`strict_zero_warn`: semantics unchanged. The structured entry point computes `strict_block` identically
to today (advisory-classification via the canonical normalize + not-demo + not-signal-only + warned>0)
and exposes it as a report field. No capital-semantics change.

## Files (scope_lock — 6)

1. `trading_app/live/preflight.py` (NEW)
2. `scripts/run_live_session.py` (re-import + keep runner/main)
3. `trading_app/live/bot_dashboard.py` (in-process adapter, fail-closed)
4. `pipeline/check_drift.py` (re-point token extractor to new residence)
5. `tests/test_scripts/test_run_live_session_preflight.py` (unchanged; stays green)
6. `tests/test_trading_app/test_preflight_module.py` (NEW)

## Implementation order

create module (move + new entry point) → re-import into launcher → re-point drift extractor →
dashboard fail-closed adapter → new tests → full suite + drift + known-bad-token injection.

## Validation

- 71 existing preflight tests stay green (behavior parity).
- `check_drift.py` passes AND a deliberate bad-token injection still trips
  `check_preflight_status_token_parity` (verify guard by injection, not by label — institutional-rigor §11).
- New module test: structured report statuses match printed-path verdicts gate-for-gate on stubbed
  all-pass and one-fail portfolios.
- Dashboard adapter test: an in-process gate exception yields a `fail` dict, not a raised exception.

## Rollback

Single-commit revert restores the monolith. No schema/data migration. Fully reversible.

## Guardians

None — no entry-model or pipeline-data change.
