---
slug: recover-stage-hygiene-active-detection
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Recover orphaned commit `eb40b35a` (codex/stage-hygiene-active-detection) — adds `_stage_file_is_closed()` helper to `pipeline/system_context.py` so closed/completed stage files (status field set, OR `## Execution Outcome` section present) are dropped from `_list_active_stages` and stop blocking edits via stage-gate-guard. Pure additive change; no callers in main expect the symbol; integrates via existing `_parse_stage_file` → `_list_active_stages` path.
---

# Stage 1: recover stage-hygiene active-detection helper

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/system_context.py
  - tests/test_pipeline/test_system_context.py
  - HANDOFF.md
  - docs/runtime/stages/recover-stage-hygiene-active-detection.md

## Blast Radius

- `pipeline/system_context.py` — additive only: 2-line insert in `_parse_stage_file` (call site) + new `_stage_file_is_closed()` helper between `_parse_stage_file` and `_list_active_stages`. No callers of `_stage_file_is_closed` in main (verified via grep on `pipeline/`, `trading_app/`, `scripts/`, `tests/`, `.claude/`). Closed-stage suppression flows transitively through `_list_active_stages` to two consumers: `stage-gate-guard.py` (uses active-stage list to enforce scope_lock on edits) and `pipeline/system_brief.py` (renders active-stages in session-start brief). Both consumers benefit from the fix — stale closed stages will silently drop out rather than block edits or pollute session-start output.
- `tests/test_pipeline/test_system_context.py` — additive only: adds one new test (`test_closed_stage_files_do_not_count_as_active`) exercising both the status-field path and the `## Execution Outcome` path. Existing tests untouched.

## Why

`codex/stage-hygiene-active-detection` (single commit `eb40b35a`, dated 2026-04-24) added a `_stage_file_is_closed()` helper to `pipeline/system_context.py` so that stage files in `docs/runtime/stages/` whose YAML frontmatter declares `status: closed/complete/completed/done/implemented` (or whose body contains a `## Execution Outcome` H2 section) are **not** treated as active stages by `_list_active_stages()`.

This closes a recurring friction: completed stage files left on disk still appeared "active" to `stage-gate-guard.py`, blocking unrelated edits because the closed stage's `scope_lock` didn't include the file being edited. The fix is the canonical path — let stage authors close out via metadata or a section heading, no manual file deletion required.

The branch was orphaned (never PR'd) but the commit content is sound. Recovering it onto a fresh branch from `origin/main` and dropping the original commit's 6 stage-doc frontmatter touches (those files are gone from main per audit verification 2026-04-27).

## Blast radius

- `pipeline/system_context.py`: ADDITIVE only — adds 2 lines to `_parse_stage_file` (call site) and a new `_stage_file_is_closed()` function. No existing logic modified.
- `tests/test_pipeline/test_system_context.py`: adds one new test (`test_closed_stage_files_do_not_count_as_active`); existing tests untouched.
- No callers of `_stage_file_is_closed` in main (verified via grep on `pipeline/`, `trading_app/`, `scripts/`, `tests/`, `.claude/`).
- Integration: `_parse_stage_file` → `_list_active_stages` → consumed by `stage-gate-guard.py` and `pipeline/system_brief.py`. Closed stages will silently drop out — desired behavior, no callers need updating.

## Acceptance

- New test `test_closed_stage_files_do_not_count_as_active` passes.
- All existing tests in `tests/test_pipeline/test_system_context.py` still pass.
- `python pipeline/check_drift.py` clean.
- Pre-commit checkpoint guard + claim hygiene + behavioral audit all pass.

## Original commit reference

- `eb40b35a` on branch `codex/stage-hygiene-active-detection`
- Author: jbot-bit, 2026-04-24
- Title: "fix(context): ignore closed runtime stage files"
- Files in original commit: `pipeline/system_context.py`, `tests/test_pipeline/test_system_context.py`, plus 6 stage-doc frontmatter touches dropped from this recovery (files no longer exist in main).
