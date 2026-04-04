# Stage-Gate Protocol (AUTO-ENFORCED)

## On every user message, you receive stage context via hook output:
- `stage: none` → no active stage
- `stage: TRIVIAL | task description` → quick fix in progress
- `stage: IMPLEMENTATION | task | (1/2)` → staged work in progress

## Automatic behaviors (no user action needed):

**If stage = none AND user asks for non-trivial work:**
→ Classify the task yourself. Write STAGE_STATE.md BEFORE editing any production code.
→ For trivial fixes on non-core files, the hook auto-creates TRIVIAL state.
→ For core files, write a proper STAGE_STATE with scope_lock.

**If stage = IMPLEMENTATION AND user asks about something else:**
→ Note: "You have an active stage: [task]. Continue, or reclassify?"
→ Don't silently abandon the active stage.

**If stage = TRIVIAL AND user starts a bigger task:**
→ Delete the TRIVIAL state. Reclassify via full staging.

## What counts as production code (hook-enforced):
pipeline/, trading_app/, scripts/ (except reports/infra/gen_*)

## Core files (NEVER TRIVIAL — hook enforces):
Pipeline logic, config, schema, validation, session, DB-write paths.
See NEVER_TRIVIAL list in `.claude/hooks/stage-gate-guard.py`.

## Stale detection
Git log on scope files since last update → if changed → STALE. >4 hours → AGE STALE.

## Mid-execution discipline
When a script fails: fix infrastructure (import, env, path) → resume. Do NOT change behavior. If fix requires system change → STOP, flag, return to user.

## Scope Discipline (anti-creep)
Adding files to scope_lock mid-implementation: (1) inform user why, (2) update blast_radius, (3) NEVER silently expand scope.

## Stage Completion — "done" means PROVEN
Before deleting STAGE_STATE.md: (1) tests pass (show output), (2) dead code swept (`grep -r`), (3) `python pipeline/check_drift.py` passes. All three required.
