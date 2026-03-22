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

## Stale detection (drift-first):
1. Git log on scope files since last update → if changed → STALE
2. >4 hours since update → AGE STALE (fallback)

## Mid-execution discipline:
When a script or command fails during execution:
- Fix the infrastructure (import, env, path) → resume the plan
- Do NOT change behavior (rewrite flow, replace scripts with manual steps, rework pipeline)
- If the fix requires a system change → STOP, flag it, return to user

## Token efficiency:
- TRIVIAL state = 3 lines. Don't over-document quick fixes.
- Stage awareness hook = 1 line per message. Minimal overhead.
- Don't re-read STAGE_STATE.md if the hook already told you the mode.
