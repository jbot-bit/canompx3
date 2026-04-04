Resume interrupted work safely by rebasing on current truth: $ARGUMENTS

Use when: returning to unfinished work, new session on old task, stale STAGE_STATE,
contradictory state between terminals, "where was I", "pick up where I left off"
Triggers: "resume", "rebase", "pick up", "continue", "where was I", "what's the state"

## STEP 1: GATHER CURRENT TRUTH (mandatory, every resume)

Run these in parallel:
1. `git log --oneline -10`
2. `git status --short`
3. `git stash list`
4. Read `docs/runtime/STAGE_STATE.md` (if exists)
5. Read `HANDOFF.md` (if exists)
6. If task involved DB: `python -c "import duckdb; db=duckdb.connect('gold.db',read_only=True); print(db.sql('SELECT table_name, estimated_size FROM duckdb_tables()').fetchall())"`
7. If task involved pipeline: `python scripts/tools/pipeline_status.py --status`

## STEP 2: STALE DETECTION (two-tier, drift-first)

Extract `updated` timestamp and scope_lock files from STAGE_STATE.md.

### Tier 1 — DRIFT STALE (authoritative, check first)

Run drift check first: `python pipeline/check_drift.py` — if drift checks fail, scope assumptions may be invalid.
Then run in parallel:
- `git log --oneline --since="[updated]" -- [each scope_lock file]`
  → Any commits touching scope files since last update? → DRIFT STALE
- `git diff --name-only HEAD -- [each scope_lock file]`
  → Uncommitted external changes to scope files? → DRIFT STALE
- If pipeline-related: `python scripts/tools/pipeline_status.py --status`
  → Data rebuilt since last update? → DRIFT STALE

### Tier 2 — AGE STALE (fallback, only if Tier 1 clean)

If Tier 1 found NO drift:
- Is `updated` timestamp >4 hours old? → AGE STALE
- If <4 hours and Tier 1 clean → NOT STALE

### Output
```
STALE CHECK:
  Tier 1 (drift): [CLEAN | STALE — reason]
  Tier 2 (age):   [CLEAN | STALE — N hours old] (only if Tier 1 clean)
  Verdict: FRESH | DRIFT STALE | AGE STALE
```

## STEP 3: CROSS-TERMINAL CONFLICT CHECK

If STAGE_STATE.md exists, check:
- `terminal` field — does it match current terminal?
- `git log --oneline --since="[updated]" -- [scope_lock files]` — any commits from another session?
- Any scope_lock files modified but uncommitted by another process?

```
CONFLICT CHECK:
  Last stage update: [timestamp] by [terminal]
  Commits since: [count] — [summary if any]
  Scope files changed externally: [list or "none"]
  Verdict: CLEAN | CONFLICT
```

If CONFLICT → flag stale assumptions, force re-preflight.

## STEP 4: RESTATE POSITION

```
TASK: [from STAGE_STATE.md]
STAGE: [N/M — description]
MODE: [current mode]
COMMITTED: [what's in git for this task]
UNCOMMITTED: [git status relevant to scope]
STALE: [what changed since last known state]
```

## STEP 5: DECISION

| Situation | Action |
|-----------|--------|
| Nothing changed, stage active | "Safe to continue Stage N." |
| Minor external changes, no scope impact | "Changes noted, no impact. Continuing." |
| External changes affect scope_lock files | "Rebase required." → re-read files, update STAGE_STATE, re-preflight |
| Contradictory state or abandoned stage | "Cannot resume safely. Reclassify via /stage-gate." |

## STEP 6: UPDATE STATE

Update `docs/runtime/STAGE_STATE.md` with refreshed truth + timestamp.
Dispatch back to /stage-gate Step 5 (continue) or Step 3 (reclassify).
