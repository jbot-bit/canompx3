---
paths:
  - "pipeline/**"
  - "trading_app/**"
  - "scripts/**"
  - "docs/runtime/stages/**"
  - "docs/runtime/STAGE_STATE.md"
---
# Stage-Gate Protocol (AUTO-ENFORCED)

**Load-policy:** auto-injected when editing production code or stage files. Stage status (none/TRIVIAL/IMPLEMENTATION) is still surfaced on every user prompt via the `stage-awareness.py` hook — so non-code turns get a 1-line status without loading the full protocol.


## Stage files live in `docs/runtime/stages/<slug>.md`

Each task gets its own file. Multiple terminals can have concurrent stages without collision.
The legacy `STAGE_STATE.md` file at `docs/runtime/` is **deprecated** — hooks still tolerate the legacy path for backwards-compat, but the file itself is gone and nothing new writes there.

## Stage file format (parser-tolerant — use these to pass `stage-gate-guard.py` first try)

Required fields: `task`, `mode`, `scope_lock`, `blast_radius`. The hook (`.claude/hooks/stage-gate-guard.py`) parses three formats — pick the one that matches your content shape:

**`scope_lock`** — YAML list under the key, OR `## Scope Lock` markdown section with `- path` bullets. Inline `[a.py, b.py]` also works.

**`blast_radius`** — must be ≥30 chars of joined text. Three accepted shapes:
1. **`## Blast Radius` markdown section** (preferred — most tolerant; parser checks this first):
   ```markdown
   ## Blast Radius
   - foo.py — new file, zero callers
   - bar.py — modifies write path; 3 callsites
   - Reads: gold.db (read-only); Writes: none
   ```
2. **Single-line YAML string**: `blast_radius: "foo.py (new), bar.py (3 callsites). Reads gold.db read-only."`
3. **YAML list** — first non-blank line after `blast_radius:` MUST start with `- `:
   ```yaml
   blast_radius:
     - foo.py — new file
     - bar.py — modifies write path
   ```

**FAILS the parser** (silently → BLOCK): nested mapping under `blast_radius:` (`reads:` / `writes:` / `affects:` children). The parser sees the first child key as "next YAML key" and bails. If you want structured content, use the markdown section format instead.

## On every user message, you receive stage context via hook output:
- `stage: none` → no active stage
- `stage: TRIVIAL | [slug] | task description` → quick fix in progress
- `stage: IMPLEMENTATION | [slug] | task | (1/2)` → staged work in progress

## Automatic behaviors (no user action needed):

**If stage = none AND user asks for non-trivial work:**
→ Classify the task yourself. Write `stages/<slug>.md` BEFORE editing any production code.
→ For trivial fixes on non-core files, the hook auto-creates TRIVIAL state in `stages/auto_trivial.md`.
→ For core files, write a proper stage file with scope_lock.

**If stage = IMPLEMENTATION AND user asks about something else:**
→ Note: "You have an active stage: [task]. Continue, or reclassify?"
→ Don't silently abandon the active stage.

**If stage = TRIVIAL AND user starts a bigger task:**
→ Delete the TRIVIAL stage file. Reclassify via full staging.

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
Before deleting your `stages/<slug>.md`: (1) tests pass (show output), (2) dead code swept (`grep -r`), (3) `python pipeline/check_drift.py` passes. All three required.
