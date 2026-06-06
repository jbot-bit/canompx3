---
goal: Build a canonical fleet-state brain so 4 parallel terminals stop colliding and losing work
plan_file: C:/Users/joshd/.claude/plans/b-full-planned-out-lucky-hummingbird.md
current_stage: 1
stages: 4
status: IN_PROGRESS
updated: 2026-06-06
unfinished: [Stage 2 point guards at the brain, Stage 3 awareness hooks + statusline, Stage 4 plan-anchor resurfacing hook + drift-arrest]
---

# Active Plan — Fleet-State Brain

**Operator pain:** 4 terminals at once waste time fucking each other up; work drifts
off mid-plan and the original deliverable is forgotten. A month of point-guards
never touched the root.

**Root causes (from the ~40-baton corpus, verified live):**
1. Split-brained liveness (7 incidents) — 3+ guards each decide "is a peer live"
   from a different source; they disagree under 4 terminals.
2. Destructive ops on stale claims (5 incidents) — `close --force` has no lease check.
3. Plan/intent drift — no anchor survives context compaction.

**The fix:** ONE brain (`scripts/tools/fleet_state.py`, canonical heartbeat-
authoritative resolver) that every guard/hook/tool reads, + a lease guard at the
destructive chokepoint, + awareness, + this plan anchor.

## Stages
- **Stage 1 (IN PROGRESS):** `fleet_state.py` resolver + `_worktree_hollow.py`
  predicate + `active_plan.py` anchor helper + tests. Read-only; no destructive
  cleanup; no force-close changes.
- **Stage 2:** point the guards (worktree_guard, close --require-unleased,
  stage_reaper, project_pulse) at the brain.
- **Stage 3:** awareness — SessionStart fleet cue + statusline badge.
- **Stage 4:** plan-anchor resurfacing hook (re-surface THIS file every
  startup/clear) + drift-arrest cue.

### Stage 2 GAPS — peer-terminal contribution (2026-06-06, execution-verified)
A sibling terminal (`...2026Thu04`) independently reached the same destructive-op
problem, verified three gaps by EXECUTION, then stood down (correctly — this tree
was live building the brain). Absorb these into Stage 2 so the guard adds real
value instead of theater:
1. **`git already self-blocks merge/pull/rebase` on dirty overlap** (proven:
   merge aborts, exit 1, file preserved). So the destructive chokepoint must
   target what git does NOT protect — `reset --hard`, `checkout -f`, `restore`,
   `clean -f`, `branch -D`, `stash drop`. Do NOT guard `merge` — it is theater.
2. **MCP-git surface is uncovered.** `.claude/hooks/mcp-git-guard.py` exists but
   is UNWIRED in settings.json. Stage 2 points "guards" at the brain but only
   names Bash-tool guards; `mcp__git__*` ops would bypass the brain entirely.
   Point the MCP path at the brain too.
3. **Registration self-check.** Add a drift guard that brain-consuming hooks stay
   wired (the orphaned mcp-git-guard proves hooks silently fall out of wiring).

## Done-criteria (Stage 1)
- `fleet_state()` classifies all 6 classes correctly over a fixture fleet.
- Hollow predicate flags the gutted tree, not a real refactor.
- Liveness DELEGATES to `worktree_guard._peer_is_live` (no re-encoding).
- `pytest tests/test_tools/test_fleet_state.py tests/test_tools/test_worktree_hollow.py` green.
- `check_drift.py` passes; dead-code grep clean.

## Stage 1 fair-fight corrections (2026-06-06 — before declaring done)
A fair-fight audit found the first "done" claim narrowed away three things; all
now closed:
- **Liveness TRUE branch** now has a real fresh-heartbeat→LIVE integration test
  (real `.beat` + PID-stub fallback). Live fleet now shows tree `...2026137`
  classified LIVE — the TRUE branch fires in production, not just in the matrix.
- **Churn-list re-encoding** removed: `fleet_state` imports the new canonical
  `scripts/tools/_worktree_churn.py` (`OPERATIONAL_CHURN_PATHS` + `is_churn_path`)
  instead of an inline `_CHURN_PATHS`.
- **HOLLOW+work-at-risk collision** guarded: a gutted tree with unpushed commits
  OR a real tracked non-deletion edit now classes NEEDS_FINISH (NOT reap-eligible).
  Stage 2's reaper inherits this — a HOLLOW verdict now guarantees no unpushed
  commits AND no real tracked edits at risk.

### Adversarial logic-review fixes (2026-06-06, independent evidence-auditor pass)
The independent logic review (operator-sequenced) found two real issues my own
self-review missed; both fixed (this is why the fair-fight pass exists):
- **`is_churn_path` was substring, now path-SEGMENT match.** `live_journal.db`
  falsely matched `tests/test_live_journal.db_helpers.py`, silently dropping a
  real edit from the work-at-risk count. Now matches exact path or trailing
  `/<segment>` only. (`_worktree_churn.py`)
- **HOLLOW work-at-risk signal is `real_nondel_dirty`, NOT `real_dirty`.** The
  auditor's literal suggestion (`real_dirty>0`) was WRONG — deletions count in
  `real_dirty`, so every hollow tree (real_dirty≈deletions) would have flipped to
  NEEDS_FINISH, defeating the class. Correct signal = non-churn, NON-deletion,
  TRACKED edits. Untracked `??` scaffolding (`.claude/`, `.codex/`) is explicitly
  EXCLUDED — verified live that counting it flipped the poisoning tree `...2026042`
  to NEEDS_FINISH and would have BLOCKED the cleanup. After the fix `...2026042`
  is HOLLOW (reapable) as intended. (`fleet_state.py::_count_dirty` / `_classify`)

### FOLLOW-UP DEBT — canonical churn-list adoption (Option A, deferred by design)
`_worktree_churn.py` is canonical-BY-DESIGN, but the genuine sibling churn list
is NOT yet migrated onto it:
- `scripts/run_live_session.py:558` `_DRIFT_IGNORE_SUFFIXES = ("live_journal.db",
  "HANDOFF.md")` — the one true semantic sibling; a CAPITAL path, so migration is
  a separate, larger stage with its own gate (do not fold into Stage 1).
- NOT churn siblings (verified — do NOT migrate): `checkpoint_guard.py`
  `DURABLE_ROOTS` (artifacts to PRESERVE), `check_root_hygiene.py` `ALLOWED_FILES`
  (root-clutter allowlist), `check_referenced_paths.py` known-root prefixes
  (path-resolution roots). They overlap on `HANDOFF.md` but answer different
  questions. The plan's "3+ divergent churn copies" was imprecise.

## ⚠ THE ORIGINAL DELIVERABLE — DO NOT FORGET (operator, 2026-06-06)
The brain/wiring is a means, not the end. The original ask was to **clean up the
worktrees**. Sequence the operator wants:
1. Build + verify this fleet-state brain works (Stage 1 → tests green).
2. Code review + logic review the new brain; implement any fixes found.
3. THEN use the verified brain to actually clean up / clear the stale worktrees
   (the hollow `...2026042` tree + any MERGED/STALE trees fleet_state flags),
   under the re-verify protocol — never `--force` a LIVE or NEEDS_FINISH tree.
Do not declare the task done after Stage 1; the cleanup is the finish line.
