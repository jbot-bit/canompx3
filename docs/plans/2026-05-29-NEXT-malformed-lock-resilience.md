# NEXT (after /clear) — Malformed-lock resilience in session-start guards

**Picked because:** closest unblocked reaper follow-up (#1 of the two recorded in the
stale-process-reaper stage closure). Self-contained, ~1–2 files, confirmed-live bug.
No sibling worktree is touching `session-start.py` (checked 2026-05-29: claude-tailor
edits `settings.json`/`CLAUDE.md`/new hooks; roi-hunt + topstep-telemetry edit
`HANDOFF.md`; ehr + nyse worktrees edit their own stage files only).

> Deferred sibling follow-up (#2): extend the reaper to kill the ~51 orphaned
> top-level node Claude-client trees (the actual slowdown root cause). Bigger blast
> radius, touches capital-safety kill logic — do that as its own stage, NOT this one.

---

## The bug (CONFIRMED LIVE 2026-05-29)

The live `.git/.claude.pid` is **invalid JSON**:

```
{"pid": 63904, ..., "worktree": "C:\Users\joshd\canompx3", ...}
                                    ^^^^^^^ unescaped backslashes (\U \j \c)
```

`json.loads()` raises `JSONDecodeError`. Every reader of the lock file catches that
and falls back to `existing = {}` / empty, which makes these guards **silently no-op**:
- branch-flip-guard (`branch_at_start` gone)
- head-flip-guard (`head_at_start` gone)
- stale-lock age reclaim (`iso_started` gone)
- the worktree-mutex health check

So the mutex/branch protections are currently OFF and nothing tells the operator.

Note: the CURRENT writer in `session-start.py:486` uses `json.dumps(...)` which DOES
escape correctly + writes `indent=2`. The live broken lock is single-line and dated
`2026-05-24` → written by an OLDER writer (or a non-Python writer). The durable defense
is therefore at the **reader / health-check layer**, not just the writer.

## Purpose

A malformed lock (from ANY writer, ANY version, ANY source) must NOT silently disable
the guards. It must either (a) be detected as malformed-and-stale and reclaimed with a
surfaced WARNING, or (b) surface "guards inactive — malformed lock" to the operator.
Fail-LOUD on malformed, never fail-silent.

## Files (expected ≤2)

- `.claude/hooks/session-start.py` — the lock reader/health-check path
  (around `existing = json.loads(...)` ~line 510 and the health-check ~line 519).
  Add: when the lock file exists but parses to empty/malformed JSON for THIS worktree,
  emit the same class of "guard inactive — recover with `rm <lock>` + restart" line that
  the no-`branch_at_start` branch already emits (~line 519–523). Today malformed JSON
  silently degrades; the no-branch case already surfaces — make malformed parity with it.
- companion test (find where session-start lock tests live first;
  `grep -rln "claude.pid\|_session_lock\|branch_at_start" tests/`).

## Blast radius (verify before editing)

- Run `/crg-context` then `/crg-blast` on `.claude/hooks/session-start.py` (minimal detail).
- All readers of `.claude.pid`: branch-flip-guard.py, head-flip-guard.py, mcp-git-guard.py,
  shared-state-commit-guard.py, _branch_state.py. Confirm they share the same parse-and-
  fallback pattern; if the fix belongs in a shared helper (`_branch_state.py`), prefer that
  over patching each reader (institutional-rigor: refactor, don't re-encode).
- **Fail-open invariant is SACRED**: a missing/unreadable lock must STILL exit 0 (never
  block a session it can't read — see branch-flip-protection.md "Fail-safe guarantee").
  The change is: malformed-but-PRESENT → surface a WARNING, not block. Missing → silent pass.

## Approach (2-pass, present design gate first)

1. **Discovery:** read the reader + health-check block, confirm whether all 5 readers
   funnel through `_branch_state.py` or each parse independently. Decide single helper vs
   per-reader. Articulate: "malformed present lock → loud WARNING; absent lock → silent pass."
2. **Implementation:** smallest change that makes malformed-JSON surface parity with the
   existing no-`branch_at_start` warning. Add a regression test: malformed `.claude.pid`
   for this worktree → health-check emits the WARNING line (and does NOT raise / does NOT block).
3. Verify: targeted test + `python pipeline/check_drift.py` (or `--fast` first) + self-review.

## Also do while here (cheap, optional)

Reclaim the stale live lock so guards come back on THIS machine:
`rm C:\Users\joshd\canompx3\.git\.claude.pid` then restart the session (session-start
rewrites a clean, properly-escaped lock). The 2026-05-24 lock is from a dead PID anyway.

## Done criteria

- Malformed-but-present lock → operator sees a "guards inactive, recover with rm+restart"
  WARNING (parity with the no-branch_at_start path). Absent lock → still silent exit 0.
- Regression test proves both branches.
- `check_drift.py` passes; dead code swept; self-review shown (not just claimed).

## Context to reload after /clear

- This file.
- `.claude/rules/branch-flip-protection.md` (fail-safe guarantee + companion head-flip-guard).
- `docs/runtime/stages/2026-05-29-stale-process-reaper.md` (the closure that recorded this follow-up).
- `memory/project_drift_speed_stage1_reaper_shipped_resume_2026_05_29.md` (the lock-writer
  JSON-escaping bug was first flagged here).
