# Overall Cleanup & Action Plan — 2026-06-10

## Context

Operator asked for a consolidated cleanup + action plan for the whole project,
noting plans already exist (don't duplicate) and warning that **other terminals
are actively working** (the fleet is live — treat all state as moving, prefer
read-only synthesis over racing edits). This plan is built from *verified* git/DB
state this session, not from baton/stage titles (which the repo's own doctrine
flags as the stalest signal — `feedback_blocker_framed_batons_outlive_their_fix`).

## Verified ground truth (this session, 2026-06-10)

- **We ARE on latest.** This worktree HEAD `03224494` == `origin/main`,
  **0 ahead / 0 behind**, working tree clean. `git fetch` pulled nothing new.
  The d3-sizing-seam sibling committed 52 min ago but only on its own branch
  (stash artifacts) — nothing new landed on main.
- **Fleet is live: 26 worktrees** (mix of `.codex/worktrees`, `.worktrees`,
  and `canompx3-*` siblings). At least one (d3-sizing-seam) actively merging.
- **Stage files: 67 total.** 53 say `mode: IMPLEMENTATION`, but **37 carry a
  done/✅/merged marker in the body** → the `mode:` field is stale, not the work.
  Bulk-deleting on `mode:` alone is unsafe; ~27 have no done-marker and need
  per-file verification (work-landed-on-main? then delete; else keep).
- **2 BROKEN live-readiness items** (from the pulse): Criterion 11 state-db
  identity mismatch + Criterion 12 SR state mismatch. Fix is mechanical:
  `python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto`.
- **38 git stashes** — possible forgotten work; needs triage, not blind drop.
- **HANDOFF.md decayed** — no longer matches the canonical action-queue render.

## The cleanup backlog (priority order)

### P0 — Live-readiness BROKEN (do first, mechanical, capital-adjacent)
Refresh C11/C12 control state for `topstep_50k_mnq_auto`. Verify the mismatch is
*real* before running (titles lie): inspect the C11/C12 state surfaces, then run
`refresh_control_state.py`. **This is a state refresh, not a code edit — but it
touches the live-arm gates, so confirm with operator before running** (Tier B).

### P1 — Stale stage-file sweep (read-only verify → delete-only)
For each of the ~27 `IMPLEMENTATION`-no-marker stages: prove work-landed via
`git log origin/main` on the actual scope_lock files (NOT keyword grep — commit
msgs don't echo slugs). Landed → delete the stage file. Not landed → keep + flag.
Mechanical, reversible, no production logic. **Tier A** (act, report after) —
BUT coordinate: stage files under `docs/runtime/stages/` are shared-state; run
the three-check protocol (`multi-terminal-shared-file-hygiene.md`) before staging
deletes, since siblings may be writing there.

### P2 — Stash triage (38 stashes)
`git stash list` → for each, `git stash show -p` summary → classify
keep/apply/drop. Cross-ref `project_stash_catalogue_2026_06_10` baton (already
started). Drop only stashes confirmed superseded by merged commits. **Never blind
`stash clear`.**

### P3 — Worktree drain/reap (26 worktrees)
`drain_worktrees.ps1` tooling already landed (`79d54f97`). Owed: reap the 2
worktrees flagged in `project_drain_worktrees_RESUME` baton — **from a PLAIN
terminal**, because the destroy-guard false-positives "unpushed" in-session.
Per-worktree: confirm 0 unique unpushed commits + clean, then
`git worktree remove` + `git branch -d`. Operator-owned step (needs a non-Claude
shell).

### P4 — HANDOFF.md re-sync
Regenerate/realign HANDOFF.md against the canonical action-queue render so the
decay signal clears. Docs-only, **Tier A**.

### P5 — Baton reconciliation (8 live batons from SessionStart)
Each `type: project` baton asserting NEXT/RESUME/OPEN: falsify against
git/DB, then walk stale titles back or delete. Several are already done on main
(e.g. defect-b spread capture `b8288909`). Docs-only, **Tier A**.

## Explicitly OUT of scope here (already have their own plans)
- C11 throttle / readiness closeout → `2026-06-03-strict-c11-readiness-closeout.md`,
  `2026-06-04-c11-*.md`
- Worktree cleanup mechanics → `2026-06-06-worktree-cleanup-handoff.md`
- Repo hygiene tidy → `2026-05-03-repo-hygiene-tidy-plan.md`
- D3 sizing seam (ACTIVE in sibling worktree — do not touch)
- App overhaul / preflight cohesion → its own staged plan

## Verification
- After P1: `python pipeline/check_drift.py` (expect 184/0) — deletes shouldn't
  affect drift, but confirm no stage referenced by a drift check.
- After any control-state refresh (P0): re-run the live preflight / pulse and
  confirm C11/C12 flip to clean.
- Throughout: `git fetch` + ahead/behind check before each commit (fleet is live).

## Coordination posture (operator's explicit warning)
Other terminals are working. This plan is **synthesis + verified-delete only**.
No production-logic edits, no schema, no live-wiring. Anything capital/schema/
destructive (P0 control-state, P3 worktree reaps) stops for operator GO and runs
from the correct shell.
