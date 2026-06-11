# Stage: drain_worktrees audit/reap mode

task: Add `-Audit` (read-only decision surface) and `-ReapRedundant` (merged-only `git branch -d`) modes to drain_worktrees.ps1, plus a REDUNDANT/STRANDED display split of the DIVERGED bucket. Correct a stale header comment asserting a non-existent drift check.

mode: IMPLEMENTATION

## Scope Lock
- scripts/tools/drain_worktrees.ps1

## Blast Radius
- scripts/tools/drain_worktrees.ps1 — only file modified. Adds two `[switch]` params (`-Audit`, `-ReapRedundant`), a display-only `Get-StrandedPlusCount` helper (git cherry `+` count), a `Get-UniqueFileCount` helper (merge-base..branch diff), the REDUNDANT/STRANDED split inside the existing DIVERGED branch of the plan loop, an audit-line emitter, and a `git branch -d` (merged-only, never `-D`) reap loop. Corrects the stale `@canonical-source` comment (lines 54-56) that claims `check_drain_worktrees_capital_prefix_parity` exists — verified this session it does NOT (real parity checks at check_drift.py:17354/17360 guard the Python hooks only).
- Reads: git refs, origin/main (already fetched at session start). `-Audit` is READ-ONLY (no fetch — works under live-peer lease). `-ReapRedundant` does a fresh fetch then `git branch -d`; reflog-recoverable; never pushes; never force-deletes.
- Writes: docs/runtime/drain_worktrees.log (append), and (only under -ReapRedundant) deletes merged local branch refs via git's own merged-only gate.
- No drift-check change. No `$CapitalPathPrefixes` edit. No Python touched.
- Live peer holds main this session — tool stays read-only by default; only -ReapRedundant mutates, operator-invoked from a plain terminal.
