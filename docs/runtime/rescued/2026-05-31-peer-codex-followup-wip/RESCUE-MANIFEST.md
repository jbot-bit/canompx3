# RESCUE MANIFEST — peer Codex WIP in shared main worktree (2026-05-31)

## How this arose
A peer Codex session (lock pid 22160, started 2026-05-30 12:52 UTC, **now DEAD** — pid+ppid both
gone) was working in the SHARED main worktree `C:/Users/joshd/canompx3`. It had:
- switched the worktree branch `main` → `codex/project-followup-automatic` (a zero-divergence
  branch at the same commit `0a2dcb9a`), and
- left two distinct groups of uncommitted WIP behind.

Discovered mid-cleanup when the branch-flip guard blocked commits. No live writer was active
(file hashes + mtimes static at capture time); this is leftover state from the dead session.

## Rescued content (two patches)

### `peer-codex-followup-wip.patch` (311 lines) — live-launch preflight guard
Tracked-file diff over:
- `scripts/run_live_session.py` (+80) — adds `PROJECT_ROOT` sys.path insertion and a NEW
  capital-safety check in `_check_repo_drift_for_live`: live launch now FAILS if the repo is
  **ahead of origin** ("push or isolate before live launch"), not just if behind/dirty.
- `tests/test_scripts/test_run_live_session_preflight.py` (+76) — tests for the above.
- `docs/plans/2026-05-29-live-trading-readiness-runbook.md`, `docs/runtime/decision-ledger.md` — docs.

### `peer-codex-project-pulse-wip.patch` (755 lines) — project_pulse expansion
Staged-index diff over:
- `scripts/tools/project_pulse.py` (+364) — large feature addition.
- `scripts/tools/live_readiness_report.py` (+3), and tests `test_project_pulse.py` (+211),
  `test_live_readiness_report.py` (+22).

## Disposition
- These are **Tier B capital-path** changes (live preflight) authored by another session.
  NOT merged to main autonomously.
- Preserved as: (a) these two patches, AND (b) a real commit on branch
  `session/joshd-peer-codex-followup-rescue` (off `0a2dcb9a`) so they survive as recoverable
  history. Operator to decide whether to PR/review/discard.
- The shared main worktree was restored to a clean `main` after capture.

## Provenance
- Rescued by main session 2026-05-31 during worktree-cleanup plan.
- Related rule: `.claude/rules/multi-terminal-shared-file-hygiene.md` (this is exactly that class).
