---
task: "Build a read-only stale-stage audit tool + archive the provably-done, uncontested stage files. 30 of 53 docs/runtime/stages/*.md are non-CLOSED but their work has long since merged — they pollute the /next + /orient brief (forces '+50 more' truncation) and create false 'active stage' signals the stage-gate hook reads every prompt. Tool classifies each stage as DONE_SAFE / LIVE_OR_CONTESTED / UNVERIFIABLE; only DONE_SAFE files are archived. Safety gates: never archive a stage whose scope touches trading_app/live/ if any scope file was committed within 48h, never archive if a peer worktree is dirty on any scope file, never archive UNVERIFIABLE (e.g. gitignored .env scope)."
mode: CLOSED
scope_lock:
  - scripts/tools/stage_reaper_audit.py
  - tests/test_tools/test_stage_reaper_audit.py
blast_radius: |
  ## Blast Radius
  - scripts/tools/stage_reaper_audit.py — NEW read-only tool. Reads docs/runtime/stages/*.md,
    parses mode + scope_lock, checks git log recency on scope files + peer-worktree dirtiness,
    emits a classified report. --apply moves DONE_SAFE files to docs/runtime/stages/archive/
    (git mv, reversible). Default dry-run. Zero writes to gold.db, zero capital paths executed.
  - tests/test_tools/test_stage_reaper_audit.py — NEW. Unit tests for the classifier with
    synthetic stage files (done/live/contested/unverifiable) in tmp_path.
  - Reads: docs/runtime/stages/*.md (read-only), git log (read-only), peer worktree status (read-only).
  - Writes: NONE in dry-run; with --apply only `git mv` of DONE_SAFE stage files into archive/.
  - Downstream: declutters claude_superpower_brief + stage-awareness hook input. No production logic.
acceptance:
  - "python scripts/tools/stage_reaper_audit.py  → prints classified report, exit 0, NO files moved"
  - "uv run pytest tests/test_tools/test_stage_reaper_audit.py -p no:timeout -q  → all pass"
  - "the 2 worktree-lease stages (merged via #326+#329) classify as DONE_SAFE"
  - "any trading_app/live stage with a <48h scope commit classifies as LIVE_OR_CONTESTED (NOT archived)"
  - "python pipeline/check_drift.py  → exit 0"
---

## STATUS (2026-05-30, Saturday build)

Picked as an uncontested Saturday build task: no peer worktree is in docs/runtime/stages/
tooling; literature coverage already clean (exit 0); action queue empty. Capital-path stage
files are explicitly protected from archival by the 48h-recency + peer-dirty gates.

### Plan
1. Build scripts/tools/stage_reaper_audit.py (dry-run default classifier). ✅
2. Tests with synthetic stages covering all 4 classes. ✅ 9/9
3. Dry-run on the real 53 stages; eyeball the DONE_SAFE list. ✅
4. --apply to archive ONLY DONE_SAFE. ✅

### Outcome (executed)
- Reuses canonical parse_field/parse_scope_lock from stage-gate-guard.py (no re-encoded logic).
- Live dry-run on 54 stages: DONE_SAFE=1, LIVE_OR_CONTESTED=26, UNVERIFIABLE=5, CLOSED=22.
  Conservative-by-design: 4 peers committing today → most stages correctly held.
- Dry-run SURFACED a classifier gap (DESIGN stage on pipeline/dst.py would have archived) →
  added DESIGN-never-archived gate. Re-run: 1 DONE_SAFE.
- Archived (git mv) the 1 DONE_SAFE: 2026-05-27-paper-trade-sync-automation.md → archive/.
- Reusable: as peer scope quiesces over coming days, re-run to retire the backlog safely.
- ✅ 9/9 tests, ✅ ruff clean, ✅ drift (see below).
