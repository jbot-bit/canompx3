# RESCUE MANIFEST — 9aa2 worktree new chordia work (2026-05-31)

## Why this is a NET, not a prune
Worktree `C:/Users/joshd/.codex/worktrees/9aa2/canompx3` is **NOT being pruned** — it
holds active, fresh (2026-05-31-dated) untracked work, indicating a live session is using it.
This rescue is a precaution only; the worktree stays.

## Rescued (untracked, dated today — beyond the committed tip 6910d0c3)
- `chordia_evidence_factory.py` (770 lines) + `test_chordia_evidence_factory.py`
- `chordia_replay_batch_bridge.py` (357 lines) + `test_chordia_replay_batch_bridge.py`
- `artifacts-chordia_evidence_factory_2026_05_31/`, `artifacts-chordia_evidence_factory_full_2026_05_31/`
- `lane-bench-plan-wip.patch` — uncommitted edit to the lane-bench-state-machine plan doc.

## Disposition
- This is in-flight chordia evidence-factory tooling, separate from the lane-bench tip already
  rescued into PRs #332 (tooling) / #333 (capital code). Left in place for its author to finish/commit.
- Copy retained here so the work is recoverable even if the worktree is later disturbed.

## Provenance
- Rescued by main session 2026-05-31 during worktree-cleanup Phase D pre-prune safety sweep.
