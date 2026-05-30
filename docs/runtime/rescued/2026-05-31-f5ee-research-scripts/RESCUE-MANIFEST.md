# RESCUE MANIFEST — f5ee worktree untracked research (2026-05-31)

## Source
- Worktree: `C:/Users/joshd/.codex/worktrees/f5ee/canompx3` (detached HEAD `788176cb`)
- `788176cb` content == `78a7cdfb` on main (ralph fast-drift fix, already landed) → orphan superseded, safe to prune.
- The committed tip carries nothing unmerged; only these **untracked** files were at risk.

## Rescued files (verbatim copy)
- `rebuild_orb_edge_inventory_2026_05_30.py` — research script (20 KB), ORB edge inventory rebuild dated 2026-05-30. Untracked, never committed anywhere.
- `artifacts-research/` — generated artifacts directory copied from the worktree's `artifacts/research/`.

## Disposition
- These are research scratch outputs, **not** canonical pipeline/trading_app code. No git history to preserve.
- Kept here as a net before pruning worktree f5ee. If the script proves worth keeping, promote it into `research/` on a branch via normal review.

## Provenance
- Rescued by main session 2026-05-31 during the worktree-cleanup plan (Phase A2).
- Pattern mirrors `docs/runtime/rescued/2026-05-30-*` capital-WIP rescue.
