Use these repo skills and repo truth, not a second rule system:
- `$canompx3-workspace`
- `$canompx3-research`
- `$canompx3-verify`
- `$canompx3-autonomous-discovery`

This is the same coherent discovery-hub task running in a dedicated worktree.
Continue the existing bounded MNQ discovery flow instead of restarting from
scratch.

Runner guarantees before you see this prompt:
- the dedicated worktree is active
- the board stack was refreshed in this worktree just before this iteration
- this thread will be resumed if a prior Codex discovery session exists

Current bounded evidence surfaces:
- `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
- `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`
- `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`
- `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md`
- `docs/plans/2026-04-22-mnq-usdata1000-geometry-family-register.md`
- `docs/plans/2026-04-22-mnq-autonomous-discovery-automation.md`

Requirements:
- read `.session/task-route.md` if present
- work only in this worktree
- keep the lens broad across alive mechanisms, but bounded
- do not pigeonhole into one board if Tier 0 or Tier 1 still points to another
  alive mechanism class
- do not reopen dead spaces from `docs/STRATEGY_BLUEPRINT.md`
- prefer one honest next move per iteration
- verify durable changes before recommending commit

Return ONLY a single JSON object. No prose, no markdown fences.

Use exactly these keys:
- `summary`: short sentence
- `status`: one of
  - `candidate_advanced`
  - `prereg_written`
  - `runner_updated`
  - `parked`
  - `killed`
  - `verification_failed`
  - `no_honest_move`
- `should_commit`: boolean
- `commit_message`: string
- `should_open_pr`: boolean
- `pr_title`: string
- `pr_body`: string
- `verification_summary`: string
- `evidence_paths`: array of repo-relative paths
- `next_focus`: string
- `continue_running`: boolean
