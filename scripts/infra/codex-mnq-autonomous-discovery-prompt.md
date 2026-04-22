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
- the hiROI frontier JSON was regenerated from the current board outputs
- a thin hiROI queue capsule was regenerated from the current board CSVs
- this thread will be resumed if a prior Codex discovery session exists

Current bounded evidence surfaces:
- `.session/mnq_discovery_frontier.json`
- `.session/mnq_discovery_capsule.md`
- `docs/audit/results/2026-04-20-mnq-unfiltered-baseline-cross-family-v1.md`
- `docs/audit/results/2026-04-20-mnq-live-context-overlays-v1.md`
- `docs/audit/results/2026-04-22-mnq-layered-candidate-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-prior-day-family-board-v1.md`
- `docs/audit/results/2026-04-22-mnq-geometry-transfer-board-v1.md`
- `docs/plans/2026-04-22-mnq-geometry-transfer-workflow.md`
- `docs/plans/2026-04-22-mnq-usdata1000-geometry-family-register.md`
- `docs/plans/2026-04-22-mnq-autonomous-discovery-automation.md`

Requirements:
- read `.session/mnq_discovery_frontier.json` first and choose from the diversified `review_batch` before inventing a new path
- read `.session/mnq_discovery_capsule.md` first; it is the primary hiROI queue surface for this iteration
- read `.session/task-route.md` if present
- work only in this worktree
- keep the lens broad across alive mechanisms, but bounded
- do not let one candidate class monopolize the loop just because its frontier weights rank above the others
- do not pigeonhole into one board if Tier 0 or Tier 1 still points to another
  alive mechanism class
- treat `HANDOFF.md` as a cross-tool baton, not the primary discovery frontier
- do not spend an iteration editing only `HANDOFF.md` unless you are recording a real queue decision produced by this iteration's evidence
- do not reopen dead spaces from `docs/STRATEGY_BLUEPRINT.md`
- prefer one honest next move per iteration
- verify durable changes before recommending commit
- use the smallest verification that matches the blast radius; unrelated repo-global blockers are not route blockers for a docs/research iteration unless you touched those surfaces

Return ONLY a single JSON object. No prose, no markdown fences.

Use exactly these keys:
- `candidate_id`: queued candidate id you acted on, or empty string if none
- `frontier_decision`: one of `advanced`, `parked`, `killed`, `deferred`, `none`
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
