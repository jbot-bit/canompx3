# Workflow Reliability And Stage Ownership Plan - 2026-06-03

## Status

Active plan. This is a workflow-control plan, not a trading or research result.

## Purpose

Stop the recurring operational failures that create ambiguous "done" states:

- wrong worktree
- wrong branch
- wrong DB
- wrong dashboard port/runtime
- stale lease or heartbeat owner
- slow drift/pre-commit overlap
- hidden async failures
- active stage bloat without ownership

Success is workflow reliability, not a lower count of stage files.

## Authority And Grounding

- `CLAUDE.md` and `.claude/` remain the canonical workflow authority.
- `docs/governance/system_authority_map.md` requires linked truth, not copied truth. This plan routes to existing code/checks instead of becoming a runtime source.
- `docs/institutional/pre_registered_criteria.md` bans post-hoc relaxation after seeing results. This plan applies the same discipline to workflow cleanup: classify first, act second, and do not redefine success after seeing the stage count.
- `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md` and `docs/institutional/literature/aronson_2007_ebta_data_snooping.md` ground the anti-selection-bias posture. Workflow equivalent: do not select the easiest stale files and call the system healthy.
- `docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md` grounds disclosure of the full tested population. Workflow equivalent: keep contested, unverifiable, and closed-already stages visible until their ownership/evidence status is resolved.

## Current Measured State

Measured from `C:\Users\joshd\canompx3` on 2026-06-03:

- Branch: `main`
- HEAD: `2e8f3b59`
- Hook path: `.githooks`
- Root stage files: 56
- Active stage files: 17
- Stage reaper summary: `DONE_SAFE=0 LIVE_OR_CONTESTED=25 UNVERIFIABLE=8 CLOSED=23`
- Dashboard port 8080: listening
- Dashboard process owner: `C:\Users\joshd\canompx3-live-launch-tokyo`
- Dashboard state: `SIGNAL/topstep_50k_mnq_auto`, stale heartbeat
- DB selected path: `C:\Users\joshd\canompx3\gold.db`
- DB probe mode: read-only only
- DB warning: hardlink count 2
- Worktrees: 19 total, 3 detached

## Decisions

1. Stage cleanup is blocked until worktree/lease safety is known.
2. Port-open is not dashboard health. Heartbeat and owner worktree must be shown.
3. Drift speed work belongs to `codex/precommit-drift-speed` until that branch is clean or abandoned.
4. Live/dashboard readiness belongs to the worktree that owns the running dashboard or live launcher state.
5. `gold.db` is protected shared truth. Workflow tools may read it only with read-only opens.
6. Agents and Codex are review/execution tools, not default state monitors. Deterministic scripts own status.

## Phase 0 - Lease And Main-Checkout Safety

Goal: prevent wrong-session mutation on main.

Checks:

```powershell
python scripts\tools\worktree_guard.py --status --json
python scripts\tools\workflow_doctor.py status
git status --short --branch
```

Action:

- If `peer_live=false`, holder PID/PPID are absent, and `sidecar_stale=true`, release stale lease with:

```powershell
python scripts\tools\worktree_guard.py --force-release --json
```

Exit criteria:

- `workflow_doctor.py status` no longer blocks on peer lease.
- Remaining blockers are explicit: dirty tree, stale dashboard, stage bloat, drift/precommit ownership.

## Phase 1 - Approved Stage Close Only

Goal: close only the single manual-close candidate already triaged.

Approved Batch A:

- `docs/runtime/stages/2026-05-29-drift-cache-proof-of-honesty.md`

Action:

```powershell
git mv docs\runtime\stages\2026-05-29-drift-cache-proof-of-honesty.md docs\runtime\stages\archive\2026-05-29-drift-cache-proof-of-honesty.md
python scripts\tools\stage_reaper_audit.py
```

Exit criteria:

- Batch A file is archived.
- Reaper still reports `DONE_SAFE=0`.
- No contested, active, or unverifiable stage is moved.

## Phase 2 - Land The Deterministic Control Plane

Goal: make one-screen ownership/status reliable enough to replace agent guessing.

Scope:

- `scripts/tools/workflow_doctor.py`
- `tests/test_tools/test_workflow_doctor.py`
- `scripts/tools/stage_reaper_audit.py`
- `tests/test_tools/test_stage_reaper_audit.py`

Required behavior:

- one-screen status
- JSON contract
- lease stale-vs-live distinction
- raw git worktree visibility
- managed metadata visibility
- dashboard port plus heartbeat owner
- DB read-only probe and hardlink warning
- stage counts without moving files by default
- drift command recommendations without running drift by default

Exit criteria:

```powershell
python -m pytest tests\test_tools\test_workflow_doctor.py tests\test_tools\test_stage_reaper_audit.py -q
ruff check scripts\tools\workflow_doctor.py tests\test_tools\test_workflow_doctor.py scripts\tools\stage_reaper_audit.py tests\test_tools\test_stage_reaper_audit.py
git diff --check
python scripts\tools\workflow_doctor.py status
python scripts\tools\stage_reaper_audit.py
```

## Phase 3 - Worktree Ownership Closeout

Goal: resolve dirty peer work by owner and safety surface.

Batches:

- Drift/precommit owner: `C:\Users\joshd\.codex\worktrees\precommit-drift-speed`
- Lease guard owner: `C:\Users\joshd\canompx3-lease-guard-fix`
- Live/dashboard owner: `C:\Users\joshd\canompx3-live-launch-tokyo`
- Main checkout workflow tooling: `C:\Users\joshd\canompx3`

Actions:

- inspect each worktree with `git status --short --branch`
- classify as active, stale, merge-ready, or abandon-ready
- do not edit peer work from main
- merge or archive only after focused verification

Exit criteria:

- no dirty worktree touches the same protected file family as another active worktree
- stale metadata is reported or cleaned by explicit approval
- detached worktrees are classified

## Phase 4 - Dashboard And Live Runtime Clarity

Goal: prevent live/dashboard confusion.

Checks:

```powershell
python scripts\tools\workflow_doctor.py dashboard
Get-NetTCPConnection -LocalPort 8080 -State Listen
```

Rules:

- port open is not health
- stale heartbeat is WARN/BLOCK depending on launch intent
- `SIGNAL` runtime is not live execution
- no `/api/action/start`, broker order, webhook, kill, flatten, or live path is exercised by workflow tooling

Exit criteria:

- status shows owner worktree, mode, profile, heartbeat age, and next command
- stale dashboard state has one operator action: restart/stop/confirm owner, not guess

## Phase 5 - Drift And Precommit Reliability

Goal: make drift/precommit cost visible and bounded. Detailed audit and tiered implementation plan: `docs/plans/active/2026-06/2026-06-04-drift-precommit-speed-audit.md`.

Owner branch:

- `codex/precommit-drift-speed`

Checks:

```powershell
git -C C:\Users\joshd\.codex\worktrees\precommit-drift-speed status --short --branch
python scripts\tools\workflow_doctor.py drift
python scripts\tools\profile_check_drift.py
python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory
```

Exit criteria:

- fast drift probe has bounded timing evidence
- docs-only commits have a measured sub-1-3s hot path
- small Python/tooling commits have a measured sub-3-8s path where possible
- hook activation state is explicit (`core.hooksPath=.githooks` or a loud fix command)
- precommit does not hide background failures
- full drift remains explicit, not default status
- heavyweight drift checks have typed ownership/stage/path-scope metadata before any commit-time skip is introduced
- pre-push or CI carries every heavyweight check moved out of commit time

## Phase 6 - Remaining Stage Disposition

Goal: close stages by evidence, not count.

Rules:

- `CLOSED_ALREADY`: human-approved archive batch only
- `UNVERIFIABLE`: recover or write explicit no-artifact decision
- `BLOCKED_BY_PEER`: wait for owner branch/worktree resolution
- `PARK`: move only if the project wants a parked-stage namespace
- `KEEP_ACTIVE`: leave active
- `SPLIT`: split only when owner and touched files are clear

Exit criteria:

- every remaining root stage has exactly one owner/action class
- no live/dashboard/drift/capital stage is closed while peer work is dirty
- `stage_reaper_audit.py` remains report-only by default

## What This Plan Does Not Authorize

- trading logic changes
- schema changes
- live execution changes
- broker/webhook/order calls
- DB writes
- changing `gold.db`
- hidden cleanup daemons
- always-on agent/Codex review
- broad refactors

## Next Command

After Phase 1:

```powershell
python scripts\tools\workflow_doctor.py status
```
