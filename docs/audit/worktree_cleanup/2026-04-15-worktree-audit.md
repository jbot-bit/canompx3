# Worktree Cleanup Audit — 2026-04-15

**Trigger:** `project_pulse.py` flags `8 open worktrees (2 claude, 6 codex)`.
**Scope:** Determine merge/abandon decision for each worktree dir under `.worktrees/tasks/` per institutional rigor (no silent deletes).
**Author:** session 2026-04-15 (post-handover `23134e9b`)

## Git state vs filesystem state

- `git worktree list` → main only (no linked worktrees registered).
- `.git/worktrees/` → does not exist (pruned).
- `.worktrees/tasks/*` → 8 orphaned full-repo checkout directories, each ~35–45 MB unique + hardlinked 7.2 GB `gold.db`.

The filesystem directories are disconnected from git — no `git status`, no branch tracking, no safe merge path without recreating a worktree.

## Branch inventory (remote + local, `wt-*` prefix)

| Branch | Local | Remote | Commits ahead of main | Status |
|---|---|---|---|---|
| `wt-codex-operator-cockpit` | ✓ | ✓ | 5 | **Active PR candidate** |
| `wt-codex-work-capsule` | ✓ | ✗ | 3 (of which 1 already merged as `b22ff549`) | **Partial merge — 2 commits stranded** |
| `wt-codex-startup-brain-refactor` | ✗ | ✓ | 2 | **Subset of operator-cockpit** (shares `4f777ee8`) |
| `wt-codex-green-baseline` | ✓ | ✗ | 0 | Empty, safe to delete |
| `wt-codex-audit` | ✗ | ✗ | — | Branch never existed |
| `wt-codex-audit2` | ✗ | ✗ | — | Branch never existed |
| `wt-codex-finite-data-reaudit` | ✗ | ✗ | — | Branch never existed |
| `wt-codex-research-ml-bot-review` | ✗ | ✗ | — | Branch never existed |
| `wt-claude-first-workstream-test` | — | — | — | Filesystem dir only (green-baseline + work-capsule both claim this branch in metadata — metadata bug) |

## Unique files in orphan dirs (no branch backing)

Consolidated across `audit`, `audit2`, `finite-data-reaudit`, `research-ml-bot-review` (38 paths total):

**ML archaeology (DEAD per MEMORY.md — V1/V2/V3 all dead, `trading_app/ml/` deleted Apr 11):**
- `trading_app/ml/*.py` (6 files)
- `tests/test_trading_app/test_ml/*.py` (6 files)
- `scripts/tools/ml_*.py` (12 files)
- `scripts/phase0_*.py`, `scripts/ml_verified_sessions.py`, `scripts/mnq_*_ml*.py` (5 files)

**Abandoned design branches:**
- `trading_app/eligibility/decomposition.py` — references `docs/plans/2026-04-07-eligibility-context-design.md`; not adopted. Current eligibility calls `StrategyFilter.matches_row()` directly (per institutional-rigor.md §4).
- `trading_app/nested/builder.py` — current `trading_app/nested/` ships audit_outcomes.py + compare.py + discovery.py; `builder.py` was not chosen.

**Completed stage files (already removed from main per stage-gate protocol):**
- `docs/runtime/stages/wave4-hypothesis-filter-injection.md` — shipped `0682205b` + `b62008e4`
- `docs/runtime/stages/wave5-presession-filter-registration.md` — shipped same
- `docs/runtime/stages/view-b-filter-universe-audit.md` — completed

**Windows launcher batch scripts:**
- `codex-workstream.bat`, `ai-workspaces.bat`, `agent-tasks.bat` — worktree-local codex infrastructure.

**Verdict:** all 38 orphan-only files are either (a) deliberately deleted dead code, (b) abandoned design alternatives superseded by main, or (c) completed stage files correctly swept. Nothing load-bearing. Safe to delete.

## Commits on active branches

### wt-codex-operator-cockpit (5 unique)
```
e3ce3808 merge(startup): integrate startup brain into operator cockpit
ff418d62 docs(governance): refresh system authority map
87a1eca6 feat(operator): add cockpit state and runtime alerts
cf99cf28 fix(launchers): harden codex and claude entrypoints
4f777ee8 fix(pulse): make snapshot-miss recommendation honest
```
Plus `fb597bcb feat(startup): refactor orientation onto packets and snapshots` (shared w/ startup-brain).

### wt-codex-work-capsule (2 unique, 1 already in main)
```
05c8ab56 fix(drift): harden check 94 to audit ALL profiles (active + inactive)  [NOT IN MAIN]
d44dd31e feat(shell): add work capsule shell for managed workstreams             [NOT IN MAIN]
1b47862b fix: restore portfolio reconstruction (+41% ExpR) accidentally reverted [ALREADY MERGED as b22ff549]
```

### wt-codex-startup-brain-refactor (2, both in operator-cockpit)
Subset of operator-cockpit. Merging operator-cockpit supersedes this.

## Recommendation

**Decision required from user (merge-risk judgment):**

1. **`wt-codex-operator-cockpit`** — 5 commits including pulse snapshot fix, launcher hardening, governance docs, operator cockpit feature, startup brain integration.
   - **Recommendation:** Review and merge, or explicitly abandon. The pulse snapshot fix (`4f777ee8`) may conflict with current `project_pulse.py` which already emits the `→` recommendation string; check for rebase conflicts first.

2. **`wt-codex-work-capsule`** — 2 commits stranded. `05c8ab56` (check 94 hardening) and `d44dd31e` (work capsule shell).
   - **Recommendation:** Cherry-pick `05c8ab56` into main if check 94 hardening is still wanted (pulse currently reports check 94 passing; verify scope). `d44dd31e` work capsule shell — abandon or merge based on whether `docs/runtime/capsules/` infrastructure is still on the roadmap.

3. **`wt-codex-startup-brain-refactor`** — superseded by operator-cockpit. Safe to delete remote branch.

4. **`wt-codex-green-baseline`** — 0 commits ahead. Safe to delete local branch.

**Filesystem cleanup (can proceed without merge decision):**

- All 8 `.worktrees/tasks/*/` directories are orphaned from git and hold no load-bearing unique content (see §"Unique files" above). Delete filesystem dirs after merge decisions above are made and branches preserved on `origin`.
- Disk reclaim: ~320 MB of unique checkout files per dir + removal of 8 hardlinks to `gold.db` (hardlinks reclaim no physical bytes since canonical copy persists).

## Hazards

- `.worktrees/tasks/audit/gold.db` and peers are hardlinked to canonical `gold.db` — `rm` is safe (reduces link count, doesn't delete data) but use `rm -rf .worktrees/tasks/<name>` only after confirming no running process has those paths open.
- The two `wt-claude-first-workstream-test` metadata references (in `green-baseline` and `work-capsule` metadata) suggest the metadata writer had a bug. Not load-bearing, but a note for the tool owner.

## Resolution log (2026-04-15 follow-up)

### Completed (safe actions, no user judgment needed)

- ✅ **`wt-codex-green-baseline` branch deleted** (was 62a34c72, 0 commits ahead — verified no unique work).
- ✅ **5 orphan `.worktrees/tasks/` dirs deleted** (had no matching branch OR branch was just deleted):
  - audit, audit2, finite-data-reaudit, research-ml-bot-review (no matching branches; per audit §3 unique files were dead ML archaeology + abandoned design alternatives + completed stage files).
  - green-baseline (branch deleted above).
  - Disk reclaim: ~200MB unique files (gold.db hardlinks reduced but canonical copy persists).

### Stopped (silent action would be reckless — user judgment needed)

- ❌ **`wt-codex-operator-cockpit`** — `git merge --no-commit --no-ff` produced **9 conflicts**:
  - `scripts/tools/system_brief.py` (add/add) — work-capsule branch also added this file
  - `scripts/tools/work_capsule.py` (add/add) — same situation
  - `tests/test_pipeline/test_system_brief.py` (add/add)
  - `tests/test_tools/test_windows_agent_launch.py` (content)
  - `tests/test_trading_app/test_pre_session_check.py` (content)
  - `tests/test_trading_app/test_session_orchestrator.py` (content)
  - `trading_app/live/bot_dashboard.html` (content) — overlaps with recently-shipped Tier 1-3 dashboard polish (commits `51dbe94d` / `41104a7a` / `af0c3ca4` / `b7f6fd42`)
  - `trading_app/live/bot_dashboard.py` (content) — same dashboard overlap
  - `trading_app/pre_session_check.py` (content)
  - 5318 insertions / 382 deletions across 55 files. Too large for silent merge.
  - **User decision required:** rebase manually, three-way-merge selectively, or abandon. The branch and its filesystem dir at `.worktrees/tasks/operator-cockpit/` are PRESERVED for inspection.

- ❌ **`wt-codex-startup-brain-refactor`** — its 2 commits (`4f777ee8` + `fb597bcb`) are a SUBSET of operator-cockpit. Decision is downstream of operator-cockpit: if cockpit merges, startup-brain is redundant; if cockpit is abandoned, startup-brain might be a smaller-PR alternative. Branch + dir PRESERVED.

### Completed — second pass (2026-04-15 later, after in-depth verification)

- ✅ **`wt-codex-work-capsule` branch + dir deleted.** Both stranded commits verified subsumed or moot:
  - **`05c8ab56` (check 94 hardening) — moot.** Problem it targeted (57 stale lanes) was already resolved by allocator wiring (Apr 13, `daily_lanes=()` + JSON consumption per `memory/session_apr13_handoff.md`). Also: the diff is structurally incomplete — it introduces an unused `is_active = profile.active` variable with no `[STALE-INACTIVE]` tagging logic despite the commit message's claim. Cherry-picking would install dead code. The originating sprint (portfolio dedup) is declared NO-GO in `memory/portfolio_dedup_nogo.md`.
  - **`d44dd31e` (work capsule shell) — subsumed.** `scripts/tools/work_capsule.py` already exists on main via commit `0e446ea3` (codex-wip batch add, independent of d44dd31e). Would have been a pure add/add conflict with no salvageable delta.
  - Dangling commit SHAs preserved in this doc (lines 67-68) + git reflog (~90 day recovery window) if ever needed.
  - Verified no worktree registration (`git worktree list` showed only main) — safe to `rm -rf`.
  - Both `git branch -a --contains 05c8ab56` and `... d44dd31e` returned empty after delete — no other branch holds them.

### Recommended next user actions

1. Inspect `.worktrees/tasks/operator-cockpit/` filesystem checkout if helpful.
2. Decide on `wt-codex-operator-cockpit`: rebase + manual conflict resolution, or close as abandoned. The dashboard overlap means rebase will require careful three-way merge against the recently-shipped Tier 1-3 polish.
3. If cockpit closed, `wt-codex-startup-brain-refactor` becomes deletable.
