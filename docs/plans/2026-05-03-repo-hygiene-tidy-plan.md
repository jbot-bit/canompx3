---
status: active
owner: codex
last_reviewed: 2026-05-03
superseded_by: ""
---

# Repo Hygiene Tidy Plan (Phased)

## Objective

Cleanly converge repo state so there are no ambiguous leftovers across git state, queue/baton state, stage files, and planning artifacts.

This plan is **operational hygiene only** (not strategy logic changes).

## MEASURED Snapshot (2026-05-03 UTC)

### Git surface

- Branch: `work` (`git status --short --branch`).
- No local changes (`git status --short --branch`).
- No remotes configured (`git remote -v` returned empty).
- No local merge/rebase/cherry-pick in progress (`MERGE_HEAD`, `REBASE_HEAD`, etc. absent).
- No stashes (`git stash list` empty).
- Only local branch is `work`; no local unmerged branches (`git branch --no-merged` empty).

### Coordination/state surface

- `HANDOFF.md` drift warning is active vs canonical queue render.
- Canonical queue has 3 open + stale items:
  - `mes_q45_exec_bridge`
  - `pr48_mgc_shadow_observation`
  - `track_d_mnq_comex_settle_gate0_runner_design`
- Active stage files warning present (2 flagged by preflight/system brief).
- Runtime stages directory currently contains 4 stage docs, including in-progress CRG/stability items.

### Health telemetry surface

- `project_pulse --fast` reports two FIX NOW items:
  1. Missing canonical DB file (`/workspace/canompx3/gold.db`)
  2. Missing Criterion 11 survival report (`python -m trading_app.account_survival --profile topstep_50k_mnq_auto`)
- `project_pulse --fast` also reports queue/handoff drift and stale queue item verification.

### Plan/doc debt surface

- Active planning docs include open token/CRG plans in `docs/plans/active/2026-04/`.
- TODO-ledger style runtime doc exists: `docs/runtime/code_review_fixes.md` (all TODO statuses).
- Some active/legacy docs reference historic stash names, but no live stash exists now.

---

## Phase 0 — Freeze + Truth Baseline (same day)

**Goal:** ensure we are not tidying against stale assumptions.

1. Run baseline checks and save outputs to a dated hygiene note under `docs/runtime/`.
   - `git status --short --branch`
   - `git branch -vv`
   - `git stash list`
   - `python3 scripts/tools/work_queue.py status`
   - `python3 scripts/tools/project_pulse.py --fast --format markdown`
2. If DB path is intentionally external, record canonical override policy in one place (not chat-only).
3. Do not close or archive anything before this baseline is captured.

**Exit criteria:** one committed snapshot note with timestamps + command outputs summary.

---

## Phase 1 — Cross-Tool Alignment (baton/queue/stage parity)

**Goal:** make all coordination surfaces say the same thing.

1. Regenerate baton from queue:
   - `python3 scripts/tools/work_queue.py render-handoff --write`
2. Re-verify queue stale items and either:
   - refresh `last_verified_at` with measured note, or
   - close/supersede if obsolete.
3. Resolve stage-file drift:
   - for each `docs/runtime/stages/*.md`, set explicit state: `active`, `closed`, or `superseded`; if closed, move to archive location or mark with closure footer.
4. Ensure `HANDOFF.md` next steps exactly match open queue order.

**Exit criteria:**
- `system_brief` and `project_pulse` no longer warn that HANDOFF drifted from queue.
- Stage warning count reduced to intentional active stages only.

---

## Phase 2 — Half-Actioned Work Triage (decision cleanup)

**Goal:** remove ambiguous "started but not governed" work.

1. Build a triage table (single doc) for all open artifacts:
   - open queue items
   - active stage docs
   - active plan docs
   - runtime TODO ledgers
2. Force one decision per artifact:
   - `EXECUTE_NOW` (date + owner)
   - `PARK_WITH_TRIGGER` (clear re-open condition)
   - `CLOSE_AS_REDUNDANT` (what superseded it)
3. For items mentioning stash-only or branch-only state, convert references into either:
   - current reproducible command path, or
   - explicit historical note (non-actionable).

**Exit criteria:** no ambiguous ownership; every open artifact has owner + next action + freshness date.

---

## Phase 3 — Redundancy & Documentation Hygiene

**Goal:** reduce duplicate or stale instruction surfaces.

1. Consolidate overlapping plan docs by adding `superseded_by` chains.
2. Move old but valuable plans to `docs/plans/archive/` with short preservation rationale.
3. Convert broad TODO docs (e.g., `docs/runtime/code_review_fixes.md`) into queue-linked tracked items or close them if stale.
4. Run doc hygiene grep for stale placeholders (`NOT YET RUN`, `TODO`, `TBD`, `WIP`) and explicitly classify each hit as:
   - intentional,
   - converted to tracked queue work,
   - removed.

**Exit criteria:** no orphan TODO registries disconnected from queue/ledger.

---

## Phase 4 — Verification + Professional Finish

**Goal:** certify tidy state and keep it tidy.

1. Re-run:
   - `python3 scripts/tools/session_preflight.py --context codex-wsl`
   - `python3 scripts/tools/system_brief.py --format text`
   - `python3 scripts/tools/project_pulse.py --fast --format markdown`
2. Add a monthly hygiene cadence item (queue entry) to re-check:
   - stash presence
   - handoff drift
   - stale queue items
   - active stage overflow
3. Add a compact “repo hygiene contract” snippet in docs/runtime or governance references (single source).

**Exit criteria:** preflight/pulse warnings limited to intentional items only.

---


## Sonnet-Executable Mode (Low-Context Runbook)

Yes — this cleanup can be executed by Sonnet safely if we keep it deterministic and command-driven.

### Guardrails for Sonnet runs

- Use queue-backed state only (`docs/runtime/action-queue.yaml` + `work_queue.py`).
- Prefer generated baton output over hand-edited baton text.
- Keep each phase in its own commit (one phase = one atomic diff).
- Never close/supersede queue items without adding a measured note.

### Minimal operator prompt (copy/paste)

```text
Run Phase <N> of docs/plans/2026-05-03-repo-hygiene-tidy-plan.md exactly as written.
Use only canonical queue-backed surfaces.
Show command output before mutating files.
Commit only files required for that phase.
```

### Deterministic command set per phase

```bash
# Baseline snapshot
python3 scripts/tools/work_queue.py status
python3 scripts/tools/system_brief.py --format text
python3 scripts/tools/project_pulse.py --fast --format markdown || true

# Baton alignment
python3 scripts/tools/work_queue.py render-handoff --write
python3 scripts/tools/work_queue.py status

# Verification pass
python3 scripts/tools/system_brief.py --format text
python3 scripts/tools/project_pulse.py --fast --format markdown || true
```

### Done definition for Sonnet execution

- Phase artifact committed.
- Queue state and baton state are consistent.
- Any remaining warnings are explicitly classified as intentional or blocked.

---

## Implementation Order (Recommended)

1. **Phase 1 first** (alignment) — highest risk reducer.
2. **Phase 2 next** (ownership clarity).
3. **Phase 3** (doc redundancy cleanup).
4. **Phase 4** (verification + cadence lock-in).

---

## Decision Rules (to keep this clean)

- If an artifact is open but lacks `next_action` + date → it is not "active"; park or close it.
- If two docs prescribe next steps, queue-backed state wins.
- If a task is important enough to keep, it must exist in `docs/runtime/action-queue.yaml`.
- Stashes are emergency buffers, not planning containers; any stash-backed intent must be promoted into committed docs/queue within 24h.

