# canompx3_autopilot_v1 — headless self-driving task runner

Complete a whole task while you're away from the PC: the runner makes its own
safe decisions, runs tests, reviews its own diff, does one repair pass, commits
to a branch, and reports — while **never** touching capital/live/schema paths,
and **pausing-to-flag** (not silently doing) anything risky.

## What it is

A Claude-Code-native pair (no new dependencies, no OpenRouter HITL — that is an
OpenRouter-SDK feature that does not fit Claude Code's own agent loop):

1. A **deterministic Tier-A/B path+action classifier** (`tier_guard.py`) that the
   runner, a PreToolUse guard, and the review tool all share.
2. A **PreToolUse guard** that blocks Tier-B Edit/Write/Bash unattended, logs a
   `BLOCKED_TIER_B` journal line, and lets the run continue other Tier-A work.
3. A **runner** that ties worktree isolation + a self-decision prompt + run +
   review + one repair + branch-commit + report together.
4. A **Stop hook** that (in autopilot only) forces the review/repair pass before
   the run can end.

## The Tier-A / Tier-B contract

- **Tier A (do unattended):** docs, config, `.claude/rules`, tests,
  `scripts/autopilot/`, non-canonical scripts. Reversible on a branch.
- **Tier B (block + journal + report, never do unattended):** `trading_app/live/*`,
  broker/execution, session orchestrator, `prop_profiles.py`,
  allocator/cap/stop sizing, `live_config`, risk/kill-switch; canonical
  `pipeline/` modules (`dst.py`, `cost_model.py`, `asset_configs.py`,
  `paths.py`, `holdout_policy.py`), DB/schema/`*.db`; `git push`/`--force`/
  `reset --hard`/`clean -fd`/merge-main; `--live`/`--demo`/arming; order
  placement; `refresh_control_state`; `rm -rf`.
- **Fail-closed:** an unknown path under `pipeline/` or `trading_app/` is treated
  as Tier B. The source of truth is `.claude/rules/autonomy-contract.md` § Tier B,
  encoded in `scripts/autopilot/tier_guard.py`.

### `BLOCKED_TIER_B` semantics (headless ≠ interactive)

Headless, nobody is at the PC, so "pause to ask me" is useless. The correct
unattended semantic is: **block that one action, write a `BLOCKED_TIER_B` line
to the run journal, keep doing other Tier-A work, and list the blocker in the
final report.** When you *are* present, the existing interactive Tier-B gate
(AskUserQuestion) is unchanged.

## Example

```bash
# From inside a feature worktree (NEVER runs on main):
bash scripts/autopilot/run_autopilot.sh "Tidy the docstrings in scripts/tools/generate_trade_sheet.py and add a usage example to its module header"

# Or spin up a fresh isolated worktree first:
bash scripts/autopilot/run_autopilot.sh --worktree "..."   # creates the worktree, then re-run inside it
```

## Set-and-forget usage

1. Be on a feature branch (the runner refuses `main`/`master`).
2. Kick it off with your task string and walk away.
3. On return, read:
   - **Terminal report** — branch, start/commit SHA, commit-safe, high-risk
     files, and any Tier-B blockers.
   - **Run journal** — `docs/runtime/autopilot/<run-id>.jsonl` (every
     `BLOCKED_TIER_B` line + lifecycle events).
   - **Drift log** — `docs/runtime/autopilot/<run-id>.drift.log` (if it gated).

It **commits to a branch** only when `check_drift.py` passes AND the review
found no high-risk (Tier-B) file in the diff. It **never** pushes, merges main,
or arms anything. If it could not safely commit, changes are left in the working
tree for your review.

## Inspecting blockers on return

```bash
RUN=docs/runtime/autopilot/<run-id>.jsonl
grep BLOCKED_TIER_B "$RUN"     # what it refused to do, and why
```

Each blocker is a thing the autopilot deliberately did NOT do because it crosses
the capital/schema/live line — it is a to-do for *you*, surfaced explicitly
rather than silently skipped.

## Activation (post-merge — REQUIRED manual step)

The PreToolUse guard and the Stop-hook extension only fire once registered in
`.claude/settings.json`, and every hook here resolves to the **main checkout**
path (`C:/Users/joshd/canompx3/.claude/hooks/...`). Registering a hook whose
target file is not yet present in the main checkout breaks the active session
(the hook command errors and blocks every Edit/Write/Bash). Therefore the
registration is deliberately **NOT** committed by this build.

**To activate after this work lands in `main`:**

1. Merge this branch to `main` (so `.claude/hooks/autopilot-tier-guard.py` and
   `scripts/autopilot/tier_guard.py` exist in the main checkout).
2. Add this block to `.claude/settings.json` → `hooks.PreToolUse` (additive):

   ```json
   {
     "matcher": "Edit|Write|Bash",
     "hooks": [
       {
         "type": "command",
         "command": "python C:/Users/joshd/canompx3/.claude/hooks/autopilot-tier-guard.py",
         "timeout": 3
       }
     ]
   }
   ```

3. The Stop hook (`completion-notify.py`) is already registered — the autopilot
   block path inside it is env-gated on `AUTOPILOT_RUN=1`.

Both are env-gated, so once activated they have **zero effect on normal
interactive sessions** and fire only inside a runner-spawned `claude -p`.

## Files

| File | Role |
|---|---|
| `scripts/autopilot/tier_guard.py` | Deterministic A/B classifier (importable + CLI) |
| `scripts/autopilot/review_diff.py` | Post-build diff review, dedupe-by-hash, JSON findings |
| `scripts/autopilot/run_autopilot.sh` | The runner (run → review → repair → commit → report) |
| `.claude/hooks/autopilot-tier-guard.py` | PreToolUse guard (env-gated, blocks Tier-B) |
| `.claude/hooks/completion-notify.py` | Stop hook — forces review/repair before end (autopilot only) |
| `docs/prompts/autopilot-task-template.md` | Self-decision system prompt |
| `tests/test_autopilot/` | Classifier + guard + review unit tests |
