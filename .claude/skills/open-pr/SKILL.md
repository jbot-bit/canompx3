---
name: open-pr
description: Open a PR with full preflight gating (branch-discipline diff-scope, stacked-base detection, dirty-tree abort, protected-path scan). Dry-run by default; --push to actually push. Never uses gh pr merge --auto. Use when user says "open pr", "create pr", "raise pr", "/open-pr", or asks to push and PR a ready branch.
allowed-tools: Bash
---

Open a PR for the current branch via `scripts/tools/pr_open.sh`: $ARGUMENTS

## How to use this skill

The user invoked /open-pr. Run the helper and surface a concise result.

### 1. Decide invocation

Pass `$ARGUMENTS` straight through. Common forms:

- (no args)                    → dry-run vs `origin/main`
- `--push`                     → push + open PR (default body resolution)
- `--base origin/main --push`  → explicit base + push
- `--base research/foo --push` → stacked-base (helper will abort if base not on origin)
- `--body-file <path> --push`  → explicit body file
- `--draft --push`             → draft PR
- `--web --push`               → open in browser instead of inline

Always run from the repo root via Bash:

```bash
bash scripts/tools/pr_open.sh $ARGUMENTS
```

### 2. Surface the right slice

Show the user ONLY:

- The `===== PR-OPEN PLAN =====` block (Branch / Base / Title / Body source / Push / Web / Draft)
- Any `BLOCKED [N]:` line (preflight aborts)
- The PR URL on success
- One-line note that auto-merge is NOT enabled

Suppress the full `git log` / `git diff --stat` walls of text unless the helper aborts (then surface them so the user can diagnose).

### 3. Body resolution priority (helper enforces)

1. `--body-file <path>` (explicit)
2. `docs/pr_bodies/<branch-slug>.md` (auto-discovered; slug = branch with `/` → `-`)
3. `gh pr create --fill` (commit messages)

If you wrote multi-commit work for this branch and want a richer body than commit messages provide, drop a body file at `docs/pr_bodies/<branch-slug>.md` BEFORE invoking.

### 4. Hard rules baked into the helper

- Default is **dry-run** — `--push` required to push.
- Stacked-base abort surfaces STACK / RETARGET options when `--base` is not on origin.
- Never `gh pr merge --auto` (per `memory/feedback_gh_pr_merge_auto_silent_register.md`).
- Pre-PR diff-scope shown (per `.claude/rules/branch-discipline.md` HARD RULE).
- Hard-block paths (`trading_app/holdout_policy`, `trading_app/live/execution_engine|risk_manager|order_router`, `live_config.py`, `lane_allocation.json`) abort the open with exit code 4.

### 5. After opening

Tell the user the merge step is theirs:

```bash
gh pr view <number>                                # status + checks
gh pr merge <number> --merge --delete-branch       # merge after CI green (manual)
```

Do NOT enable auto-merge. Do NOT issue the merge yourself unless the user explicitly says "merge it".

## Reference

- Helper: `scripts/tools/pr_open.sh`
- Preflight (subset): `scripts/tools/pr_preflight.sh` (use directly when you want verify-without-push)
- Body convention: `docs/pr_bodies/README.md`
- Stage spec: `docs/runtime/stages/pr-open-helper.md`
