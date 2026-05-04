---
name: ship
description: Ship a branch — alias for /open-pr. Opens a PR with full preflight gating (branch-discipline diff-scope, stacked-base detection, dirty-tree abort, protected-path scan). Dry-run by default; --push to actually push. Never uses gh pr merge --auto. Use when user says "ship", "ship it", "/ship", or asks to wrap up + PR a ready branch.
allowed-tools: Bash
---

Ship the current branch via `scripts/tools/pr_open.sh`: $ARGUMENTS

`/ship` is a memorable alias for `/open-pr`. Identical behaviour and constraints — see `.claude/skills/open-pr/SKILL.md` for the full spec.

## How to use this skill

1. Run the helper with `$ARGUMENTS` passed through:

   ```bash
   bash scripts/tools/pr_open.sh $ARGUMENTS
   ```

2. Common invocations:
   - `/ship`                            → dry-run vs `origin/main`
   - `/ship --push`                     → push + open PR (default body resolution)
   - `/ship --base origin/main --push`  → explicit base + push
   - `/ship --base research/foo --push` → stacked-base (helper aborts if base not on origin)
   - `/ship --body-file <path> --push`  → explicit body file
   - `/ship --draft --push`             → draft PR

3. Surface ONLY:
   - The `===== PR-OPEN PLAN =====` block
   - Any `BLOCKED [N]:` line on abort
   - The PR URL on success
   - One-line note that auto-merge is NOT enabled

4. Suppress the full `git log` / `git diff --stat` walls unless the helper aborts (then surface them so the user can diagnose).

## Hard rules baked into the helper

- Default is **dry-run** — `--push` required to push.
- Stacked-base abort surfaces STACK / RETARGET options when `--base` is not on origin.
- Never `gh pr merge --auto`.
- Pre-PR diff-scope shown (`branch-discipline.md` HARD RULE).
- Hard-block protected paths (`trading_app/holdout_policy`, `trading_app/live/(execution_engine|risk_manager|order_router)`, `live_config.py`, `lane_allocation.json`) abort with exit 4.

## After shipping

The merge step is the user's. Do NOT enable auto-merge. Do NOT issue the merge yourself unless the user explicitly says "merge it".

```bash
gh pr view <number>                                # status + checks
gh pr merge <number> --merge --delete-branch       # manual after CI green
```

## Reference

- Helper: `scripts/tools/pr_open.sh`
- Preflight: `scripts/tools/pr_preflight.sh`
- Bodies: `docs/pr_bodies/<branch-slug>.md` (auto-discovered)
- Convention: `docs/pr_bodies/README.md`
- Sister skill: `.claude/skills/open-pr/SKILL.md`
