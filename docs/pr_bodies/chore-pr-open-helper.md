## Scope

Workflow infra: replace ~3k-token-per-PR copy-paste blocks with one repo-local helper + skill. **No production / trading / research / holdout / canonical-truth files touched.**

## Why

User feedback 2026-04-28: "i dont like wasting tokens and it seems lots of work is done around gits. maybe streamlining it without losing rigor could help."

Today's session generated ~3k output tokens per PR via inline command blocks; the same workflow is now a 10-character `/open-pr` invocation backed by a single helper.

## Commits

Single commit (this branch is single-stage infra).

## Files added

- `scripts/tools/pr_preflight.sh` — pure verification, no push, no PR create. Diff-scope, stacked-base, dirty-tree, protected-path checks. Exit codes by failure mode.
- `scripts/tools/pr_open.sh` — full helper. Dry-run by default; `--push` gate; body resolution (`--body-file` → `docs/pr_bodies/<slug>.md` → `gh --fill`); never `--auto`. Delegates preflight.
- `.claude/skills/open-pr/SKILL.md` — `/open-pr` skill that wraps the helper and surfaces a concise result.
- `docs/pr_bodies/README.md` — convention + naming + body-resolution order.
- `docs/pr_bodies/chore-close-landed-stages-2026-04-28.md` — body for branch A (already merge-ready)
- `docs/pr_bodies/chore-pr-open-helper.md` — body for THIS PR (self-referential — also exercises the convention)
- `docs/pr_bodies/research-2026-04-28-phase-d-mes-europe-flow-pathway-b.md` — body for branch B (already merge-ready)
- `docs/runtime/stages/pr-open-helper.md` — stage spec

## Hard rules baked into the helper (per user-approved scope)

1. **Dry-run default** — `--push` required to actually push.
2. **Never `gh pr merge --auto`** — per `memory/feedback_gh_pr_merge_auto_silent_register.md`.
3. **Show `git log base..HEAD` + `git diff --stat base..HEAD`** before any push — per `.claude/rules/branch-discipline.md` HARD RULE.
4. **Stacked-base abort** — when `--base` is not on origin, abort with STACK / RETARGET options. Per `memory/feedback_codex_stack_collapse.md`.
5. **Protected-path scan** — soft notice on `pipeline/check_drift|paths|dst|cost_model|asset_configs`, `trading_app/prop_profiles|outcome_builder|risk_manager|live/`, `live_config`, `lane_allocation`, `validated_setups`. **Hard-block** abort on `trading_app/holdout_policy.py`, `trading_app/live/(execution_engine|risk_manager|order_router).py`, `live_config.py`, `lane_allocation.json`.
6. **No trading/research logic touched** — helpers are workflow infra only.

## Evidence

```
$ bash scripts/tools/pr_preflight.sh --help        # usage prints
$ bash scripts/tools/pr_open.sh --help             # usage prints
$ bash scripts/tools/pr_open.sh --base origin/main # dry-run, shows scope, exits 0
```

Dry-run validation against the two pre-existing merge-ready branches (`chore/close-landed-stages-2026-04-28` and `research/2026-04-28-phase-d-mes-europe-flow-pathway-b`) included — see commit message + dry-run output preserved in CI logs.

## Disconfirming Checks

- Helper does NOT auto-push under any flag combination — `--push` is the only path that touches origin.
- Helper does NOT enable auto-merge — explicit follow-up `gh pr merge --merge` is the user's responsibility.
- Stacked-base check uses `git ls-remote --heads origin <base>` — caught the research/phase-d → research/mnq-unfiltered-high-rr-family stale-base case correctly in the integration test.
- Body resolution falls back to `gh pr create --fill` when no body file present — chore branches with good commit messages don't need explicit bodies.

## Grounding

- `.claude/rules/branch-discipline.md` — pre-PR diff-scope verification HARD RULE (lines 33-37)
- `memory/feedback_no_push_to_other_terminal_branch.md` — never auto-push without explicit OK
- `memory/feedback_codex_stack_collapse.md` — stacked-PR collapse risk; abort with options
- `memory/feedback_gh_pr_merge_auto_silent_register.md` — `gh pr merge --auto` silent-register risk
- `memory/feedback_gha_merge_ref_staleness.md` — only `git push` to head branch refreshes GHA `pull_request` ref

## Token math

| Workflow | Output tokens per PR |
|---|---|
| Before (manual copy-paste blocks) | ~3000 |
| After (`/open-pr --push`) | ~80 (URL + verdict) |
| Break-even | ~2 PRs |

## Not done by this PR

- No CLAUDE.md or skill-routing changes — defer until helper proves out.
- No multi-PR / stacked-PR creation in one shot — kept simple per "tiny commands → user; multi-step → helper" doctrine.
- No commit / branch-creation helpers — separate scope if wanted.
- No auto-merge — forbidden per project memory.
- F12 (PostCompact reinject path) is a separate workstream.
