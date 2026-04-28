---
slug: pr-open-helper
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-28
updated: 2026-04-28
task: Build pr_preflight.sh + pr_open.sh + /open-pr slash command + docs/pr_bodies/ to replace ~3k-token-per-PR copy-paste blocks with one repo-local helper. Enforces branch-discipline.md diff-scope gate, stacked-base detection, dirty-tree abort, protected-path scan; defaults to dry-run; never uses gh pr merge --auto.
---

# Stage 1: PR-open helper (single-stage infra)

## Scope Lock

- scripts/tools/pr_preflight.sh
- scripts/tools/pr_open.sh
- .claude/skills/open-pr/SKILL.md
- docs/pr_bodies/README.md
- docs/pr_bodies/chore-close-landed-stages-2026-04-28.md
- docs/pr_bodies/chore-pr-open-helper.md
- docs/pr_bodies/research-2026-04-28-phase-d-mes-europe-flow-pathway-b.md
- docs/runtime/stages/pr-open-helper.md

## Blast Radius

- All four deliverables are NEW files. No existing production code, research, doctrine, or trading logic is touched.
- `scripts/tools/` already contains shell helpers (no precedent collision); these add 2 more.
- `.claude/commands/` already exists with several command markdowns; this adds one more.
- `docs/pr_bodies/` is a NEW directory; convention introduced via README.md.
- The slash command `/open-pr` only invokes the helper — does not write back to the repo.
- Helpers default to dry-run (no remote action); pushing requires explicit `--push`.
- Hard-coded NEVER touched: trading_app/holdout_policy, trading_app/live/*, lane_allocation,
  validated_setups, live_config, pipeline/cost_model, pipeline/paths.

## Why this stage exists

User feedback 2026-04-28: "i dont like wasting tokens and it seems lots of work is done around gits.
maybe streamlinining it iwihout losing rigor could help." Today's session generated ~3k output
tokens per PR via inline copy-paste command blocks; the same workflow can be a 10-character `/open-pr`
invocation backed by a single repo helper.

Hard constraints (per user-approved scope):
1. Dry-run default; require `--push` to actually push.
2. Never use `gh pr merge --auto` (per `memory/feedback_gh_pr_merge_auto_silent_register.md`).
3. Show `git log base..HEAD` + `git diff --stat base..HEAD` before any push (per
   `.claude/rules/branch-discipline.md` HARD RULE — pre-PR diff-scope verification).
4. Stacked-base detection — abort with STACK / RETARGET options when `--base` is not on origin
   (per `memory/feedback_codex_stack_collapse.md`).
5. No trading/research logic touched (helpers are workflow infra, not pipeline).
6. F12 (PostCompact) decision remains a separate workstream.

## Acceptance

- `bash scripts/tools/pr_preflight.sh --help` prints usage cleanly.
- Dry-run on `chore/close-landed-stages-2026-04-28` produces correct diff-scope output and exits 0.
- Dry-run on `research/2026-04-28-phase-d-mes-europe-flow-pathway-b` triggers stacked-base abort
  (because base is not on origin) with STACK/RETARGET options surfaced.
- `gh` and `git` invocations only — no third-party deps introduced.
- Pre-commit gauntlet 8/8 PASS.
- `python pipeline/check_drift.py` PASS.

## Out of scope

- Auto-merge wiring — explicitly forbidden.
- Multi-PR / stacked-PR creation in one shot — kept simple per "tiny commands" doctrine.
- Updating CLAUDE.md or auto-skill-routing.md — defer until helper proves out.
- Migrating other workflows (commit, branch-creation) to similar helpers — separate stage if wanted.

## Verification

After this stage lands:
- Open PR for `chore/close-landed-stages-2026-04-28` via `/open-pr --base origin/main --push` (separate user GO).
- Open PR for `research/2026-04-28-phase-d-mes-europe-flow-pathway-b` via the same helper after deciding STACK or RETARGET (separate user decision).
