# PR Body Convention — `docs/pr_bodies/`

This directory holds **committed PR body markdown files** that
`scripts/tools/pr_open.sh` and the `/open-pr` skill auto-discover at PR-open
time.

## Why bodies live in the repo

- Reviewable in advance — can be edited / read like any other file.
- Survive context compaction — body is not generated from chat tokens at
  open time.
- Reusable — re-running `/open-pr --push` after a force-push or re-target
  uses the same body without re-keying.
- Audit-friendly — `git blame` shows who wrote each body and when.

## Naming convention

```
docs/pr_bodies/<branch-slug>.md
```

where `<branch-slug>` is the branch name with `/` replaced by `-`.

| Branch | File |
|---|---|
| `chore/close-landed-stages-2026-04-28` | `chore-close-landed-stages-2026-04-28.md` |
| `research/2026-04-28-phase-d-mes-europe-flow-pathway-b` | `research-2026-04-28-phase-d-mes-europe-flow-pathway-b.md` |
| `chore/pr-open-helper` | `chore-pr-open-helper.md` |

`scripts/tools/pr_open.sh` derives the slug from `git rev-parse --abbrev-ref HEAD`
and looks up `docs/pr_bodies/<slug>.md` automatically.

## Body resolution order

1. `--body-file <path>` (explicit override)
2. `docs/pr_bodies/<branch-slug>.md` (auto-discovered)
3. `gh pr create --fill` (commit messages — fine for trivial chore branches)

## What a good body has

Borrowed from `.claude/rules/branch-discipline.md` and the
`code-review:code-review` skill PR_REQUIRED_SECTIONS:

- `## Scope` — one short paragraph: what this PR does and what it does NOT touch
- `## Commits` — bullet list of commits with short rationale
- `## Files changed` — categorized (production / docs / tests / infra)
- `## Evidence` — drift / tests / gauntlet output snippets
- `## Disconfirming Checks` — what could go wrong + what the PR did to rule it out
- `## Grounding` — citations to rules / memory / literature when relevant
- For research PRs: include verdict, gates, kill-criteria results, theory citations

## What NOT to put in a body

- Secrets / API keys / DB paths
- Internal URLs that won't render for reviewers
- Speculative future work — link out to plan docs instead
- Long copy-paste from upstream diffs — `gh` already renders the diff
