---
paths:
  - ".github/**"
  - "HANDOFF.md"
  - "docs/plans/**"
  - "docs/handoffs/**"
  - ".claude/rules/branch-discipline.md"
---
# Branch Discipline — Base-Branch Integrity

**Load-policy:** auto-injected when touching PR-adjacent files (HANDOFF, plans, .github). For pure `git`/`gh` CLI work without file edits, read this rule on demand before branching. A future enhancement could route injection via a UserPromptSubmit hook keyed on "PR"/"branch"/"push"/"commit" in the user prompt.


**Authority:** applies to every branch created for a PR. Triggered by `2026-04-18-a88505cd-bleed-into-doctrine-PR` incident (HANDOFF § 2026-04-18 governance).

---

## The rule

**Never branch a PR from local `main` without verifying it matches `origin/main`.**

Specifically:
- Docs-only PRs, doctrine updates, small refactors: branch from `origin/main`, not local `main`.
- Research branches that intend to carry un-merged local commits forward: acceptable to branch from local `main`, but be explicit about what's being carried.

## Canonical procedure for a new docs/doctrine PR

```bash
git fetch origin
git checkout -b <branch-name> origin/main        # NOT local main
# make edits
git add ...
git commit -m "..."
git push -u origin <branch-name>
# before opening PR:
git log --oneline origin/main..<branch-name>     # must show ONLY the intended commits
git diff --stat origin/main <branch-name>        # must match expected file/line scope
# open PR only if scope matches
gh pr create --base main --head <branch-name> ...
```

## Pre-PR diff-scope verification (hard rule)

Before running `gh pr create`, ALWAYS run:

```bash
git log --oneline origin/main..HEAD
git diff --stat origin/main HEAD
```

Compare both outputs to your expected scope. If either shows more than expected:
- **STOP.** Do not open the PR.
- Diagnose: either rebase/reset the branch, or explicitly widen the PR scope description.
- This catches unpushed-local-commit bleed before it reaches review.

## What NOT to do

- `git checkout main && git checkout -b new_branch` without `git fetch origin` first.
- Assume local `main` == `origin/main` because "I pushed recently" — verify.
- Rely on `gh pr create` to compute the "right" diff — it computes diff vs base, and the base is origin's state. If local has unpushed commits, they get bundled.
- Branch from a branch that's ahead of its upstream for a PR targeting `main`.

## When it's OK to branch from local main

If you INTEND to carry local unpushed commits forward in the PR (e.g., unifying two staged commits into one PR), explicit is fine — name the branch to reflect the scope and document in the PR body which local commits are included.

## Reference

Incident report: `HANDOFF.md` § "2026-04-18 governance — merge side-effect recorded".
