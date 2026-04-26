---
mode: TRIVIAL
task: Pre-commit hook auto-format restage fails on `.claude/hooks/*.py` due to gitignore-whitelist quirk
scope_lock:
  - .githooks/pre-commit
acceptance:
  - Reformatting a `.py` file that lives under `.claude/hooks/` (whitelisted via `!.claude/hooks/` in `.gitignore`) gets correctly re-staged by the auto-format step. Manually verified by `git add` working post-fix.
  - Existing behavior for files in pipeline/ trading_app/ scripts/ tests/ unchanged.
updated: 2026-04-26
---

# pre-commit auto-format restage gitignore fix

## Why

When the pre-commit hook's ruff-format step (`.githooks/pre-commit:101-118`) auto-fixes a Python file, it tries to re-stage every staged `.py` file. The branching logic at line 109-113 uses `git check-ignore` to decide whether to use `-f`:

```bash
if git check-ignore -q "$staged_file"; then
    git add -f -- "$staged_file"
else
    git add -- "$staged_file"
fi
```

For files under `.claude/hooks/` the gitignore re-include pattern (`!.claude/hooks/` after `.claude/*`) makes `git check-ignore` return "not ignored" (exit 1). So the hook takes the `else` branch and runs `git add` without `-f`. But `git add` itself warns "The following paths are ignored: .claude/hooks" and exits 1, so the file does NOT get restaged. The format diff stays as a working-tree dirty bit, the commit either skips the new format or fails outright.

This caused real friction in PR #141 (mutex tests + AQ surfacer): first commit attempt left the test file as "Changes not staged" after format. I reset and worked around. A future author hitting the same wall on a `.claude/hooks/` edit will rediscover the same trap.

## Approach

Single change at `.githooks/pre-commit:111-112`: always use `git add -f --` in the restage step. Comment explains why. Justification: at this point in the script, every file in `STAGED_FOR_FMT` was already staged for commit (line 105: `git diff --cached --name-only`) — we KNOW it should be added. `-f` only suppresses the warning; it cannot add a file that wasn't already meant to be tracked.

## Out of scope

- The `git check-ignore` itself is not buggy in our case; the underlying issue is git's own treatment of paths inside whitelisted-but-otherwise-ignored directories. We don't fix git; we just stop relying on `git check-ignore`'s answer for this edge case.
- The `[0b]` CRLF auto-renormalize block at line 60-78 has the same logical pattern (`git add --renormalize -- "$f"`); it has not been observed to fail on `.claude/hooks/*.py` because that block uses `--renormalize` which has different ignore handling. Not changing it; if it surfaces later we'll address.

## Verification

Manually tested:
- `git add .claude/hooks/session-start.py` → exits 1 with the warning.
- `git add -f -- .claude/hooks/session-start.py` → succeeds, file staged.
- After fix: pre-commit hook successfully re-stages the file post-format.
