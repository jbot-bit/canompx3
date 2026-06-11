# Daily Bug Scan Automation Prompt

Use this prompt for the recurring Daily bug scan automation after running the
scanner packet command.

## Hard Scope

1. Run the scanner first:
   `python scripts/tools/daily_bug_scan.py --since <last-run-iso-or-fallback> --base-ref origin/main --include-local-head --max-commits 5 --format json`
2. If `--since` cannot be used, fall back to `--hours 24`.
3. Inspect only candidate commits emitted by the scanner.
4. Land at most one bug fix by default.
5. Do not invent weak findings. Skip when evidence is not concrete.

## Required Fix Closeout

When a real bug is fixed, do not leave it in a detached or local-only worktree.
Use the finalizer helper:

```powershell
python scripts/tools/daily_bug_scan_finalize_fix.py `
  --branch codex/daily-bug-scan-<short-slug> `
  --message "fix(<scope>): <specific bug>" `
  --verify "python -m pytest <exact focused pytest targets> -q" `
  --verify "python -m py_compile <changed python files>" `
  --allowed-dirty HANDOFF.md
```

The helper must:

- create or reuse the named clean fix branch;
- run the supplied focused verification commands;
- commit only the fix package paths, excluding allowed baton churn;
- register `register_auto_merge_fix_task.ps1` with AutoRebase enabled;
- print `fix_sha` and `origin_main_contains_fix`.

## Final Report

Always report:

- scanner verification mode: `full`, `static_only`, or `blocked`;
- candidate SHA that produced the bug evidence;
- focused verification commands and results;
- fix branch and fix SHA;
- whether `origin/main` contains the fix SHA now;
- if not merged yet, the registered scheduled task name and log path.
