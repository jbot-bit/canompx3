---
task: "Durable drain routine — generalize auto_merge_reaper_prefix_guard.ps1 to drain_worktrees.ps1 (DRY-RUN ONLY, never auto-push) + session-start drain nudge. Recurrence fix for stranded-worktree drift. Operator GO 2026-06-10, autonomy=dry-run-only."
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/drain_worktrees.ps1
  - .claude/hooks/drain-nudge-sessionstart.py
  - .claude/settings.json
  - tests/test_tools/test_drain_worktrees.py
---

## Blast Radius

- `scripts/tools/drain_worktrees.ps1` — NEW tooling script. Dry-run by default
  (reports DRAIN/CAPITAL/DIVERGED/MERGED classification of every local branch vs
  origin/main). Pushes NOTHING unless `-Execute`, and even then only clean,
  non-capital fast-forwards, peer-lease-checked, fetch-before-push, never --force.
  Inherits every safety property of the proven `auto_merge_reaper_prefix_guard.ps1`.
  Capital classification delegates to the canonical `_CAPITAL_PATH_PREFIXES`
  (`.claude/hooks/judgment-review-nudge.py`) — mirrored with @canonical-source
  provenance + a parity drift check.
- `.claude/hooks/drain-nudge-sessionstart.py` — NEW advisory-only SessionStart
  hook. Counts branches-ahead-of-main + capital stashes; emits a one-line cue if
  above threshold. Fail-open (try/except BaseException -> exit 0), one-shot, never
  blocks. Same contract as the existing memory-capture SessionStart hooks.
- `.claude/settings.json` — additive: register the new SessionStart hook alongside
  the existing ones (additionalContext channel; concatenated, not exclusive).
- `tests/test_tools/test_drain_worktrees.py` — NEW. Tests the classification logic
  (DRAIN vs CAPITAL vs DIVERGED) + the nudge threshold, with a tmp git repo fixture.

- Capital-path risk: LOW. The script CAN push to main under `-Execute`, but never
  a capital branch (always SKIP) and never unattended. The nudge is advisory-only.
  Nothing here touches pipeline/ or trading_app/ production logic.

## Verification
- check_drift.py exit 0 (incl. the new capital-prefix parity check).
- test_drain_worktrees.py green (classification + nudge threshold).
- PowerShell `-WhatIf`/dry-run lists correct DRAIN/SKIP split against real branches.
- Nudge hook live-fires on a piped SessionStart payload.

## Status
mode: IMPLEMENTATION — operator GO 2026-06-10. Autonomy locked to DRY-RUN-ONLY
(no scheduled task, no unattended push) per operator decision.
