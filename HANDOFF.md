# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-29
- **Summary:** Committed MGC O30 pooled-base research (`f1cc6435`, VERDICT DEAD_FOR_ORB) and fixed an `audit_behavioral.py` triple-join false positive that was blocking the commit (`_looks_like_sql()` now requires a line-led SQL clause; +2 regression tests). gold.db write-lock race resolved (dry-run `drifted=0` proved data canonical). Local `main` is ahead of `origin/main` — not pushed. Detail → `memory/feedback_behavioral_audit_with_join_prose_false_positive_2026_05_29.md`.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
