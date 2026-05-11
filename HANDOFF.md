# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex (WSL)
- **Date:** 2026-05-12
- **Commit:** b5cca6c6 — chore(codex): remove redundant gold db launch modes
- **Files changed:** 11 files
  - `.codex/COMMANDS.md`
  - `.codex/INTEGRATIONS.md`
  - `.codex/STARTUP.md`
  - `CODEX.md`
  - `HANDOFF.md`
  - `codex.bat`
  - `docs/reference/codex-operator-handbook.md`
  - `scripts/infra/codex-project-gold-db.sh`
  - `scripts/infra/codex-project-search-gold-db.sh`
  - `scripts/infra/windows-agent-launch.ps1`
  - `scripts/infra/windows_agent_launch.py`

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
