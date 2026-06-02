task: Per-worktree venv isolation — opt-in UV_PROJECT_ENVIRONMENT so a peer worktree's `uv sync` cannot strip the shared `.venv` dev group (tonight's recurring failure #1) or contend its DLLs (#2).
mode: IMPLEMENTATION
stage: (1/1)

## Scope Lock
- scripts/tools/new_session.sh
- START_WORKTREE.bat
- .githooks/pre-commit
- .claude/rules/worktree-venv-isolation.md

## Blast Radius
- scripts/tools/new_session.sh — adds an isolated-venv bootstrap (`uv sync --locked --group dev` with worktree-local UV_PROJECT_ENVIRONMENT) after `git worktree add`. New behavior only on NEW worktrees; existing worktrees untouched.
- START_WORKTREE.bat — same bootstrap, Windows side. New worktrees only.
- .githooks/pre-commit — ADDITIVE: probe order already prefers worktree-local `.venv` first (lines 156-197). No behavior change for existing worktrees that share the canonical venv; the canonical-sibling fallback stays. Adds one comment + keeps both WSL/.venv-wsl and Windows/.venv branches exactly as-is.
- .claude/rules/worktree-venv-isolation.md — NEW doctrine file documenting the isolation model so the project "understands" it.
- Reads: none destructive. Writes: only the 4 files above.
- Main checkout `canompx3/.venv` — UNTOUCHED (already has its own venv; zero risk to live bot env).
- Existing 13 worktrees — UNTOUCHED (keep shared-venv fallback until they opt in by re-running bootstrap).
- Capital path — NOT touched. No trading_app/ or pipeline/ logic.
