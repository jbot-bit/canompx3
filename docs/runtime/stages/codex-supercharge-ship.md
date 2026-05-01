---
task: codex-supercharge-ship
mode: IMPLEMENTATION
slug: codex-supercharge-ship
worktree: C:/Users/joshd/canompx3-codex-supercharge
branch: ship/codex-supercharge-roadmap
opened: 2026-05-01
roadmap: docs/plans/2026-05-01-codex-supercharge-roadmap.md
scope_lock:
  - .codex/COMMANDS.md
  - .codex/INTEGRATIONS.md
  - .codex/OPENAI_CODEX_STANDARDS.md
  - .codex/STARTUP.md
  - .codex/WORKFLOWS.md
  - .codex/config.toml
  - .codex/skills/README.md
  - .codex/hooks/session_start.py
  - .codex/hooks/user_prompt_submit_grounding.py
  - .agents/skills/
  - .mcp.json
  - CODEX.md
  - HANDOFF.md
  - context/institutional.py
  - docs/reference/codex-claude-operator-setup.md
  - docs/reference/codex-operator-handbook.md
  - docs/plans/2026-05-01-codex-supercharge-roadmap.md
  - docs/runtime/stages/codex-supercharge-ship.md
  - scripts/infra/codex-capital-review.sh
  - scripts/infra/codex-project-search.sh
  - scripts/infra/codex-project.sh
  - scripts/infra/codex-review.sh
  - scripts/infra/run-repo-state-mcp.sh
  - scripts/infra/run-research-catalog-mcp.sh
  - scripts/tools/repo_state_mcp_server.py
  - scripts/tools/research_catalog_mcp_server.py
  - tests/test_tools/test_repo_state_mcp_server.py
  - tests/test_tools/test_research_catalog_mcp_server.py
---

## Task

Ship the in-flight Codex layer alignment plus two new read-only MCP servers
(`repo-state`, `research-catalog`) per
`docs/plans/2026-05-01-codex-supercharge-roadmap.md`. Work was authored in a
prior Codex session, left uncommitted, and transported here via
`git stash push -u` from the main worktree per parallel-session-awareness +
branch-flip-protection rules.

This is a NON-CODE-PATH ship — the new MCP servers are read-only operator
tooling that wraps existing canonical scripts (`context_resolver`,
`task_route_packet`, `project_pulse`, `system_context`, `context_views`) and
read-only doc surfaces (`docs/institutional/literature/`,
`docs/audit/hypotheses/`, `docs/audit/results/`). They do NOT touch
`pipeline/` or `trading_app/` truth paths. They do NOT write to `gold.db`.
They do NOT introduce a second truth surface.

## Scope Lock

See `scope_lock:` block above. NEW files are the four MCP server files
(`repo_state_mcp_server.py`, `research_catalog_mcp_server.py`,
`run-repo-state-mcp.sh`, `run-research-catalog-mcp.sh`), the two test files,
the two `.codex/hooks/` Python hooks, the `.agents/skills/` wrapper layer,
the roadmap doc, and this stage file. EVERYTHING ELSE is a modification of
documentation, Codex layer config, or Codex launcher shell scripts.

OUT OF SCOPE — explicitly EXCLUDED from any commit on this branch:
- `CodePilot/` (separate Electron project, gitignored anyway)
- `chatgpt-legal-kit/` (unrelated personal kit)
- `.github/commands/gemini-*.toml` (Gemini integration, separate effort)
- `docs/external/code-review-graph/eval-2026-04-29/*.csv` (CRG eval scratch)
- `plugins/superpower-claude/commands/brief-workspace.md` (superpower plugin)

## Blast Radius

- New MCP servers (`repo_state_mcp_server.py`,
  `research_catalog_mcp_server.py`): read-only fastmcp stdio servers.
  Zero callers in `pipeline/` or `trading_app/`. Imported by their own
  test files only. Registered in root `.mcp.json` for both Claude Code and
  Codex consumption.
- Launcher shell scripts (`run-repo-state-mcp.sh`,
  `run-research-catalog-mcp.sh`): WSL/Linux launchers; pick `.venv-wsl/bin/python`
  if present else `python3`. No PATH or environment writes.
- `.mcp.json`: adds two server entries; existing `gold-db` and
  `code-review-graph` entries unchanged.
- `.codex/*` files + `.agents/skills/`: Codex-side operator surface only.
  No imports from `pipeline/` or `trading_app/`.
- `.codex/hooks/session_start.py`, `.codex/hooks/user_prompt_submit_grounding.py`:
  Codex-side hooks running in the Codex process; not visible to Claude Code.
- `context/institutional.py`: small change — INSPECT before commit, this is
  the only Python module under `context/` modified by the stash.
- `HANDOFF.md`: append session log entry.
- `docs/reference/codex-*.md`, `CODEX.md`: doc updates, no code impact.
- Tests cover the two new MCP servers (14 tests, all currently green).

Reads: doc tree under `docs/institutional/`, `docs/audit/`,
`docs/runtime/lane_allocation.json` (read by `repo-state` MCP via
`context_views.py`); no DB reads from these MCPs.

Writes: NONE. Both servers are strictly read-only.

## Verification gates (pre-PR)

1. `python pipeline/check_drift.py` clean
2. `pytest tests/test_tools/test_repo_state_mcp_server.py tests/test_tools/test_research_catalog_mcp_server.py` green
3. Smoke-launch each MCP via fastmcp `list_tools` handshake
4. `git diff --stat origin/main HEAD` matches scope_lock — no scope creep
5. `git log --oneline origin/main..HEAD` shows ONLY the planned commits

## Commit plan (logical chunks)

- (a) **MCP: repo-state** — server + launcher + tests + `.mcp.json` entry
- (b) **MCP: research-catalog** — server + launcher + tests + `.mcp.json` entry
- (c) **Codex: layer alignment** — `.codex/*`, `.agents/skills/`,
      `.codex/hooks/*`, `CODEX.md`, `docs/reference/codex-*.md`,
      `context/institutional.py`, `scripts/infra/codex-*.sh`
- (d) **Docs: roadmap + handoff** — `docs/plans/2026-05-01-codex-supercharge-roadmap.md`,
      `HANDOFF.md`, this stage file

## Done criteria

All five gates above pass + PR opened against `main` + stage file deleted on
PR merge. The next roadmap item (`strategy-lab` MCP) is OUT OF SCOPE for this
stage and will get its own stage file.
