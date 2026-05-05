---
task: DeepSeek-via-OpenRouter agent launcher (Claude-Code-equivalent UX, model = DeepSeek)
mode: IMPLEMENTATION
slug: deepseek-agent-launcher
scope_lock:
  - scripts/tools/deepseek-agent.ps1
blast_radius: |
  scripts/tools/deepseek-agent.ps1 — NEW file. Zero callers, zero importers, zero references in pipeline/ or trading_app/.
  Does not touch canonical sources, schemas, drift checks, hooks, MCPs, or production code.
  Only effect: when the user runs `deepseek-agent` (or `.\scripts\tools\deepseek-agent.ps1`), spawns aider in the
  current directory wired to DeepSeek-V3 via OpenRouter. Per-invocation only — no env persistence, no profile edits.
  Reads gold.db / repo files only via aider (read-only or git-tracked diff with user approval per aider's default UX).
  Risk: low. Reversibility: delete the file.
---

# DeepSeek Agent Launcher — Stage File

## What

Single PowerShell script `scripts/tools/deepseek-agent.ps1` that launches **aider** (autonomous coding agent CLI) wired to **DeepSeek-V3 via OpenRouter** in the current repo. Same UX category as Claude Code (multi-turn, file edits, shell commands, git-aware) but the model is DeepSeek instead of Sonnet.

## Why

User wants Claude-Code-equivalent agent behavior (read repo, edit files, run commands, multi-turn) on a cheap model (DeepSeek-V3 ~30× cheaper than Sonnet 4.6). aider is the canonical OSS tool for this — model-agnostic, OR-compatible out of the box, git-aware, runs in any terminal.

## Files

| Path | Action | Reason |
|---|---|---|
| `scripts/tools/deepseek-agent.ps1` | CREATE | The launcher. Only file touched. |

## Approach (no ad-hoc, no drift)

1. **Dependency check.** Script first-runs: detect `aider` on PATH. If missing, `pip install aider-chat` (one command, well-known package, ~30s install). User can opt out by setting `DEEPSEEK_AGENT_SKIP_INSTALL=1`.

2. **Key resolution chain — mirrors `ask.py`'s `_load_dotenv_chain` exactly** (no new mechanism):
   - `$env:OPENROUTER_API_KEY` (shell wins, mirrors `ask.py:139`)
   - `$env:OPEN_ROUTER_API_KEY` (typo fallback, mirrors `ask.py:139`)
   - `~/.canompx-ask/.env` (install-root .env, mirrors `ask.py:131`)
   - `<repo>/.env` (project-root .env, mirrors `ask.py:134`)
   - `~/.canompx-ask/config.toml` `[openrouter].api_key` (parsed via `python -c "import tomllib"`; mirrors `ask.py:150-156`)
   - Reject empty / `sk-or-your*` placeholder (mirrors `ask.py:176`). Fail with the same doc URL `ask.py` cites.

3. **Launch aider** with:
   - `--model openrouter/deepseek/deepseek-chat-v3.1` (default; user can override via `-Model <or_id>`)
   - `--api-key openrouter=<key>` (aider's documented OR auth path)
   - Current working directory = repo root (script does NOT `cd`; aider uses `$PWD`)
   - All other args passed through verbatim (`@Args`)

4. **Per-invocation only.** No env persistence, no `$PROFILE` edits, no User-scope vars. Each invocation resolves the key fresh.

5. **No edits anywhere else.** No drift check additions (script doesn't touch production code). No MCP wiring (aider has its own model loop). No hooks. No `ask.py` changes.

## Existing patterns reused

- `param([string]$Model = "...", [switch]$NoLaunch)` + `$ErrorActionPreference = "Stop"` — matches `scripts/ralph_loop_runner.ps1` lines 11-19, `scripts/infra/windows-agent-launch.ps1` lines 1-10.
- Key resolution chain — verbatim mirror of `~/.canompx-ask/ask.py` `_load_dotenv_chain` (lines 128-134) + `_resolve_api_key` (lines 137-143) + placeholder guard (line 176).
- TOML read — `python -c "import tomllib; ..."` one-liner (canonical Python 3.11+, no new dep).
- Banner format — last-4-of-key, no full key in stdout. Matches existing telemetry hygiene.

## Out of scope

- No claude-code-router. No proxy. No fork of any tool.
- No bash sibling.
- No drift-check additions (script doesn't touch production code).
- No MCP integration with aider (aider's model loop is independent).
- No persistent install of OR key (per-invocation resolution only).
- No `$PROFILE` modification (user adds `Set-Alias` themselves if desired).

## Acceptance criteria

1. `pwsh -File scripts/tools/deepseek-agent.ps1 -NoLaunch` (in a shell with OR key set via any of the 5 sources above) prints a banner with `model=openrouter/deepseek/deepseek-chat-v3.1 key=...<last4>` and exits 0.
2. Same command with no key set anywhere → exits 1 with a clear "set OPENROUTER_API_KEY at one of: <5 paths>" message.
3. `pwsh -File scripts/tools/deepseek-agent.ps1` (with key) launches aider in the current directory; aider prints its prompt; user can issue an edit; aider commits via git when accepted.
4. If `aider` is not on PATH and `pip install aider-chat` fails (network failure, sandbox, conflicting deps, or `DEEPSEEK_AGENT_SKIP_INSTALL=1` set), script exits 1 with an explicit "aider not installed; run `pip install aider-chat` manually or set DEEPSEEK_AGENT_SKIP_INSTALL=1 after installing" message — no Python traceback bleeds through.
5. `python pipeline/check_drift.py` passes unchanged (no production-code touch).
6. `git diff --stat` shows exactly 1 new file: `scripts/tools/deepseek-agent.ps1`.

## Model-ID evidence (captured 2026-05-06)

OpenRouter `/api/v1/models` queried directly. DeepSeek IDs returned (filtered):
- `deepseek/deepseek-chat`
- `deepseek/deepseek-chat-v3-0324`
- `deepseek/deepseek-chat-v3.1` ← **default** (latest stable v3 chat)
- `deepseek/deepseek-r1`, `deepseek/deepseek-r1-0528`
- `deepseek/deepseek-v3.1-terminus`
- `deepseek/deepseek-v3.2`, `deepseek/deepseek-v3.2-exp`, `deepseek/deepseek-v3.2-speciale`
- `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`

The originally-drafted ID `deepseek/deepseek-chat-v3` (no suffix) does not exist in the catalog — replaced with `deepseek/deepseek-chat-v3.1`. With aider's `openrouter/` prefix the full string is `openrouter/deepseek/deepseek-chat-v3.1`. Users wanting v3.2 or v4 can override via `-Model`.

## Verification (after implementation)

```powershell
# Sanity — script syntax-clean
pwsh -NoProfile -Command "& { . .\scripts\tools\deepseek-agent.ps1 -NoLaunch }"

# Drift check unchanged
python pipeline/check_drift.py

# Diff scope
git status --short
git diff --stat
```

## Done definition

All five required:
- [ ] `-NoLaunch` mode prints correct banner (key resolved + model shown)
- [ ] Missing-key path fails with documented error
- [ ] Missing-aider + failed-install path fails with documented error (no traceback)
- [ ] `python pipeline/check_drift.py` passes
- [ ] Diff is exactly 1 new file
