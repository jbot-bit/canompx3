---
task: OpenCode agent launcher (Claude-Code-equivalent UX, model = DeepSeek via OpenRouter)
mode: IMPLEMENTATION
slug: opencode-agent-launcher
scope_lock:
  - scripts/tools/opencode-agent.ps1
  - scripts/tools/deepseek-agent.ps1
  - docs/runtime/stages/opencode-agent-launcher.md
blast_radius: "scripts/tools/opencode-agent.ps1 (NEW, ~120 lines, spawns opencode wired to OpenRouter); scripts/tools/deepseek-agent.ps1 (REWRITE to ~10-line delegator, preserves deepseek shim entry point); stage doc (NEW). Zero production-code touch. No schema, no canonical sources, no MCP, no hooks. Risk: low. Reversibility: git revert."
---

# OpenCode Agent Launcher — Stage File

## What

Replace the aider-based `deepseek-agent.ps1` (PR #242, 2026-05-06) with an OpenCode-based
launcher. New companion `opencode-agent.ps1` is the canonical name; `deepseek-agent.ps1`
becomes a thin delegator for backwards-compat with the existing `deepseek.bat` shim and
PS profile function.

## Why

Aider's UX is `/add file` → edit. It hides tool use from the model. Confirmed via
official OpenRouter docs (verified 2026-05-06): DeepSeek-V3.1 supports tool calling
natively at $0.15/M in / $0.75/M out (~20× cheaper than Sonnet 4.6). The blocker
was the harness, not the model. OpenCode exposes tool use, autonomous file reads,
and MCP — actual Claude-Code-equivalent UX in the terminal.

## Files

| Path | Action | Reason |
|---|---|---|
| `scripts/tools/opencode-agent.ps1` | CREATE | Canonical launcher: detect opencode, resolve key, spawn in cwd. |
| `scripts/tools/deepseek-agent.ps1` | REWRITE | Thin delegator to opencode-agent.ps1. Preserves entry points. |
| `docs/runtime/stages/opencode-agent-launcher.md` | CREATE | This stage doc. |

## Approach

1. **PATH bootstrapping**: prepend `$HOME\AppData\Roaming\npm` so `opencode` resolves
   even in PS sessions that started before npm was on PATH.

2. **Dependency check**: detect `opencode` on PATH; if missing, run
   `npm install -g opencode-ai` (verified working on host 2026-05-06; takes ~25s).
   Opt-out via `OPENCODE_AGENT_SKIP_INSTALL=1`. Guard against missing node/npm.

3. **Key resolution chain — verbatim mirror of ask.py:128-143,176**:
   1. `$env:OPENROUTER_API_KEY`
   2. `$env:OPEN_ROUTER_API_KEY` (typo fallback)
   3. `~/.canompx-ask/.env`
   4. `<repo>/.env`
   5. `~/.canompx-ask/config.toml [openrouter].api_key`

   Reject empty / `sk-or-your*` placeholders with full guidance. Export the resolved
   key as `OPENROUTER_API_KEY` so opencode reads it natively
   (per https://opencode.ai/docs/providers/).

4. **Banner**: `[opencode-agent] tool=opencode version=<X> model=openrouter/deepseek-chat-v3.1 key=...<last4> source=<env|file>`. Last-4 only.

5. **`-NoLaunch`**: banner + exit 0. No spawn. Smoke-test mode.

6. **Launch**: `opencode --model openrouter/deepseek-chat-v3.1 @args`. CWD is whatever
   the user invoked from. Exit code propagated.

7. **Backwards compat**: `deepseek-agent.ps1` becomes a 5-line delegator. `deepseek.bat`
   (off-repo at `~/.local/bin/`) and the PS profile `deepseek` function continue
   working unchanged.

## Existing patterns reused

- `param([switch]$NoLaunch)` + `$ErrorActionPreference = "Stop"` — same idiom as
  `ralph_loop_runner.ps1` and the prior `deepseek-agent.ps1`.
- 5-source key resolver — verbatim port from prior `deepseek-agent.ps1` (reuse, not
  re-encode).
- `$LASTEXITCODE` propagation — matches `windows-agent-launch.ps1`.
- `OPENCODE_AGENT_SKIP_INSTALL=1` sentinel — same idiom as
  `DEEPSEEK_AGENT_SKIP_INSTALL=1` in the aider launcher.

## Out of scope

- No `.gitignore` change (opencode auth lives off-repo at `~/.local/share/opencode/`).
- No `~/.canompx-ask/` modification.
- No drift-check additions.
- No MCP wiring inside opencode (opencode-side `opencode.json` is user-managed).
- No `$PROFILE` modification (already handled in PR #242).
- No `deepseek.bat` change (already off-repo at `~/.local/bin/`).

## Acceptance criteria

1. `pwsh -File scripts/tools/opencode-agent.ps1 -NoLaunch` prints banner with
   tool/version/model/key/source populated; exits 0.
2. With no OR key anywhere → exits 1, message names all 5 sources.
3. With `OPENCODE_AGENT_SKIP_INSTALL=1` and opencode missing → exits 1 with install
   guidance, no traceback.
4. `pwsh -File scripts/tools/deepseek-agent.ps1 -NoLaunch` produces SAME banner as
   `opencode-agent.ps1 -NoLaunch` (delegation works).
5. `python pipeline/check_drift.py` passes (advisory checks tolerated).
6. `git diff --stat` shows exactly 3 files changed: `opencode-agent.ps1` (new),
   `deepseek-agent.ps1` (rewrite), this stage doc (new).

## Verification

```powershell
pwsh -NoProfile -File scripts\tools\opencode-agent.ps1 -NoLaunch
pwsh -NoProfile -File scripts\tools\deepseek-agent.ps1 -NoLaunch  # delegates
$env:OPENROUTER_API_KEY = ''; pwsh -NoProfile -File scripts\tools\opencode-agent.ps1 -NoLaunch  # exit 1
python pipeline\check_drift.py
git status --short ; git diff --stat
```

## Done definition

All required:
- [ ] `opencode-agent.ps1 -NoLaunch` prints correct banner
- [ ] `deepseek-agent.ps1 -NoLaunch` delegates and prints same banner
- [ ] Missing-key exits 1 with all 5 sources named
- [ ] Missing-opencode + skip-install exits 1 with clean message
- [ ] `python pipeline/check_drift.py` passes
- [ ] Diff is exactly 3 files
