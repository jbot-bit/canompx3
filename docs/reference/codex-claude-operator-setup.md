# Codex + Claude Operator Setup

This repo supports both Claude and Codex. Use both, but do not let them mutate
the same checkout at the same time.

## Recommended split

- `claude.bat`
  - planning
  - broad repo navigation
  - review and verification
- `codex.bat linux`
  - normal Codex implementation work against a WSL-home clone
- `codex.bat linux-power`
  - heavier Codex sessions when you want maximum reasoning depth
- `ai-workstreams.bat`
  - separate worktrees when Claude and Codex need to work in parallel

## Non-negotiables

- Keep the Codex repo under WSL home, for example `~/canompx3`, not
  `/mnt/c/...`.
- Use separate worktrees if Claude and Codex are both editing in parallel.
- Do not chase Codex hook parity on Windows. OpenAI documents hooks as
  experimental and Windows support is currently disabled.

## Why this setup

OpenAI's Codex docs say:

- use WSL2 when your workflow already lives in Linux
- keep repositories under the Linux filesystem for faster I/O and fewer
  permission and symlink issues
- prefer the native Windows `elevated` sandbox when running Codex natively on
  Windows
- use local environments for setup scripts, cleanup scripts, and top-bar
  actions in the Codex app

Sources:

- `https://developers.openai.com/codex/windows`
- `https://developers.openai.com/codex/config-reference`
- `https://developers.openai.com/codex/app/local-environments`
- `https://developers.openai.com/codex/hooks`

## Codex app local environments

Use the Codex app settings pane and paste these commands into the local
environment fields.

### Default setup script

For WSL/Linux:

```bash
bash scripts/infra/codex-app-setup.sh
```

For Windows override:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\infra\codex-app-setup.ps1
```

### Default cleanup script

For WSL/Linux:

```bash
bash scripts/infra/codex-app-cleanup.sh
```

For Windows override:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\infra\codex-app-cleanup.ps1
```

The cleanup script is intentionally safe. It removes local caches, build
artifacts, and `__pycache__` folders, but it does not run `git clean` or delete
tracked files.

### Suggested actions

Add these actions to the Codex app header.

Status:

```bash
python3 scripts/infra/codex_local_env.py status --platform wsl
```

Lint:

```bash
python3 scripts/infra/codex_local_env.py lint --platform wsl
```

Tests:

```bash
python3 scripts/infra/codex_local_env.py tests --platform wsl
```

Drift:

```bash
python3 scripts/infra/codex_local_env.py drift --platform wsl
```

Windows fallback actions:

```powershell
py -3 scripts/infra/codex_local_env.py status --platform windows
py -3 scripts/infra/codex_local_env.py lint --platform windows
py -3 scripts/infra/codex_local_env.py tests --platform windows
py -3 scripts/infra/codex_local_env.py drift --platform windows
```

## Codex profiles

The repo-scoped `.codex/config.toml` now provides:

- `canompx3`
  - normal day-to-day profile
- `canompx3_search`
  - live search profile
- `canompx3_power`
  - maximum-reasoning Codex profile
- `canompx3_windows`
  - native Windows fallback profile

Quick launcher front doors:

- `codex.bat`
- `codex.bat power`
- `codex.bat linux`
- `codex.bat linux-power`

## Practical best practice

- Use `claude.bat` when you want review, planning, or repo-wide judgment.
- Use `codex.bat linux` when you want implementation speed.
- Use `codex.bat linux-power` when the task is hard enough to justify extra
  reasoning cost and latency.
- Use `ai-workstreams.bat` when both tools need to be active on different tasks.
