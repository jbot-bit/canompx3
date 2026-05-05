# DeepSeek Agent Launcher
#
# Spawns aider (autonomous coding agent) in the current directory wired to
# DeepSeek-V3.1 via OpenRouter. Same UX category as Claude Code (multi-turn,
# file edits, shell commands, git-aware) but runs on a much cheaper model.
#
# Usage:
#   .\scripts\tools\deepseek-agent.ps1                          # launch aider
#   .\scripts\tools\deepseek-agent.ps1 -NoLaunch                # banner only, no spawn
#   .\scripts\tools\deepseek-agent.ps1 -Model openrouter/deepseek/deepseek-v3.2
#
# Key resolution chain (mirrors ~/.canompx-ask/ask.py):
#   1. $env:OPENROUTER_API_KEY                              (ask.py:139)
#   2. $env:OPEN_ROUTER_API_KEY  (typo fallback)            (ask.py:139)
#   3. ~/.canompx-ask/.env       (install-root)             (ask.py:131)
#   4. <repo>/.env               (project-root)             (ask.py:134)
#   5. ~/.canompx-ask/config.toml [openrouter].api_key      (ask.py:150-156)
#
# Opt-out for auto-install: $env:DEEPSEEK_AGENT_SKIP_INSTALL=1

param(
    [string]$Model = "openrouter/deepseek/deepseek-chat-v3.1",
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

# ---------- helpers ----------

function Read-DotEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return @{} }
    $out = @{}
    foreach ($raw in Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or ($line -notmatch "=")) { continue }
        $eq = $line.IndexOf("=")
        $k = $line.Substring(0, $eq).Trim()
        $v = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        if ($k) { $out[$k] = $v }
    }
    return $out
}

function Resolve-OpenRouterKey {
    foreach ($name in @("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")) {
        $val = [Environment]::GetEnvironmentVariable($name)
        if ($val -and $val.Trim()) { return @{ key = $val.Trim(); source = "env:$name" } }
    }

    $installEnv = Join-Path $HOME ".canompx-ask\.env"
    $kv = Read-DotEnv -Path $installEnv
    foreach ($name in @("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")) {
        if ($kv.ContainsKey($name) -and $kv[$name]) {
            return @{ key = $kv[$name]; source = $installEnv }
        }
    }

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    $projEnv = Join-Path $repoRoot ".env"
    $kv = Read-DotEnv -Path $projEnv
    foreach ($name in @("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY")) {
        if ($kv.ContainsKey($name) -and $kv[$name]) {
            return @{ key = $kv[$name]; source = $projEnv }
        }
    }

    $tomlPath = Join-Path $HOME ".canompx-ask\config.toml"
    if (Test-Path -LiteralPath $tomlPath) {
        $py = @"
import sys, tomllib
try:
    with open(r'$tomlPath', 'rb') as fh:
        cfg = tomllib.load(fh)
    k = (cfg.get('openrouter', {}) or {}).get('api_key', '')
    sys.stdout.write(str(k).strip())
except Exception:
    sys.stdout.write('')
"@
        $key = & python -c $py 2>$null
        if ($LASTEXITCODE -eq 0 -and $key -and $key.Trim()) {
            return @{ key = $key.Trim(); source = $tomlPath }
        }
    }

    return @{ key = ""; source = "" }
}

function Get-KeyTail {
    param([string]$Key)
    if (-not $Key -or $Key.Length -lt 4) { return "????" }
    return $Key.Substring($Key.Length - 4)
}

function Test-AiderInstalled {
    $cmd = Get-Command aider -ErrorAction SilentlyContinue
    return [bool]$cmd
}

function Install-Aider {
    if ($env:DEEPSEEK_AGENT_SKIP_INSTALL -eq "1") {
        Write-Error "aider not on PATH and DEEPSEEK_AGENT_SKIP_INSTALL=1 - install manually: pip install aider-chat"
        return $false
    }
    Write-Host "[deepseek-agent] aider not found; running: pip install aider-chat" -ForegroundColor Yellow
    & python -m pip install aider-chat 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install aider-chat failed (exit $LASTEXITCODE). Install manually then re-run, or set DEEPSEEK_AGENT_SKIP_INSTALL=1 once aider is on PATH."
        return $false
    }
    if (-not (Test-AiderInstalled)) {
        Write-Error "pip install reported success but aider is still not on PATH. Add Python's user-scripts directory to PATH (e.g., %APPDATA%\Python\Scripts) and re-run."
        return $false
    }
    return $true
}

# ---------- main ----------

$resolved = Resolve-OpenRouterKey
$key = $resolved.key
$source = $resolved.source

if (-not $key -or $key.StartsWith("sk-or-your")) {
    Write-Host ""
    Write-Host "OPENROUTER_API_KEY missing or placeholder." -ForegroundColor Red
    Write-Host "  Get one: https://openrouter.ai/keys"
    Write-Host "  Set it via any of:"
    Write-Host "    - shell env: `$env:OPENROUTER_API_KEY = 'sk-or-...'"
    Write-Host "    - install env: write OPENROUTER_API_KEY=sk-or-... to $HOME\.canompx-ask\.env"
    Write-Host "    - project env: write OPENROUTER_API_KEY=sk-or-... to <repo>\.env"
    Write-Host "    - config file: $HOME\.canompx-ask\config.toml  [openrouter] api_key = '...'"
    exit 1
}

$tail = Get-KeyTail $key
Write-Host "[deepseek-agent] model=$Model key=...$tail source=$source"

if ($NoLaunch) {
    exit 0
}

if (-not (Test-AiderInstalled)) {
    if (-not (Install-Aider)) { exit 1 }
}

$aiderArgs = @(
    "--model", $Model,
    "--api-key", "openrouter=$key"
) + $args

& aider @aiderArgs
exit $LASTEXITCODE
