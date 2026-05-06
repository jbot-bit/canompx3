# OpenCode Agent Launcher
#
# Spawns `opencode` (autonomous coding TUI) in the current directory wired to
# DeepSeek-V3.1 via OpenRouter. Same UX category as Claude Code (multi-turn,
# autonomous file edits, shell commands, MCP, git-aware) on a much cheaper model.
#
# Replaces the aider-based deepseek-agent.ps1 (PR #242, 2026-05-06): aider
# hid tool use and forced /add ceremony. opencode exposes tool use natively.
#
# Usage:
#   .\scripts\tools\opencode-agent.ps1                          # launch opencode
#   .\scripts\tools\opencode-agent.ps1 -NoLaunch                # banner only, no spawn
#   .\scripts\tools\opencode-agent.ps1 -Model openrouter/deepseek/deepseek-v3.2
#
# Key resolution chain (mirrors ~/.canompx-ask/ask.py:128-143,176):
#   1. $env:OPENROUTER_API_KEY                              (ask.py:139)
#   2. $env:OPEN_ROUTER_API_KEY  (typo fallback)            (ask.py:139)
#   3. ~/.canompx-ask/.env       (install-root)             (ask.py:131)
#   4. <repo>/.env               (project-root)             (ask.py:134)
#   5. ~/.canompx-ask/config.toml [openrouter].api_key      (ask.py:150-156)
#
# Resolved key is exported as $env:OPENROUTER_API_KEY so opencode picks it up
# natively per https://opencode.ai/docs/providers/.
#
# Opt-out for auto-install: $env:OPENCODE_AGENT_SKIP_INSTALL=1

param(
    # canonical-default-fallback: openrouter/deepseek-chat-v3.1
    # Used only when the canonical profile resolver cannot return a model
    # (e.g. CANOMPX3_AI_DEEPSEEK_CODING_MODEL unset). The drift check
    # `check_hardcoded_openrouter_model_in_launcher` exempts this annotation;
    # any second hardcoded openrouter/<vendor>/<model> in this file flips it
    # to a hard block.
    [string]$Model = "openrouter/deepseek-chat-v3.1",
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

# ---------- PATH bootstrap ----------
# npm-global on Windows lives at %APPDATA%\npm. Prepend so `opencode` resolves
# even in PS sessions launched before npm was on PATH.
$npmGlobal = Join-Path $HOME "AppData\Roaming\npm"
if ((Test-Path -LiteralPath $npmGlobal) -and ($env:Path -notlike "*$npmGlobal*")) {
    $env:Path = "$npmGlobal;$env:Path"
}

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

function Test-OpencodeInstalled {
    $cmd = Get-Command opencode -ErrorAction SilentlyContinue
    return [bool]$cmd
}

function Get-OpencodeVersion {
    try {
        $v = & opencode --version 2>$null
        if ($LASTEXITCODE -eq 0 -and $v) { return ($v | Out-String).Trim() }
    } catch {}
    return "unknown"
}

function Install-Opencode {
    if ($env:OPENCODE_AGENT_SKIP_INSTALL -eq "1") {
        Write-Error "opencode not on PATH and OPENCODE_AGENT_SKIP_INSTALL=1 - install manually: npm install -g opencode-ai"
        return $false
    }
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        Write-Error "npm not found. Install Node 20+ from https://nodejs.org and re-run."
        return $false
    }
    Write-Host "[opencode-agent] opencode not found; running: npm install -g opencode-ai" -ForegroundColor Yellow
    & npm install -g opencode-ai 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Error "npm install -g opencode-ai failed (exit $LASTEXITCODE). Install manually then re-run, or set OPENCODE_AGENT_SKIP_INSTALL=1 once opencode is on PATH."
        return $false
    }
    if (-not (Test-OpencodeInstalled)) {
        Write-Error "npm install reported success but opencode is still not on PATH. Add %APPDATA%\npm to PATH and re-run."
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

# Export for opencode (reads OPENROUTER_API_KEY natively per opencode.ai/docs/providers/).
$env:OPENROUTER_API_KEY = $key

if (-not (Test-OpencodeInstalled)) {
    if (-not (Install-Opencode)) { exit 1 }
}

# ---------- canonical model resolution ----------
# Delegate to provider_registry.get_profile("deepseek_coding"); single
# source of truth for the model ID. If the env var is unset (or any other
# validation_errors), fall back to the launcher default with explicit WARN.
$repoRootForResolver = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$resolverScript = Join-Path $repoRootForResolver "scripts\tools\opencode_resolve_model.py"
$modelSource = "launcher default"
if (Test-Path -LiteralPath $resolverScript) {
    $resolved = & python $resolverScript 2>$null
    if ($LASTEXITCODE -eq 0 -and $resolved -and $resolved.Trim()) {
        $Model = $resolved.Trim()
        $modelSource = "canonical profile"
    } else {
        Write-Host ("[opencode-agent] WARN: canonical profile not configured; " +
            "using launcher default model=$Model. Set " +
            "CANOMPX3_AI_DEEPSEEK_CODING_MODEL=<openrouter/...> for single-source-of-truth.") `
            -ForegroundColor Yellow
    }
}

$version = Get-OpencodeVersion
$tail = Get-KeyTail $key
Write-Host "[opencode-agent] tool=opencode version=$version model=$Model ($modelSource) key=...$tail source=$source"

if ($NoLaunch) {
    exit 0
}

# Activate the pre-commit review gate (step 0d in .githooks/pre-commit).
# Every commit during this session is reviewed by Claude (seven-sins rubric)
# before it lands. See scripts/tools/claude_review_deepseek.py.
$env:OPENCODE_AGENT_ACTIVE = "1"

& opencode --model $Model @args
exit $LASTEXITCODE
