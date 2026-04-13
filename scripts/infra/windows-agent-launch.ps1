param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("claude", "codex", "codex-search", "codex-project", "codex-project-gold-db", "codex-project-search-gold-db", "list", "close", "close-pick", "resume", "menu", "prune")]
    [string]$Mode,

    [string]$Task,
    [string]$Tool
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$launcherPy = Join-Path $repoRoot "scripts\infra\windows_agent_launch.py"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

$launcherArgs = @("--mode", $Mode)
if ($Task) {
    $launcherArgs += @("--task", $Task)
}
if ($Tool) {
    $launcherArgs += @("--tool", $Tool)
}

if (Test-Path $venvPython) {
    & $venvPython $launcherPy @launcherArgs
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $launcherPy @launcherArgs
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $launcherPy @launcherArgs
    exit $LASTEXITCODE
}

Write-Error "No usable Python launcher found for AI Workstreams. Expected .venv\\Scripts\\python.exe, `py -3`, or `python`."
exit 1
