param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    & $venvPython (Join-Path $repoRoot "scripts\infra\codex_local_env.py") cleanup --platform windows
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 (Join-Path $repoRoot "scripts\infra\codex_local_env.py") cleanup --platform windows
    exit $LASTEXITCODE
}

& python (Join-Path $repoRoot "scripts\infra\codex_local_env.py") cleanup --platform windows
exit $LASTEXITCODE
