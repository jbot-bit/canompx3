param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$helper = Join-Path $repoRoot "scripts\infra\codex_local_env.py"

if (Get-Command uv -ErrorAction SilentlyContinue) {
    & uv run --frozen python $helper cleanup --platform windows
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $helper cleanup --platform windows
    exit $LASTEXITCODE
}

& python $helper cleanup --platform windows
exit $LASTEXITCODE
