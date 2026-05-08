param(
    [Parameter(Mandatory = $true)]
    [string]$Mode,

    [string]$Task = "",
    [string]$Title = "Codex",
    [switch]$Inline
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$launcherPs1 = Join-Path $repoRoot "scripts\infra\windows-agent-launch.ps1"

if (-not $Inline) {
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath,
        "-Mode", $Mode,
        "-Title", $Title,
        "-Inline"
    )
    if ($Task) {
        $args += @("-Task", $Task)
    }
    Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList $args | Out-Null
    exit 0
}

try {
    $host.UI.RawUI.WindowTitle = $Title
} catch {
    # Best effort only; some hosts do not expose RawUI.
}

$start = Get-Date
$launcherArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $launcherPs1,
    "-Mode", $Mode
)
if ($Task) {
    $launcherArgs += @("-Task", $Task)
}
& powershell.exe @launcherArgs
$exitCode = $LASTEXITCODE
$elapsedSeconds = ((Get-Date) - $start).TotalSeconds
$quickExitThresholdSeconds = 2.0
$quickExitExemptModes = @("doctor")
$suspiciousQuickExit = (
    $exitCode -eq 0 -and
    $elapsedSeconds -lt $quickExitThresholdSeconds -and
    $Mode -notin $quickExitExemptModes
)

if ($exitCode -ne 0 -or $suspiciousQuickExit) {
    Write-Host ""
    if ($suspiciousQuickExit) {
        Write-Host (
            "Codex closed after {0:N1}s with exit code 0. Treating this as a failed launch so the window stays open." -f
            $elapsedSeconds
        ) -ForegroundColor Yellow
        Write-Host "Run `codex.bat doctor` if the lines above did not make the cause obvious." -ForegroundColor Yellow
        $exitCode = 1
    } else {
        Write-Host "Codex launch failed with exit code $exitCode." -ForegroundColor Red
    }
    Write-Host "Press Enter to close."
    Read-Host | Out-Null
}

exit $exitCode
