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
$quickExitExemptModes = @("doctor", "cleanup")
$interactiveHoldModes = @(
    "claude",
    "codex",
    "codex-search",
    "codex-project",
    "codex-project-gold-db",
    "codex-project-search-gold-db",
    "codex-project-smart",
    "codex-project-smart-gold-db",
    "codex-project-smart-search-gold-db",
    "codex-project-smart-power",
    "codex-project-power",
    "codex-project-linux",
    "codex-project-linux-gold-db",
    "codex-project-linux-search-gold-db",
    "codex-project-linux-power",
    "green-codex",
    "green-claude"
)
$suspiciousQuickExit = (
    $exitCode -eq 0 -and
    $elapsedSeconds -lt $quickExitThresholdSeconds -and
    $Mode -notin $quickExitExemptModes
)
$holdAfterInteractiveExit = $Mode -in $interactiveHoldModes

if ($exitCode -ne 0 -or $suspiciousQuickExit -or $holdAfterInteractiveExit) {
    Write-Host ""
    if ($suspiciousQuickExit) {
        Write-Host (
            "Codex closed after {0:N1}s with exit code 0. Treating this as a failed launch so the window stays open." -f
            $elapsedSeconds
        ) -ForegroundColor Yellow
        Write-Host "Run `codex.bat doctor` if the lines above did not make the cause obvious." -ForegroundColor Yellow
        $exitCode = 1
    } elseif ($holdAfterInteractiveExit) {
        Write-Host "Codex session exited with code $exitCode." -ForegroundColor Yellow
        Write-Host "The window is staying open so the exit output is visible." -ForegroundColor Yellow
    } else {
        Write-Host "Codex launch failed with exit code $exitCode." -ForegroundColor Red
    }
    Write-Host "Press Enter to close."
    Read-Host | Out-Null
}

exit $exitCode
