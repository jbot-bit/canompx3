#requires -Version 5.1
<#
.SYNOPSIS
  List and (after y/N confirmation) kill stale live-trading python processes.

.DESCRIPTION
  Finds python processes whose command line matches any of:
    - run_live_session
    - bot_dashboard
    - webhook_server
  Prints a table (PID, StartTime, command-line fragment), then prompts
  before killing. Never auto-kills. Reversible by simply not confirming.

  Use when preflight check 6 reports:
    "LOCKED by PID <n>. Run: scripts/tools/stop_live.ps1 to clear, then retry."

.PARAMETER NoPrompt
  Skip the y/N prompt and kill matches. Use only in automation.
#>

[CmdletBinding()]
param(
    [switch]$NoPrompt
)

$ErrorActionPreference = 'Stop'

$patterns = @('run_live_session', 'bot_dashboard', 'webhook_server')

$matches = Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $cmd = $_.CommandLine
    if (-not $cmd) { return $false }
    foreach ($p in $patterns) {
        if ($cmd -match $p) { return $true }
    }
    return $false
}

if (-not $matches -or $matches.Count -eq 0) {
    Write-Host "No live-trading python processes found." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Candidate processes to stop:" -ForegroundColor Yellow
$rows = foreach ($m in $matches) {
    $proc = Get-Process -Id $m.ProcessId -ErrorAction SilentlyContinue
    $startTime = if ($proc) { $proc.StartTime } else { '?' }
    $cmd = $m.CommandLine
    $frag = if ($cmd.Length -gt 90) { $cmd.Substring(0, 90) + '...' } else { $cmd }
    [PSCustomObject]@{
        PID       = $m.ProcessId
        StartTime = $startTime
        Command   = $frag
    }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host

if (-not $NoPrompt) {
    $answer = Read-Host "Kill these $($rows.Count) process(es)? (y/N)"
    if ($answer -notmatch '^[yY]') {
        Write-Host "Aborted. No processes killed." -ForegroundColor Cyan
        exit 0
    }
}

$killed = 0
foreach ($m in $matches) {
    try {
        Stop-Process -Id $m.ProcessId -Force -ErrorAction Stop
        Write-Host "  Killed PID $($m.ProcessId)" -ForegroundColor Green
        $killed++
    } catch {
        Write-Host "  FAILED PID $($m.ProcessId): $_" -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "Stopped $killed of $($matches.Count) process(es)." -ForegroundColor Green
