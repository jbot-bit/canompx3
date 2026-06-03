<#
.SYNOPSIS
  Register a DAILY scheduled task running git_hygiene.ps1 — local branch/worktree
  pruning, lease-guarded, for unattended runs.

.NOTES
  One-shot setup. Built 2026-06-03. Idempotent (-Force replaces existing).
  Runs daily at 04:00 local; StartWhenAvailable catches up if the PC was off.
  Complements GitHub's server-side delete_branch_on_merge (remote side) and the
  interactive commit-commands:clean_gone plugin (at-keyboard side).
#>

$ErrorActionPreference = 'Stop'

$TaskName = 'CanonMPX_GitHygiene'
$Script   = 'C:\Users\joshd\canompx3\scripts\tools\git_hygiene.ps1'

if (-not (Test-Path $Script)) {
    throw "git_hygiene.ps1 not found at $Script (must be merged to main first)."
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`""

$trigger = New-ScheduledTaskTrigger -Daily -At 4am

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description 'Daily local git hygiene: fetch --prune, worktree prune, delete [gone] local branches. Lease-guarded; skips while a live peer holds main.' `
    -Force | Out-Null

Write-Output "Registered scheduled task: $TaskName (daily 04:00, survives reboot)"
Write-Output "  Log: C:\Users\joshd\canompx3\docs\runtime\git_hygiene.log"
