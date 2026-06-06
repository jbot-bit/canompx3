<#
.SYNOPSIS
  Register a 30-min scheduled task that runs auto_merge_reaper_prefix_guard.ps1
  until the fix branch is merged into main (then the task self-unregisters).

.NOTES
  One-shot setup. Built 2026-06-03. Run once; the merge then happens hands-free.
  Idempotent: re-running replaces any existing task of the same name.
#>

$ErrorActionPreference = 'Stop'

$TaskName = 'CanonMPX_AutoMerge_ReaperPrefixGuard'
$Script   = 'C:\Users\joshd\canompx3-reaper-prefix-guard\scripts\tools\auto_merge_reaper_prefix_guard.ps1'

if (-not (Test-Path $Script)) { throw "merge script not found: $Script" }

$action  = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`""

# Fire 5 min after registration, then every 30 min, indefinitely (until the
# script unregisters the task on success). Survives reboot via the trigger.
$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(5)) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

# Run as the current user so git creds / SSH agent resolve exactly as in a normal shell.
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description 'Auto-FF-merge session/joshd-reaper-prefix-guard into main when peer lease is free and FF is clean. Self-unregisters on success.' `
    -Force | Out-Null

Write-Output "Registered scheduled task: $TaskName"
Write-Output "  First run: ~5 min from now, then every 30 min."
Write-Output "  Survives reboot. Self-removes after a successful merge."
Write-Output "  Log: C:\Users\joshd\canompx3\docs\runtime\auto_merge_reaper_prefix_guard.log"
