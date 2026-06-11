<#
.SYNOPSIS
  Register a scheduled task that auto-merges a verified fix branch to main.

.DESCRIPTION
  This is the reusable version of the one-off reaper merge task. Use it after a
  scheduled automation has produced and verified a fix branch. The registered
  task retries until auto_merge_fix_branch.ps1 can fast-forward origin/main, then
  unregisters itself.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\tools\register_auto_merge_fix_task.ps1 `
    -Branch codex/daily-bug-scan-survival-sweep-guard `
    -Worktree C:\Users\joshd\.codex\worktrees\cf1c\canompx3 `
    -PushBranch
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Branch,

    [Parameter(Mandatory = $true)]
    [string]$Worktree,

    [string]$MainRepo = 'C:\Users\joshd\canompx3',

    [string]$TaskName = '',

    [int]$FirstRunMinutes = 5,

    [int]$IntervalMinutes = 15,

    [switch]$PushBranch,

    [switch]$DeleteBranchOnSuccess,

    [switch]$RemoveWorktreeOnSuccess,

    [string[]]$AllowedDirtyPath = @('HANDOFF.md'),

    [switch]$NoAutoRebase
)

$ErrorActionPreference = 'Stop'

if (-not $TaskName) {
    $safeBranch = ($Branch -replace '[^A-Za-z0-9_-]', '_')
    $TaskName = "CanonMPX_AutoMerge_$safeBranch"
}

$Script = Join-Path $MainRepo 'scripts\tools\auto_merge_fix_branch.ps1'
if (-not (Test-Path $Script)) {
    $localScript = Join-Path $PSScriptRoot 'auto_merge_fix_branch.ps1'
    if (Test-Path $localScript) {
        $Script = $localScript
    } else {
        throw "merge script not found: $Script"
    }
}

if (-not (Test-Path $Worktree)) { throw "fix worktree not found: $Worktree" }

$LogFile = Join-Path $MainRepo 'docs\runtime\auto_merge_fix_branch.log'
$runnerArgs = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-WindowStyle', 'Hidden',
    '-File', "`"$Script`"",
    '-Branch', "`"$Branch`"",
    '-Worktree', "`"$Worktree`"",
    '-MainRepo', "`"$MainRepo`"",
    '-TaskName', "`"$TaskName`"",
    '-LogFile', "`"$LogFile`""
)
if ($PushBranch) { $runnerArgs += '-PushBranch' }
if ($DeleteBranchOnSuccess) { $runnerArgs += '-DeleteBranchOnSuccess' }
if ($RemoveWorktreeOnSuccess) { $runnerArgs += '-RemoveWorktreeOnSuccess' }
if (-not $NoAutoRebase) { $runnerArgs += '-AutoRebase' }
foreach ($path in $AllowedDirtyPath) {
    $runnerArgs += @('-AllowedDirtyPath', "`"$path`"")
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ($runnerArgs -join ' ')

$trigger = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes($FirstRunMinutes)) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Auto-merge verified fix branch '$Branch' into origin/main when it is a clean fast-forward." `
    -Force | Out-Null

Write-Output "Registered scheduled task: $TaskName"
Write-Output "  Branch: $Branch"
Write-Output "  Worktree: $Worktree"
Write-Output "  First run: ~$FirstRunMinutes min from now, then every $IntervalMinutes min."
Write-Output "  Auto-rebase: $(-not $NoAutoRebase)"
Write-Output "  Log: $LogFile"
