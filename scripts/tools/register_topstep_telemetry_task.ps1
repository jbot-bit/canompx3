param(
    [string]$RepoRoot = "C:\Users\joshd\canompx3",
    [string]$DuckDbPath = "C:\Users\joshd\canompx3\gold.db",
    [string]$TaskName = "CanonMPX_TopstepTelemetry_SignalOnly",
    [string]$At = "22:20"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$script = Join-Path $RepoRoot "scripts\run_live_session.py"
$log = "logs\telemetry_accrual_signal_only.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found at $python"
}
if (-not (Test-Path -LiteralPath $script)) {
    throw "run_live_session.py not found at $script"
}
if (-not (Test-Path -LiteralPath $DuckDbPath)) {
    throw "gold.db not found at $DuckDbPath"
}

$cmd = "cd /d $RepoRoot && set DUCKDB_PATH=$DuckDbPath && set CANOMPX3_DASHBOARD_ORIGIN=1 && .venv\Scripts\python.exe scripts\run_live_session.py --profile topstep_50k_mnq_auto --signal-only >> $log 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$cmd`""
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable:$false

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Start canompx3 topstep_50k_mnq_auto in signal-only mode daily to accrue profile-scoped telemetry without opening dashboard or placing orders." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Format-List TaskName,State,Description
Get-ScheduledTaskInfo -TaskName $TaskName | Format-List LastRunTime,LastTaskResult,NextRunTime
