# Register weekly Task Scheduler job for gold.db backup.
# Run as Administrator:  powershell.exe -ExecutionPolicy Bypass -File .\scripts\tools\register_gold_db_backup_task.ps1
#
# Safe to re-run: unregisters any prior copy first.

$ErrorActionPreference = "Stop"

$taskName = "GoldDB-WeeklyBackup"
$python   = "C:\Users\joshd\canompx3\.venv\Scripts\python.exe"
$script   = "C:\Users\joshd\canompx3\scripts\tools\backup_gold_db.py"
$workdir  = "C:\Users\joshd\canompx3"

if (-not (Test-Path $python)) { throw "python.exe not found: $python" }
if (-not (Test-Path $script)) { throw "backup script not found: $script" }

# Remove existing task if present (idempotent re-run).
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$taskName'..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action   = New-ScheduledTaskAction -Execute $python -Argument $script -WorkingDirectory $workdir
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable

Register-ScheduledTask `
    -TaskName    $taskName `
    -Action      $action `
    -Trigger     $trigger `
    -Settings    $settings `
    -RunLevel    Highest `
    -Description "Weekly gzipped snapshot of gold.db to C:\backups\gold-db (keeps last 4)"

Write-Host ""
Write-Host "Registered. Verify with:"
Write-Host "  Get-ScheduledTask -TaskName $taskName"
Write-Host "Test-fire with:"
Write-Host "  Start-ScheduledTask -TaskName $taskName"
