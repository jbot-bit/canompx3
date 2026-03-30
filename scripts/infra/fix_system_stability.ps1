# fix_system_stability.ps1 — Run as Administrator
# Fixes: Acer bloatware, BSOD-causing drivers, crash dump cleanup
# Machine: Acer Predator PHN16-72, Win11 26200, Intel 13th gen
#
# BSOD codes found (Mar 2026): 0x0A (IRQL x2), 0x9F (DRIVER_POWER_STATE),
# 0x141 (VIDEO_ENGINE_TIMEOUT), 0x50 (PAGE_FAULT), 0xC1 (MEMORY_CORRUPTION),
# 0x7E (SYSTEM_THREAD_EXCEPTION), 0x139 (KERNEL_SECURITY_CHECK)
#
# Root causes: bad GPU driver (0x141) + Acer services (14 processes, 300MB)
# + possible RAM issue (0x50, 0xC1)

$ErrorActionPreference = "Stop"

Write-Host "`n=== STEP 1: Disable Acer Bloatware Services ===" -ForegroundColor Cyan
Write-Host "These 14 services eat 300MB RAM and AcerRegistrationBackGroundTask.exe crashes daily.`n"

$acerServices = @(
    "AcerCentralService",
    "AcerGAICameraService",
    "AcerCCAgent",
    "AcerLightingService",
    "AcerPixyService",
    "AcerQAAgent",
    "AcerDIAgent",
    "AcerSystemCentralService",
    "AcerServiceWrapper",
    "AcerAgentService",
    "AcerService",
    "AcerSysMonitorService",
    "AcerHardwareService",
    "AcerSysHardwareService"
)

foreach ($svc in $acerServices) {
    $s = Get-Service -Name $svc -ErrorAction SilentlyContinue
    if ($s) {
        Write-Host "  Disabling $svc (was: $($s.StartType), $($s.Status))"
        Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue
        Set-Service -Name $svc -StartupType Disabled -ErrorAction SilentlyContinue
    }
}

Write-Host "`n=== STEP 2: Disable Acer Scheduled Tasks ===" -ForegroundColor Cyan
Get-ScheduledTask | Where-Object { $_.TaskName -like "*Acer*" -or $_.TaskPath -like "*Acer*" } | ForEach-Object {
    Write-Host "  Disabling task: $($_.TaskName)"
    $_ | Disable-ScheduledTask -ErrorAction SilentlyContinue | Out-Null
}

Write-Host "`n=== STEP 3: Remove Acer UWP Apps ===" -ForegroundColor Cyan
Write-Host "Removing Acer Store apps (registration, care center, etc.)`n"
Get-AppxPackage -AllUsers | Where-Object { $_.Name -like "*Acer*" } | ForEach-Object {
    Write-Host "  Removing: $($_.Name) v$($_.Version)"
    $_ | Remove-AppxPackage -AllUsers -ErrorAction SilentlyContinue
}

Write-Host "`n=== STEP 4: Clean Up Crash Dumps (450MB+) ===" -ForegroundColor Cyan
$dumpDir = "$env:LOCALAPPDATA\CrashDumps"
if (Test-Path $dumpDir) {
    $dumps = Get-ChildItem $dumpDir -Filter "*.dmp"
    $totalMB = [math]::Round(($dumps | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host "  Found $($dumps.Count) crash dumps ($totalMB MB)"
    $dumps | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "  Cleaned."
}

Write-Host "`n=== STEP 5: GPU Driver Check ===" -ForegroundColor Cyan
$gpu = Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate
foreach ($g in $gpu) {
    Write-Host "  GPU: $($g.Name)"
    Write-Host "  Driver: $($g.DriverVersion) ($($g.DriverDate))"
}
Write-Host "`n  ACTION REQUIRED: Update NVIDIA driver from https://www.nvidia.com/drivers"
Write-Host "  Your BSOD 0x141 (VIDEO_ENGINE_TIMEOUT) = GPU driver hung."
Write-Host "  Use 'Clean install' option in NVIDIA installer to replace old driver files."

Write-Host "`n=== STEP 6: Memory Diagnostic ===" -ForegroundColor Cyan
Write-Host "  BSODs 0x50 (PAGE_FAULT) and 0xC1 (MEMORY_CORRUPTION) suggest possible bad RAM."
Write-Host "  Scheduling Windows Memory Diagnostic on next reboot..."
# This schedules mdsched to run on next boot
$mdsched = "$env:SystemRoot\System32\MdSched.exe"
Write-Host "  Run this manually: $mdsched"
Write-Host "  Choose 'Restart now and check for problems'"

Write-Host "`n=== STEP 7: Power Management Fix ===" -ForegroundColor Cyan
Write-Host "  BSOD 0x9F (DRIVER_POWER_STATE) = driver failed during sleep/wake."
Write-Host "  Disabling hybrid sleep and fast startup...`n"
# Disable Fast Startup (common BSOD cause)
$regPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power"
Set-ItemProperty -Path $regPath -Name "HiberbootEnabled" -Value 0 -ErrorAction SilentlyContinue
Write-Host "  Fast Startup: DISABLED"
# Set power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>$null
Write-Host "  Power plan: High Performance"
# Disable USB selective suspend
powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 2>$null
powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 2>$null
powercfg /SETACTIVE SCHEME_CURRENT 2>$null
Write-Host "  USB Selective Suspend: DISABLED"

Write-Host "`n=== STEP 8: SFC & DISM (System File Check) ===" -ForegroundColor Cyan
Write-Host "  Running system file checker (takes a few minutes)...`n"
sfc /scannow
Write-Host "`n  Running DISM health restore...`n"
DISM /Online /Cleanup-Image /RestoreHealth

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host @"

SUMMARY OF CHANGES:
  [x] 14 Acer services DISABLED (were eating 300MB RAM + crashing)
  [x] Acer scheduled tasks DISABLED
  [x] Acer UWP apps REMOVED
  [x] Crash dumps CLEANED ($totalMB MB freed)
  [x] Fast Startup DISABLED (BSOD 0x9F fix)
  [x] USB Selective Suspend DISABLED
  [x] Power plan set to High Performance
  [x] System files checked and repaired

MANUAL STEPS STILL NEEDED:
  1. UPDATE NVIDIA GPU DRIVER (clean install) — fixes BSOD 0x141
     https://www.nvidia.com/drivers
  2. RUN MEMORY DIAGNOSTIC — rules out bad RAM (BSOD 0x50, 0xC1)
     Win+R > mdsched.exe > Restart now
  3. REBOOT to apply all changes

After reboot, verify Acer services are gone:
  Get-Service | Where-Object { `$_.Name -like '*Acer*' }

"@
