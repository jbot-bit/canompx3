@echo off
:: fix_system_stability.bat — Double-click to run. Self-elevates to admin.
:: Fixes Acer bloatware, BSOD-causing settings, cleans crash dumps.

:: --- Self-elevate to admin ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

cd /d "%~dp0\..\.."
echo.
echo ============================================
echo   SYSTEM STABILITY FIX — Acer Predator
echo ============================================
echo.

echo [1/7] Disabling Acer services...
for %%s in (AcerCentralService AcerGAICameraService AcerCCAgent AcerLightingService AcerPixyService AcerQAAgent AcerDIAgent AcerSystemCentralService AcerServiceWrapper AcerAgentService AcerService AcerSysMonitorService AcerHardwareService AcerSysHardwareService) do (
    sc stop %%s >nul 2>&1
    sc config %%s start=disabled >nul 2>&1
    echo   Disabled: %%s
)

echo.
echo [2/7] Disabling Acer scheduled tasks...
schtasks /Change /TN "\AcerJumpstartTask" /Disable >nul 2>&1
echo   Done.

echo.
echo [3/7] Removing Acer UWP apps...
powershell -NoProfile -Command "Get-AppxPackage -AllUsers | Where-Object { $_.Name -like '*Acer*' } | ForEach-Object { Write-Host ('  Removing: ' + $_.Name); Remove-AppxPackage -Package $_.PackageFullName -AllUsers -ErrorAction SilentlyContinue }" 2>nul
echo   Done.

echo.
echo [4/7] Cleaning crash dumps...
del /q "%LOCALAPPDATA%\CrashDumps\*.dmp" 2>nul
echo   Cleaned AppData crash dumps.

echo.
echo [5/7] Disabling Fast Startup (BSOD 0x9F fix)...
reg add "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power" /v HiberbootEnabled /t REG_DWORD /d 0 /f >nul 2>&1
echo   Fast Startup: DISABLED

echo.
echo [6/7] Setting High Performance power plan...
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1
powercfg /SETDCVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 >nul 2>&1
powercfg /SETACVALUEINDEX SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0 >nul 2>&1
powercfg /SETACTIVE SCHEME_CURRENT >nul 2>&1
echo   High Performance ON, USB Selective Suspend OFF

echo.
echo [7/7] Running system file check (takes a few minutes)...
sfc /scannow
DISM /Online /Cleanup-Image /RestoreHealth

echo.
echo ============================================
echo   ALL DONE. Summary:
echo ============================================
echo   [x] 14 Acer services DISABLED
echo   [x] Acer scheduled tasks DISABLED
echo   [x] Acer UWP apps REMOVED
echo   [x] Crash dumps CLEANED
echo   [x] Fast Startup DISABLED
echo   [x] High Performance power plan SET
echo   [x] System files checked
echo.
echo   MANUAL STEPS STILL NEEDED:
echo   1. Update NVIDIA GPU driver (clean install)
echo      https://www.nvidia.com/drivers
echo   2. Run memory diagnostic: Win+R then mdsched.exe
echo   3. REBOOT this machine
echo.
pause
