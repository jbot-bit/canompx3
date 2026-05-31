@echo off
setlocal
title Topstep MNQ Live Pilot
color 0C
cd /d "%~dp0"

echo ============================================
echo   TOPSTEP MNQ LIVE PILOT
echo ============================================
echo.
echo Profile:    topstep_50k_mnq_auto
echo Instrument: MNQ
echo Copies:     1
echo.
echo This launcher runs the readiness gates first.
echo If all gates pass, the live runner will still require typing CONFIRM.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo BLOCKED: .venv\Scripts\python.exe not found.
    echo Run the project setup first, then rerun this launcher.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\tools\start_topstep_live_pilot.py
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
    echo Live pilot launcher exited with code %EXITCODE%.
) else (
    echo Live pilot launcher exited cleanly.
)
pause
exit /b %EXITCODE%
