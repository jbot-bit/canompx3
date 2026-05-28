@echo off
title ORB Trading Bot
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
echo.

:: Keep this banner static. Dynamic Git probing in this launcher can break
:: double-click startup on Windows; use docs/reference/start-bot-checkout-model.md
:: for the branch/merge model instead.
echo [Repo] %CD%
echo [Repo] This shortcut runs the Windows checkout above.
echo [Repo] WSL/Codex branch pushes do not change this app until merged or pulled here.
echo.

:: Weekend guard — don't start on Saturday (7) or Sunday (1)
for /f "tokens=1" %%d in ('powershell -command "(Get-Date).DayOfWeek.value__"') do set DOW=%%d
if "%DOW%"=="0" (
    echo [WARNING] It's SUNDAY. Markets are closed.
    echo Press any key to start anyway, or Ctrl+C to cancel.
    pause
)
if "%DOW%"=="6" (
    echo [WARNING] It's SATURDAY. Markets are closed.
    echo Press any key to start anyway, or Ctrl+C to cancel.
    pause
)

:: Active profile — change ACTIVE_PROFILE to deploy a different account profile.
:: See trading_app/prop_profiles.py ACCOUNT_PROFILES for the registry.
set ACTIVE_PROFILE=topstep_50k_mnq_auto

:: Default mode is SIGNAL (no broker orders). For demo/live, edit BOT_MODE_FLAGS.
:: Multi-instrument profile + --demo/--live is rejected by run_live_session.py.
:: 2026-05-28: --demo for Stage 4 NYSE_OPEN order-path smoke (TopstepX practice
:: endpoint, ZERO real capital). After smoke green, revert to --signal-only
:: until Stage 5 explicitly flips to --live. See
:: docs/runtime/stages/2026-05-28-live-golive-stage4-demo-smoke.md.
set BOT_MODE_FLAGS=--demo

:: Step 1: Clean up stale lock + stop files (don't kill python — other terminals may be running)
::   The .stop file triggers graceful-shutdown on the next feed scan; a stale one
::   from a previous abort will kill a fresh orchestrator within ~60s.
echo [1/5] Removing stale lock + stop files...
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1
del /f /q "live_session.stop" >nul 2>&1

:: Step 2: Clear stale bot state so dashboard shows clean STOPPED until orchestrator writes fresh data
echo [2/5] Clearing stale state...
del /f /q "data\bot_state.json" >nul 2>&1
.venv\Scripts\python.exe scripts\tools\sweep_orphan_rings.py >nul 2>&1

:: Step 2b: Publish the canonical planned-launch surface so the dashboard can
:: render the unambiguous "Next launch: <MODE> · <profile> · <instruments> ·
:: N broker accounts" banner BEFORE the orchestrator boots. The orchestrator's
:: own state (bot_state.json) supersedes this once running. Schema/mode is
:: validated by trading_app/live/planned_launch.py — bad inputs refuse to write.
if /i "%BOT_MODE_FLAGS%"=="--signal-only" set PLANNED_MODE=SIGNAL
if /i "%BOT_MODE_FLAGS%"=="--demo" set PLANNED_MODE=DEMO
if /i "%BOT_MODE_FLAGS%"=="--live" set PLANNED_MODE=LIVE
if not defined PLANNED_MODE set PLANNED_MODE=SIGNAL
.venv\Scripts\python.exe -m trading_app.live.planned_launch write --profile %ACTIVE_PROFILE% --mode %PLANNED_MODE% --source START_BOT.bat >nul

:: Step 3: Data freshness check
echo [3/5] Checking data freshness...
.venv\Scripts\python.exe -c "from pipeline.paths import GOLD_DB_PATH; import duckdb; con=duckdb.connect(str(GOLD_DB_PATH),read_only=True); r=con.execute('SELECT MAX(trading_day) FROM daily_features WHERE orb_minutes=5').fetchone(); print(f'  Latest daily_features: {r[0]}'); con.close()" 2>nul
echo.

:: Step 4: Launch orchestrator (trading engine) in a separate console window.
:: This is the process that connects to the broker, watches bars, evaluates entries,
:: and writes bot_state.json. Without it the dashboard reads stale state.
echo [4/5] Launching orchestrator: profile=%ACTIVE_PROFILE% %BOT_MODE_FLAGS%
:: Suppress the orchestrator's own dashboard spawn — this bat opens the dashboard
:: directly on line 78. Without this env var, two dashboards + two browser tabs
:: open and race for live_journal.db (fails preflight check 6).
start "ORB Orchestrator (%ACTIVE_PROFILE%)" /min cmd /k "set CANOMPX3_DASHBOARD_ORIGIN=1 && .venv\Scripts\python.exe -m scripts.run_live_session --profile %ACTIVE_PROFILE% %BOT_MODE_FLAGS%"

:: Step 5: Launch dashboard + open browser in this (main) console.
:: Closing this window stops the dashboard; the orchestrator keeps running until
:: you also close its minimised window (or use the dashboard's kill endpoint).
echo [5/5] Launching dashboard...
echo.
echo ============================================
echo   Profile:   %ACTIVE_PROFILE% %BOT_MODE_FLAGS%
echo   Dashboard: http://localhost:8080
echo   TopStepX:  https://app.topstepx.com
echo   Press Ctrl+C to stop
echo ============================================
echo.

:: Open browser after 2 second delay (gives server time to start)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

:: Minimise THIS console ~3s after launch so the browser is the visible surface.
:: Dashboard process keeps running here; closing this window still stops it
:: (see Step 5 contract above). FindWindow-by-title uses the `title` set on line 2.
:: Fail-open: any PowerShell error leaves the console as-is.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; try { Add-Type -Name W -Namespace U -MemberDefinition '[System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern System.IntPtr FindWindow(string c, string w); [System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern bool ShowWindowAsync(System.IntPtr h, int n);'; $h=[U.W]::FindWindow($null,'ORB Trading Bot'); if ($h -ne [System.IntPtr]::Zero) { [void][U.W]::ShowWindowAsync($h,6) } } catch {}"

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
