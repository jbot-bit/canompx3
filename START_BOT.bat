@echo off
:: Per-launch unique title suffix: prevents FindWindow collision in the
:: auto-minimise step (line ~104) when a stale or sibling START_BOT
:: console shares the title. %RANDOM% is 0-32767 per cmd instance;
:: doubled gives ~1 in 1.07B collision odds — sufficient for single-user
:: same-machine scope.
set BOT_TITLE=ORB Trading Bot [%RANDOM%%RANDOM%]
title %BOT_TITLE%
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

:: Market-closed guard — canonical truth, NOT the local weekday.
:: The machine runs Brisbane time (UTC+10), so it is "Saturday" locally during
:: the entire Friday US session when the CME market is actually OPEN. A naive
:: (Get-Date).DayOfWeek check warned "markets closed" mid-session. Delegate to
:: pipeline.market_calendar.is_market_open_at (the same source the orchestrator
:: uses) so the warning fires ONLY when the market is genuinely closed.
::   prints OPEN / CLOSED; fail-open (prints OPEN) on any error so the guard
::   never blocks a launch it cannot evaluate.
for /f "tokens=1" %%s in ('.venv\Scripts\python.exe -c "from datetime import datetime; from zoneinfo import ZoneInfo; from pipeline.market_calendar import is_market_open_at; print('OPEN' if is_market_open_at(datetime.now(ZoneInfo('UTC'))) else 'CLOSED')" 2^>nul') do set MKT=%%s
if "%MKT%"=="CLOSED" (
    echo [WARNING] CME market is currently CLOSED ^(weekend gap / holiday^).
    echo Press any key to start anyway, or Ctrl+C to cancel.
    pause
)

:: Active profile — change ACTIVE_PROFILE to deploy a different account profile.
:: See trading_app/prop_profiles.py ACCOUNT_PROFILES for the registry.
set ACTIVE_PROFILE=topstep_50k_mnq_auto

:: Default mode is SIGNAL (no broker orders). START_BOT is the control-room
:: entrypoint; demo/live starts are initiated from the dashboard after gates
:: run. Do not set --live here.
set BOT_MODE_FLAGS=--signal-only

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
.venv\Scripts\python.exe -m trading_app.live.planned_launch write --profile %ACTIVE_PROFILE% --mode %PLANNED_MODE% --source START_BOT.bat --copies 1 --instrument MNQ >nul

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
:: This console auto-minimises ~3s after launch (line ~107). To stop the
:: dashboard: restore the console from the taskbar, then press Ctrl+C, or
:: close the window. Orchestrator window stays /min until closed separately
:: (or use the dashboard's kill endpoint).
echo [5/5] Launching dashboard...
echo.
echo ============================================
echo   Profile:   %ACTIVE_PROFILE% %BOT_MODE_FLAGS%
echo   Dashboard: http://localhost:8080
echo   TopStepX:  https://app.topstepx.com
echo   To stop:   restore this window from taskbar, then Ctrl+C
echo ============================================
echo.

:: Open browser after 2 second delay (gives server time to start)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

:: Minimise THIS console ~3s after launch so the browser is the visible surface.
:: Dashboard process keeps running here; restoring the console + Ctrl+C still
:: stops it (see Step 5 contract above). FindWindow-by-title targets %BOT_TITLE%
:: (set with unique per-launch suffix on line 2) to avoid collision with stale
:: or sibling START_BOT consoles. Fail-open: any PowerShell error leaves the
:: console as-is.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; try { Add-Type -Name W -Namespace U -MemberDefinition '[System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern System.IntPtr FindWindow(string c, string w); [System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern bool ShowWindowAsync(System.IntPtr h, int n);'; $h=[U.W]::FindWindow($null,'%BOT_TITLE%'); if ($h -ne [System.IntPtr]::Zero) { [void][U.W]::ShowWindowAsync($h,6) } } catch {}"

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
