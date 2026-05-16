@echo off
title ORB Trading Bot
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
echo.

:: Show exactly which checkout this shortcut is serving. This prevents the
:: Windows START_BOT shortcut from looking identical to a newer WSL/Codex
:: branch that has not been merged or pulled into this checkout yet.
set "GIT_BRANCH=<unknown>"
set "GIT_HEAD=<unknown>"
set "GIT_UPSTREAM=<none>"
set "GIT_AHEAD=0"
set "GIT_BEHIND=0"
for /f "usebackq delims=" %%b in (`git branch --show-current 2^>nul`) do set "GIT_BRANCH=%%b"
for /f "usebackq delims=" %%h in (`git rev-parse --short HEAD 2^>nul`) do set "GIT_HEAD=%%h"
for /f "usebackq delims=" %%u in (`git rev-parse --abbrev-ref --symbolic-full-name @{u} 2^>nul`) do set "GIT_UPSTREAM=%%u"
for /f "usebackq tokens=1,2" %%a in (`git rev-list --left-right --count HEAD...@{u} 2^>nul`) do (
    set "GIT_AHEAD=%%a"
    set "GIT_BEHIND=%%b"
)
echo [Repo] %CD%
echo [Repo] branch=%GIT_BRANCH% commit=%GIT_HEAD% upstream=%GIT_UPSTREAM% ahead=%GIT_AHEAD% behind=%GIT_BEHIND%
if not "%GIT_BEHIND%"=="0" (
    echo.
    echo [WARNING] This START_BOT checkout is behind its upstream by %GIT_BEHIND% commit(s).
    echo           Pull or merge before expecting recently pushed dashboard changes to appear here.
)
echo [Note] Codex WSL branch work does not change this Windows shortcut until it is merged or pulled here.
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

:: Step 1: Clean up stale lock files (don't kill python — other terminals may be running)
echo [1/4] Removing stale lock files...
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1

:: Step 2: Clear stale bot state so dashboard shows clean STOPPED
echo [2/4] Clearing stale state...
del /f /q "data\bot_state.json" >nul 2>&1

:: Step 3: Data freshness check
echo [3/4] Checking data freshness...
.venv\Scripts\python.exe -c "from pipeline.paths import GOLD_DB_PATH; import duckdb; con=duckdb.connect(str(GOLD_DB_PATH),read_only=True); r=con.execute('SELECT MAX(trading_day) FROM daily_features WHERE orb_minutes=5').fetchone(); print(f'  Latest daily_features: {r[0]}'); con.close()" 2>nul
echo.

:: Step 4: Launch dashboard + open browser
echo [4/4] Launching dashboard...
echo.
echo ============================================
echo   Dashboard: http://localhost:8080
echo   TopStepX:  https://app.topstepx.com
echo   Press Ctrl+C to stop
echo ============================================
echo.

:: Open browser after 2 second delay (gives server time to start)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
