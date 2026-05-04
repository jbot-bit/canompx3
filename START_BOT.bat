@echo off
title ORB Trading Bot
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
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
