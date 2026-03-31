@echo off
title ORB Trading Bot
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
echo.

:: Step 1: Clean up stale lock files (don't kill python — other terminals may be running)
echo [1/3] Removing stale lock files...
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1

:: Step 2: Clear stale bot state so dashboard shows clean STOPPED
echo [2/3] Clearing stale state...
del /f /q "bot_state.json" >nul 2>&1

:: Step 3: Launch dashboard + open browser
echo [3/3] Launching dashboard...
echo.
echo ============================================
echo   Dashboard: http://localhost:8080
echo   Press Ctrl+C to stop
echo ============================================
echo.

:: Open browser after 2 second delay (gives server time to start)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
