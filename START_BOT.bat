@echo off
title ORB Trading Bot
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
echo.

:: Step 1: Kill any stale python processes holding DB locks
echo [1/4] Cleaning up stale processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: Step 2: Clean up stale lock files
echo [2/4] Removing stale lock files...
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1

:: Step 3: Clear stale bot state so dashboard shows clean STOPPED
echo [3/4] Clearing stale state...
del /f /q "data\bot_state.json" >nul 2>&1

:: Step 4: Launch dashboard (opens browser automatically)
echo [4/4] Launching dashboard...
echo.
echo ============================================
echo   Dashboard: http://localhost:8080
echo   Press Ctrl+C to stop
echo ============================================
echo.

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
