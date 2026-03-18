@echo off
title AI Workstreams
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode menu
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo AI Workstreams failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
