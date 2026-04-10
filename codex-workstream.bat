@echo off
setlocal
title Codex Isolated Workstream
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "TASK=%~1"
if "%TASK%"=="" (
    set /p "TASK=Codex isolated workstream name: "
)

if "%TASK%"=="" (
    echo Isolated workstream name required.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode codex -Task "%TASK%"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Codex isolated workstream failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
