@echo off
setlocal
title Codex Workstream Shortcut
cd /d "%~dp0\..\..\.."

set "TASK=%~1"
if "%TASK%"=="" (
    set /p "TASK=Codex workstream name: "
)

if "%TASK%"=="" (
    echo Workstream name required.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode codex -Task "%TASK%"
