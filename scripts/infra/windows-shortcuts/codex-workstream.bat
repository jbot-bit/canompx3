@echo off
setlocal
title Codex Isolated Workstream Shortcut
cd /d "%~dp0\..\..\.."

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
