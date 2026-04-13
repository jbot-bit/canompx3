@echo off
setlocal
title Codex
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "ACTION=%~1"
set "MODE=codex-project"

if /I "%ACTION%"=="gold-db" (
    set "MODE=codex-project-gold-db"
) else if /I "%ACTION%"=="search-gold-db" (
    set "MODE=codex-project-search-gold-db"
) else if /I "%ACTION%"=="linux" (
    set "MODE=codex-project-linux"
) else if /I "%ACTION%"=="linux-gold-db" (
    set "MODE=codex-project-linux-gold-db"
) else if /I "%ACTION%"=="green" (
    set "MODE=green-codex"
) else if /I "%ACTION%"=="task" (
    shift
    call "ai-workstreams.bat" codex %*
    exit /b %ERRORLEVEL%
) else if /I "%ACTION%"=="search" (
    shift
    call "ai-workstreams.bat" search %*
    exit /b %ERRORLEVEL%
) else if /I "%ACTION%"=="help" (
    echo Usage:
    echo   codex.bat
    echo   codex.bat gold-db
    echo   codex.bat search-gold-db
    echo   codex.bat linux
    echo   codex.bat linux-gold-db
    echo   codex.bat green
    echo   codex.bat task ^<name^>
    echo   codex.bat search ^<name^>
    exit /b 0
)

if not "%ACTION%"=="" if /I not "%ACTION%"=="gold-db" if /I not "%ACTION%"=="search-gold-db" if /I not "%ACTION%"=="linux" if /I not "%ACTION%"=="linux-gold-db" if /I not "%ACTION%"=="green" if /I not "%ACTION%"=="task" if /I not "%ACTION%"=="search" if /I not "%ACTION%"=="help" (
    echo Unknown codex mode: %ACTION%
    echo Run `codex.bat help` for usage.
    exit /b 2
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode %MODE%
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Codex launch failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
