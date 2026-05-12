@echo off
setlocal
title Codex
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "ACTION=%~1"
set "MODE=codex-project-linux"

if /I "%ACTION%"=="power" (
    set "MODE=codex-project-linux-power"
) else if /I "%ACTION%"=="gold-db" (
    set "MODE=codex-project-linux-gold-db"
) else if /I "%ACTION%"=="search-gold-db" (
    set "MODE=codex-project-linux-search-gold-db"
) else if /I "%ACTION%"=="windows" (
    set "MODE=codex-project"
) else if /I "%ACTION%"=="windows-power" (
    set "MODE=codex-project-power"
) else if /I "%ACTION%"=="linux" (
    set "MODE=codex-project-linux"
) else if /I "%ACTION%"=="linux-power" (
    set "MODE=codex-project-linux-power"
) else if /I "%ACTION%"=="linux-gold-db" (
    set "MODE=codex-project-linux-gold-db"
) else if /I "%ACTION%"=="green" (
    set "MODE=green-codex"
) else if /I "%ACTION%"=="doctor" (
    set "MODE=doctor"
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
    echo   codex.bat power
    echo   codex.bat gold-db
    echo   codex.bat search-gold-db
    echo   codex.bat search ^<name^>
    echo   codex.bat task ^<name^>
    echo.
    echo Advanced compatibility modes:
    echo   codex.bat windows
    echo   codex.bat windows-power
    echo   codex.bat linux
    echo   codex.bat linux-power
    echo   codex.bat linux-gold-db
    echo   codex.bat green
    echo   codex.bat doctor
    exit /b 0
)

if not "%ACTION%"=="" if /I not "%ACTION%"=="power" if /I not "%ACTION%"=="gold-db" if /I not "%ACTION%"=="search-gold-db" if /I not "%ACTION%"=="windows" if /I not "%ACTION%"=="windows-power" if /I not "%ACTION%"=="linux" if /I not "%ACTION%"=="linux-power" if /I not "%ACTION%"=="linux-gold-db" if /I not "%ACTION%"=="green" if /I not "%ACTION%"=="doctor" if /I not "%ACTION%"=="task" if /I not "%ACTION%"=="search" if /I not "%ACTION%"=="help" (
    echo Unknown codex mode: %ACTION%
    echo Run `codex.bat help` for usage.
    exit /b 2
)

if defined CANOMPX3_WINDOWS_LAUNCH_INLINE (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode %MODE%
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-sticky-launch.ps1" -Mode %MODE% -Title "Codex"
)
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Codex launch failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
