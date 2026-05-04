@echo off
setlocal
title AI Workstreams
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "ACTION=%~1"
if "%ACTION%"=="" goto gui

if /I "%ACTION%"=="menu" goto gui
if /I "%ACTION%"=="list" call :run_mode list & goto end
if /I "%ACTION%"=="resume" call :run_mode resume & goto end
if /I "%ACTION%"=="finish" call :run_mode close-pick & goto end
if /I "%ACTION%"=="clean" call :run_mode prune & goto end
if /I "%ACTION%"=="claude" goto claude_task
if /I "%ACTION%"=="codex" goto codex_task
if /I "%ACTION%"=="search" goto search_task
if /I "%ACTION%"=="green" goto green_task

set "TASK=%*"
call :run_task codex "%TASK%"
goto end

:gui
call :run_gui
goto end

:claude_task
set "TASK=%~2"
call :run_task claude "%TASK%"
goto end

:codex_task
set "TASK=%~2"
call :run_task codex "%TASK%"
goto end

:search_task
set "TASK=%~2"
call :run_task codex-search "%TASK%"
goto end

:green_task
if /I "%~2"=="claude" (
    call :run_mode green-claude
) else (
    call :run_mode green-codex
)
goto end

:run_task
set "MODE=%~1"
set "TASK=%~2"
if "%TASK%"=="" (
    set /p "TASK=Workstream name: "
)
if "%TASK%"=="" (
    echo Workstream name required.
    set "EXITCODE=1"
    goto :eof
)
if defined CANOMPX3_WINDOWS_LAUNCH_ECHO_ONLY (
    echo MODE=%MODE% TASK=%TASK%
    set "EXITCODE=0"
    goto :eof
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode %MODE% -Task "%TASK%"
set "EXITCODE=%ERRORLEVEL%"
goto :eof

:run_mode
set "MODE=%~1"
if defined CANOMPX3_WINDOWS_LAUNCH_ECHO_ONLY (
    echo MODE=%MODE%
    set "EXITCODE=0"
    goto :eof
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode %MODE%
set "EXITCODE=%ERRORLEVEL%"
goto :eof

:run_gui
if defined CANOMPX3_WINDOWS_LAUNCH_ECHO_ONLY (
    echo GUI=1
    set "EXITCODE=0"
    goto :eof
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-workstreams-gui.ps1"
set "EXITCODE=%ERRORLEVEL%"
goto :eof

:end
if not defined EXITCODE set "EXITCODE=0"
if not "%EXITCODE%"=="0" (
    echo.
    echo AI Workstreams failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
