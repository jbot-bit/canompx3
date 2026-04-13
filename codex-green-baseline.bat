@echo off
setlocal
title Codex Green Baseline
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "WT=%~dp0.worktrees\tasks\green-baseline"
if not exist "%WT%" (
    echo Clean green-baseline worktree not found:
    echo   %WT%
    echo.
    echo Recreate it from WSL or ask Codex to rebuild it.
    pause
    exit /b 1
)

wsl.exe bash -lc "cd '/mnt/c/Users/joshd/canompx3/.worktrees/tasks/green-baseline' && exec ./scripts/infra/codex-project.sh --no-alt-screen"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Codex green-baseline launch failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
