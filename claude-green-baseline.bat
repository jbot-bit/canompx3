@echo off
setlocal
title Claude Green Baseline
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "WT=%~dp0.worktrees\tasks\green-baseline"
if not exist "%WT%" (
    echo Clean green-baseline worktree not found:
    echo   %WT%
    echo.
    echo Recreate it from WSL or ask Codex/Claude to rebuild it.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $wt=(Resolve-Path '.worktrees\tasks\green-baseline').Path; $cmd=Get-Command claude,claude.exe -ErrorAction SilentlyContinue | Select-Object -First 1; if (-not $cmd) { throw 'Claude CLI not found on PATH.' }; & $cmd.Source -C $wt"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Claude green-baseline launch failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
