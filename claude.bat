@echo off
setlocal
title Claude Code
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "ACTION=%~1"

if /I "%ACTION%"=="task" (
    shift
    call "ai-workstreams.bat" claude %*
    exit /b %ERRORLEVEL%
) else if /I "%ACTION%"=="green" (
    call "ai-workstreams.bat" green claude
    exit /b %ERRORLEVEL%
) else if /I "%ACTION%"=="help" (
    echo Usage:
    echo   claude.bat                Launch Claude Code
    echo   claude.bat task ^<name^>    Start a Claude workstream
    echo   claude.bat green          Run green baseline ^(Claude^)
    exit /b 0
)

:: Default: launch Claude Code CLI in this project
claude
