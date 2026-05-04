@echo off
setlocal
title ask
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "scripts\tools\ask.py" %*
set "EXITCODE=%ERRORLEVEL%"

REM Keep window open if double-clicked with no args (REPL exits cleanly anyway).
if "%~1"=="" if not "%EXITCODE%"=="0" (
    echo.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
