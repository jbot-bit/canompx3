@echo off
setlocal
title Claude Code
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "ACTION=%~1"

if /I "%ACTION%"=="" goto launch_default
if /I "%ACTION%"=="bot" goto launch_bot
if /I "%ACTION%"=="signals" goto launch_signals
if /I "%ACTION%"=="preflight" goto launch_preflight
if /I "%ACTION%"=="pulse" goto launch_pulse
if /I "%ACTION%"=="data" goto launch_data
if /I "%ACTION%"=="drift" goto launch_drift
if /I "%ACTION%"=="task" goto launch_task
if /I "%ACTION%"=="green" goto launch_green
if /I "%ACTION%"=="help" goto show_help
goto launch_default

:show_help
echo.
echo   claude.bat             Launch Claude Code (default)
echo   claude.bat bot         Start trading bot (signal-only, TopStep)
echo   claude.bat signals     Same as bot
echo   claude.bat preflight   Run preflight checks only
echo   claude.bat pulse       Project health pulse
echo   claude.bat data        Download + build latest data (Databento)
echo   claude.bat drift       Run drift checks
echo   claude.bat task ^<name^>  Start a Claude workstream
echo   claude.bat green       Run green baseline (Claude)
echo   claude.bat help        Show this help
echo.
exit /b 0

:launch_bot
:launch_signals
echo.
echo   Starting ORB Trading Bot (signal-only)...
echo   Dashboard: http://localhost:8080
echo.
:: Kill stale locks
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1
.venv\Scripts\python.exe -m scripts.run_live_session --profile topstep_50k_mnq_auto --signal-only
goto done

:launch_preflight
.venv\Scripts\python.exe -m scripts.run_live_session --profile topstep_50k_mnq_auto --signal-only --preflight
goto done

:launch_pulse
.venv\Scripts\python.exe scripts\tools\project_pulse.py --fast
goto done

:launch_data
echo.
echo   Downloading latest data from Databento...
echo.
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="DATABENTO_API_KEY" set "DATABENTO_API_KEY=%%b"
)
.venv\Scripts\python.exe -m scripts.databento_daily --days 3
goto done

:launch_drift
.venv\Scripts\python.exe -m pipeline.check_drift
goto done

:launch_task
shift
call "ai-workstreams.bat" claude %*
exit /b %ERRORLEVEL%

:launch_green
call "ai-workstreams.bat" green claude
exit /b %ERRORLEVEL%

:launch_default
where claude >nul 2>&1
if %ERRORLEVEL%==0 (
    claude
) else (
    echo Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code
    echo.
    echo Available commands: claude.bat help
)
goto done

:done
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Failed with exit code %EXITCODE%.
    echo Press any key to close.
    pause >nul
)
exit /b %EXITCODE%
