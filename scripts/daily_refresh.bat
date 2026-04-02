@echo off
REM Daily data refresh — download, ingest, build daily_features + O5 outcomes
REM Schedule via: schtasks /create /tn "CanonMPX_DailyRefresh" /tr "C:\Users\joshd\canompx3\scripts\daily_refresh.bat" /sc daily /st 07:30 /rl highest /f
REM Runs at 07:30 Brisbane = after all US sessions close, before 08:00 CME_REOPEN trading

cd /d C:\Users\joshd\canompx3

REM Activate venv
call .venv\Scripts\activate.bat

REM Check if Sunday — full outcome rebuild weekly
for /f %%d in ('powershell -NoProfile -c "(Get-Date).DayOfWeek"') do set DOW=%%d

if "%DOW%"=="Sunday" (
    echo [%date% %time%] Sunday — full rebuild >> logs\daily_refresh.log
    python -m scripts.tools.refresh_data --full-rebuild 2>&1 >> logs\daily_refresh.log
) else (
    python -m scripts.tools.refresh_data 2>&1 >> logs\daily_refresh.log
)

REM Log completion
echo [%date% %time%] Daily refresh completed >> logs\daily_refresh.log
