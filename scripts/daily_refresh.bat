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

REM Sync forward-paper accrual from freshly-built orb_outcomes (idempotent;
REM per-lane MAX(trading_day)+1 cursor self-heals any missed days).
echo [%date% %time%] Paper-trade sync >> logs\daily_refresh.log
python -m trading_app.paper_trade_logger --sync 2>&1 >> logs\daily_refresh.log

REM REGIME shadow accrual — record-ALL forward accumulation for sub-100 (REGIME)
REM strategies into paper_trades.execution_source='shadow'. Runs AFTER CORE sync
REM so deploy lanes accrue first; idempotent per-lane MAX(trading_day)+1 cursor.
echo [%date% %time%] REGIME shadow accrual >> logs\daily_refresh.log
python -m scripts.tools.regime_shadow_runner 2>&1 >> logs\daily_refresh.log

REM Log completion
echo [%date% %time%] Daily refresh completed >> logs\daily_refresh.log
