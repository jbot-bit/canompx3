@echo off
title Daily Execution Sheet
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m trading_app.prop_portfolio --daily --profile topstep_50k_mnq_auto
echo.
echo ---- TopStep Conditional MGC ----
echo.
python -m trading_app.prop_portfolio --daily --profile topstep_50k
echo.
echo ---- Cross-Account Overview ----
echo.
python -m trading_app.prop_portfolio --daily
pause
