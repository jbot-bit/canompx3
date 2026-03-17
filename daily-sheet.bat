@echo off
title Daily Execution Sheet
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m trading_app.prop_portfolio --daily --profile apex_50k_manual
echo.
echo ---- TopStep MGC ----
echo.
python -m trading_app.prop_portfolio --daily --profile topstep_50k
echo.
echo ---- Cross-Account Overview ----
echo.
python -m trading_app.prop_portfolio --daily
pause
