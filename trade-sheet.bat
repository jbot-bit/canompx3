@echo off
title Trade Sheet
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/tools/generate_trade_sheet.py %*
