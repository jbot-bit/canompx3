@echo off
title Active AI Workstreams
cd /d "%~dp0\..\..\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode list
pause
