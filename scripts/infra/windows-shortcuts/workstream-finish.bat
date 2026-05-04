@echo off
title Finish AI Workstream
cd /d "%~dp0\..\..\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode close-pick
pause
