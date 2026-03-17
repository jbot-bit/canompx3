@echo off
title AI Workstreams
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\infra\windows-agent-launch.ps1" -Mode menu
