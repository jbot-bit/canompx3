@echo off
:: ============================================================================
::  Claude Code REMOTE CONTROL host
::  Lets you drive THIS local machine (gold.db, hooks, worktrees all stay here)
::  from claude.ai/code or the Claude mobile app. Compute is LOCAL, not cloud.
::
::  Root cause this fixes: a Claude Code update kills the running host process
::  and nothing relaunches it -> phone shows "no terminals". This launcher (and
::  the companion scheduled task, see INSTALL_REMOTE_TASK below) keeps a host up.
::
::  --spawn worktree: each incoming phone/web session gets its own isolated git
::  worktree, honouring the one-Claude-per-worktree rule + the .venv/node_modules
::  symlink config in .claude/settings.json.
::
::  To connect: leave this window open, then open claude.ai/code or the mobile
::  app -> "canompx3 - main" appears. Close the window to disconnect.
::  Press space in this window to show a QR code for fast mobile pairing.
:: ============================================================================
title Claude Remote Control [canompx3]
color 0B
cd /d "%~dp0"

echo ============================================
echo   CLAUDE REMOTE CONTROL - HOST
echo ============================================
echo.
echo [Repo] %CD%
echo [Mode] --spawn worktree (isolated worktree per remote session)
echo [Help] space = QR code  ^|  w = toggle spawn mode  ^|  Ctrl+C = stop
echo.
echo Connect from: https://claude.ai/code  or the Claude mobile app
echo.

:: Auto-restart loop: if the host exits (update, network blip, crash), relaunch
:: after a short backoff instead of leaving the phone with no terminal. Ctrl+C
:: twice breaks out cleanly.
:loop
claude remote-control --spawn worktree
echo.
echo [WARN] Remote Control host exited. Restarting in 5s (Ctrl+C to stop)...
timeout /t 5 /nobreak >nul
goto loop
