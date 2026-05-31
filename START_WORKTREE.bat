@echo off
:: START_WORKTREE.bat — create/reuse an isolated git worktree and launch Claude
:: there. Hooks cannot auto-ENTER a worktree; this is the thin launcher that can.
::
:: Usage:
::   START_WORKTREE.bat [descriptor]
::   START_WORKTREE.bat --dry-run [descriptor]     (classify only; launch nothing)
::   set CANOMPX3_LAUNCHER_DRYRUN=1 & START_WORKTREE.bat [descriptor]
::
:: Decision (from scripts/tools/worktree_launch_preflight.py):
::   NEW         -> git worktree add -b session/<user>-<desc> off origin/main, launch
::   REUSE_CLEAN -> launch into the existing clean worktree
::   REFUSE_HOT  -> dirty OR a live peer session holds the lease; refuse (exit 3)
::
:: Refusing a hot worktree is the whole point: two Claudes in one worktree
:: corrupt .git/index (parallel-session-isolation.md). Launching triggers
:: session-start.py + the worktree_guard PreToolUse hook normally — guards are
:: NOT bypassed.

setlocal EnableDelayedExpansion
set WT_TITLE=Claude Worktree [%RANDOM%%RANDOM%]
title %WT_TITLE%
cd /d "%~dp0"

:: --- parse args: optional --dry-run, optional descriptor -------------------
set DRYRUN=0
if /I "%CANOMPX3_LAUNCHER_DRYRUN%"=="1" set DRYRUN=1
set DESCRIPTOR=
if /I "%~1"=="--dry-run" (
    set DRYRUN=1
    set DESCRIPTOR=%~2
) else (
    set DESCRIPTOR=%~1
)

:: default descriptor = timestamp (YYYYMMDD-HHMMSS, locale-independent-ish)
if "%DESCRIPTOR%"=="" (
    for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do set DESCRIPTOR=wt-%%c%%a%%b-%%d%%e%%f
    set DESCRIPTOR=!DESCRIPTOR: =0!
)

:: --- classify via the read-only preflight ----------------------------------
set DECISION=
set WTPATH=
set BRANCH=
for /f "usebackq tokens=1,* delims==" %%k in (`python "%~dp0scripts\tools\worktree_launch_preflight.py" --descriptor "!DESCRIPTOR!"`) do (
    if "%%k"=="DECISION" set DECISION=%%l
    if "%%k"=="WTPATH" set WTPATH=%%l
    if "%%k"=="BRANCH" set BRANCH=%%l
)

if "%DECISION%"=="" (
    echo [ERROR] preflight produced no decision — aborting.
    endlocal & exit /b 5
)

echo ============================================
echo   Descriptor: !DESCRIPTOR!
echo   Worktree:   !WTPATH!
echo   Branch:     !BRANCH!
echo   Decision:   !DECISION!
echo ============================================

:: --- dry-run: print decision, do nothing else ------------------------------
if "%DRYRUN%"=="1" (
    echo [DRY-RUN] no worktree created, no Claude launched.
    endlocal & exit /b 0
)

:: --- REFUSE_HOT: do not launch ---------------------------------------------
if "%DECISION%"=="REFUSE_HOT" (
    echo [REFUSED] worktree is dirty or a live Claude session holds its lease.
    echo           Pick another descriptor or resolve the peer session first.
    endlocal & exit /b 3
)

:: --- NEW: create the worktree off origin/main ------------------------------
if "%DECISION%"=="NEW" (
    git fetch origin --quiet
    git worktree add -b "!BRANCH!" "!WTPATH!" origin/main
    if errorlevel 1 (
        echo [ERROR] git worktree add failed.
        endlocal & exit /b 4
    )
)

:: --- launch Claude in the worktree (guards fire normally) ------------------
echo Launching Claude in !WTPATH! ...
start "%WT_TITLE%" /d "!WTPATH!" claude.exe
endlocal & exit /b 0
