@echo off
:: Per-launch unique title suffix: prevents FindWindow collision in the
:: auto-minimise step (line ~104) when a stale or sibling START_BOT
:: console shares the title. %RANDOM% is 0-32767 per cmd instance;
:: doubled gives ~1 in 1.07B collision odds — sufficient for single-user
:: same-machine scope.
set BOT_TITLE=ORB Trading Bot [%RANDOM%%RANDOM%]
title %BOT_TITLE%
color 0A
cd /d "%~dp0"

echo ============================================
echo   ORB TRADING BOT - STARTING UP
echo ============================================
echo.

:: Keep this banner static. Dynamic Git probing in this launcher can break
:: double-click startup on Windows; use docs/reference/start-bot-checkout-model.md
:: for the branch/merge model instead.
echo [Repo] %CD%
echo [Repo] This shortcut runs the Windows checkout above.
echo [Repo] WSL/Codex branch pushes do not change this app until merged or pulled here.
echo.

:: Market-closed guard — canonical truth, NOT the local weekday.
:: The machine runs Brisbane time (UTC+10), so it is "Saturday" locally during
:: the entire Friday US session when the CME market is actually OPEN. A naive
:: (Get-Date).DayOfWeek check warned "markets closed" mid-session. Delegate to
:: pipeline.market_calendar.is_market_open_at (the same source the orchestrator
:: uses) so the warning fires ONLY when the market is genuinely closed.
::   prints OPEN / CLOSED; fail-open (prints OPEN) on any error so the guard
::   never blocks a launch it cannot evaluate.
for /f "tokens=1" %%s in ('.venv\Scripts\python.exe -c "from datetime import datetime; from zoneinfo import ZoneInfo; from pipeline.market_calendar import is_market_open_at; print('OPEN' if is_market_open_at(datetime.now(ZoneInfo('UTC'))) else 'CLOSED')" 2^>nul') do set MKT=%%s
if "%MKT%"=="CLOSED" (
    echo [WARNING] CME market is currently CLOSED ^(weekend gap / holiday^).
    echo Press any key to start anyway, or Ctrl+C to cancel.
    pause
)

:: Active profile — change ACTIVE_PROFILE to deploy a different account profile.
:: See trading_app/prop_profiles.py ACCOUNT_PROFILES for the registry.
set ACTIVE_PROFILE=topstep_50k_mnq_auto

:: Mode: SIGNAL by default (no broker orders). START_BOT is the control-room
:: entrypoint. The dashboard START LIVE button still does the gated demo/live
:: handoff after preflight — that path is unchanged.
::
:: Optional arg lets the operator launch a mode directly from this front-end
:: launcher (their preferred surface — never the bare CLI):
::   START_BOT.bat            -> signal-only (default, no capital)
::   START_BOT.bat demo       -> demo (paper broker)
::   START_BOT.bat live       -> LIVE (REAL MONEY). run_live_session still runs
::                               the full 14-gate preflight + CONFIRM before any
::                               order; --live arms capital.
:: Startup is no-menu (operator 2026-06-11): a bare double-click just boots the
:: bot in SIGNAL + brings up the dashboard. MODE (signal/demo/LIVE) AND the broker
:: ACCOUNT (Express vs Combine) are chosen in the dashboard UI — the LIVE button +
:: account selector are the sole interactive arming path. This removes the old
:: 1/2/3 menu AND the .bat-direct-LIVE path that bypassed the account selector
:: (live-launch blocker #13). An explicit mode ARG is still honored for
:: scripting/shortcuts (e.g. `START_BOT.bat live`), but is NOT the normal flow.
set BOT_MODE_FLAGS=
if /i "%~1"=="signal" set BOT_MODE_FLAGS=--signal-only
if /i "%~1"=="demo"   set BOT_MODE_FLAGS=--demo
if /i "%~1"=="live"   set BOT_MODE_FLAGS=--live
if /i "%~1"=="--live" set BOT_MODE_FLAGS=--live
:: Bare double-click (no arg) → SIGNAL + dashboard. LIVE is armed from the UI.
if not defined BOT_MODE_FLAGS set BOT_MODE_FLAGS=--signal-only

:: Step 1: Clean up stale lock + stop files (don't kill python — other terminals may be running)
::   The .stop file triggers graceful-shutdown on the next feed scan; a stale one
::   from a previous abort will kill a fresh orchestrator within ~60s.
echo [1/5] Removing stale lock + stop files...
del /f /q "%TEMP%\canompx3\bot_*.lock" >nul 2>&1
del /f /q "live_session.stop" >nul 2>&1

:: Step 2: Clear stale bot state so dashboard shows clean STOPPED until orchestrator writes fresh data
echo [2/5] Clearing stale state...
del /f /q "data\bot_state.json" >nul 2>&1
.venv\Scripts\python.exe scripts\tools\sweep_orphan_rings.py >nul 2>&1

:: Step 2b: Publish the canonical planned-launch surface so the dashboard can
:: render the unambiguous "Next launch: <MODE> · <profile> · <instruments> ·
:: N broker accounts" banner BEFORE the orchestrator boots. The orchestrator's
:: own state (bot_state.json) supersedes this once running. Schema/mode is
:: validated by trading_app/live/planned_launch.py — bad inputs refuse to write.
if /i "%BOT_MODE_FLAGS%"=="--signal-only" set PLANNED_MODE=SIGNAL
if /i "%BOT_MODE_FLAGS%"=="--demo" set PLANNED_MODE=DEMO
if /i "%BOT_MODE_FLAGS%"=="--live" set PLANNED_MODE=LIVE
if not defined PLANNED_MODE set PLANNED_MODE=SIGNAL
.venv\Scripts\python.exe -m trading_app.live.planned_launch write --profile %ACTIVE_PROFILE% --mode %PLANNED_MODE% --source START_BOT.bat --copies 1 --instrument MNQ >nul

:: Step 2c: Safety-state preflight (read-only). Tells the operator if a
:: persisted kill switch will halt today's session BEFORE launch — so a halt is
:: never a silent 0-trade mystery. A STALE (prior-day) kill switch auto-expires
:: in the orchestrator and is purely informational here; a SAME-DAY kill switch
:: prints a clear warning + the path to clear it. Fail-open: never blocks launch.
echo [2c/5] Safety-state preflight...
.venv\Scripts\python.exe -m trading_app.live.session_safety_state --profile %ACTIVE_PROFILE% --instrument MNQ

:: Step 3: Data freshness check
echo [3/5] Checking data freshness...
.venv\Scripts\python.exe -c "from pipeline.paths import GOLD_DB_PATH; import duckdb; con=duckdb.connect(str(GOLD_DB_PATH),read_only=True); r=con.execute('SELECT MAX(trading_day) FROM daily_features WHERE orb_minutes=5').fetchone(); print(f'  Latest daily_features: {r[0]}'); con.close()" 2>nul
echo.

:: Step 3b: For a direct LIVE launch ONLY, run the SAME blocking gate the
:: dashboard START LIVE button runs before arming real money: refresh control
:: state, then a strict-zero-warn preflight. This closes the direct-launch
:: bypass — without it, START_BOT.bat live skipped the dashboard's mode=live
:: arm guard (bot_dashboard.action_start), which blocks a real-money launch on
:: ANY advisory (WARN/SKIPPED) check. --strict-zero-warn ports that exact rule
:: (via the canonical run_live_session preflight) to this path. Routing flags
:: mirror _live_pilot_cli_args (MNQ, 1 copy, account 21944866 = EXPRESS default).
:: Signal/demo launches skip this block (no live orders -> advisory checks stay advisory).
if /i "%BOT_MODE_FLAGS%"=="--live" (
    echo [3b/5] Refreshing LIVE control state: profile=%ACTIVE_PROFILE%
    .venv\Scripts\python.exe -m scripts.tools.refresh_control_state --profile %ACTIVE_PROFILE%
    if errorlevel 1 (
        echo.
        echo [BLOCKED] LIVE control-state refresh failed. Fix failures above before real-money launch.
        pause
        exit /b 1
    )
    echo [3b/5] Running LIVE strict preflight: profile=%ACTIVE_PROFILE%
    .venv\Scripts\python.exe -m scripts.run_live_session --profile %ACTIVE_PROFILE% --instrument MNQ --copies 1 --account-id 21944866 --live --preflight --strict-zero-warn
    if errorlevel 1 (
        echo.
        echo [BLOCKED] LIVE preflight failed or has advisory WARN/SKIPPED checks. Fix before real-money launch.
        pause
        exit /b 1
    )
    echo.
)

:: Step 3c: For a DEMO launch, auto-heal stale C11/C12 control state. The demo
:: orchestrator's own preflight (trading_app/live/preflight.py) treats a C11/C12
:: db-identity mismatch as a HARD FAIL with no demo downgrade-to-WARN, so a stale
:: cache (e.g. after a merge or a daily_features rebuild changes the gold.db
:: identity hash) blocks the demo entirely. This mirrors the LIVE block at [3b/5]
:: but is FAIL-OPEN: a refresh failure warns and continues (demo routes paper
:: orders only — the orchestrator's preflight remains the real gate). The refresh
:: is selective: without --force, refresh_control_state only recomputes the
:: criterion that is actually invalid/mismatched, so a clean demo is a near no-op
:: and pays no MC-sim cost. SIGNAL launches skip this block (C11/C12 are SKIPPED
:: for signal-only).
if /i "%BOT_MODE_FLAGS%"=="--demo" (
    echo [3c/5] Refreshing DEMO control state: profile=%ACTIVE_PROFILE%
    .venv\Scripts\python.exe -m scripts.tools.refresh_control_state --profile %ACTIVE_PROFILE%
    if errorlevel 1 (
        echo.
        echo [WARN] DEMO control-state refresh reported invalid C11/C12 after refresh.
        echo        Continuing — demo routes paper orders only; the orchestrator preflight is the real gate.
    )
    echo.
)

:: Step 4: Launch orchestrator (trading engine) in a separate console window.
:: This is the process that connects to the broker, watches bars, evaluates entries,
:: and writes bot_state.json. Without it the dashboard reads stale state.
echo [4/5] Launching orchestrator: profile=%ACTIVE_PROFILE% %BOT_MODE_FLAGS%
:: Suppress the orchestrator's own dashboard spawn — this bat opens the dashboard
:: directly on line 78. Without this env var, two dashboards + two browser tabs
:: open and race for live_journal.db (fails preflight check 6).
::
:: Window state: signal/demo launch minimized (background). LIVE launches VISIBLE
:: so the operator can type CONFIRM at the real-money prompt — a minimized window
:: would hang on input() and never arm. The typed CONFIRM is the human gate.
set WIN_STATE=/min
if /i "%BOT_MODE_FLAGS%"=="--live" set WIN_STATE=
:: Live orchestrator MUST carry the same broker-routing flags the [3b/5] preflight
:: and the dashboard's _live_pilot_cli_args pass. Without --account-id the engine
:: re-runs its own boot preflight, sees 2 broker accounts, and FAILS check [13]
:: ("no --account-id would default to accounts[0]") -> never arms. Mirror line 137.
:: Account = 21944866 (EXPRESS) — topstep_50k_mnq_auto's C11 survival proof is
:: validated on this tier (is_express_funded). INVARIANT (audit Finding 2): this
:: explicit-arg `START_BOT.bat live` escape hatch ALWAYS routes Express and does
:: NOT consult the dashboard selection — it is for scripted/recovery launches only.
:: The NORMAL flow leaves this menu in SIGNAL and arms LIVE from the dashboard
:: account selector (#acct-select / clickable cards -> selectedAccountId), which
:: threads the chosen account through the SAME _live_pilot_cli_args builder so the
:: dashboard's preflight and launch bind one id ([13] parity). If you ever want
:: the .bat escape hatch to trade Combine, change this id deliberately.
set LIVE_ROUTING=
if /i "%BOT_MODE_FLAGS%"=="--live" set LIVE_ROUTING=--instrument MNQ --copies 1 --account-id 21944866
start "ORB Orchestrator (%ACTIVE_PROFILE%)" %WIN_STATE% cmd /k "set CANOMPX3_DASHBOARD_ORIGIN=1 && .venv\Scripts\python.exe -m scripts.run_live_session --profile %ACTIVE_PROFILE% %BOT_MODE_FLAGS% %LIVE_ROUTING%"

:: Step 5: Launch dashboard + open browser in this (main) console.
:: This console auto-minimises ~3s after launch (line ~107). To stop the
:: dashboard: restore the console from the taskbar, then press Ctrl+C, or
:: close the window. Orchestrator window stays /min until closed separately
:: (or use the dashboard's kill endpoint).
echo [5/5] Launching dashboard...
echo.
echo ============================================
echo   Profile:   %ACTIVE_PROFILE% %BOT_MODE_FLAGS%
echo   Dashboard: http://localhost:8080
echo   TopStepX:  https://app.topstepx.com
echo   To stop:   restore this window from taskbar, then Ctrl+C
echo ============================================
echo.

:: Open browser after 2 second delay (gives server time to start)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080"

:: Minimise THIS console ~3s after launch so the browser is the visible surface.
:: Dashboard process keeps running here; restoring the console + Ctrl+C still
:: stops it (see Step 5 contract above). FindWindow-by-title targets %BOT_TITLE%
:: (set with unique per-launch suffix on line 2) to avoid collision with stale
:: or sibling START_BOT consoles. Fail-open: any PowerShell error leaves the
:: console as-is.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; try { Add-Type -Name W -Namespace U -MemberDefinition '[System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern System.IntPtr FindWindow(string c, string w); [System.Runtime.InteropServices.DllImport(\"user32.dll\")] public static extern bool ShowWindowAsync(System.IntPtr h, int n);'; $h=[U.W]::FindWindow($null,'%BOT_TITLE%'); if ($h -ne [System.IntPtr]::Zero) { [void][U.W]::ShowWindowAsync($h,6) } } catch {}"

.venv\Scripts\python.exe -m trading_app.live.bot_dashboard
