@echo off
setlocal enabledelayedexpansion
title M2.5 Audit Launcher
cd /d "%~dp0"

:menu
cls
echo.
echo  =====================================================================
echo   M2.5 SECOND-OPINION AUDIT LAUNCHER
echo  =====================================================================
echo   Every audit auto-injects: architecture context (250 lines),
echo   imported modules, DB schema, config values, test files, git history.
echo   Presets use --triage to auto-filter the ~41%% false positive noise.
echo  =====================================================================
echo.
echo   QUICK AUDIT (pick a file, pick a lens)
echo   -----------------------------------------------
echo    [1]  bugs          Type/None/timezone/resource bugs
echo    [2]  bias          Data snooping, look-ahead, survivorship
echo    [3]  joins         daily_features triple-join, row inflation
echo    [4]  architect     Design patterns, coupling, extensibility
echo    [5]  discovery     Dead code, unreachable branches
echo    [6]  improvements  Head-of-quant institutional suggestions
echo    [7]  general       Structured good + findings + recs
echo    [8]  deep          3-turn reasoning (understand/hypothesise/verify)
echo.
echo   DEEP DIVE (audit entire subsystems)
echo   -----------------------------------------------
echo    [A]  Pipeline core       ingest + 5m bars + daily_features + DST
echo    [B]  Trading engine      outcome_builder + discovery + validator
echo    [C]  ML system           meta_label + training + inference
echo    [D]  Live trading        data_feed + orchestrator + order router
echo    [E]  Guardrails          check_drift + health_check + audit tools
echo    [F]  Cost + config       cost_model + config + asset_configs
echo    [G]  Research scripts    research/*.py (bias/methodology scan)
echo    [H]  Full stack          Pipeline + Trading + ML (the works)
echo.
echo   TAILORED PRESETS (one-click, enriched, auto-triaged)
echo   -----------------------------------------------
echo   [P1]  Outcome builder integrity     (pnl_r, exits, look-ahead)
echo   [P2]  Validator gate system          (BH FDR, samples, WF)
echo   [P3]  Pipeline data flow             (ingest ^> 5m ^> features)
echo   [P4]  Cost model accuracy            (ticks, spread, slippage)
echo   [P5]  Live trading safety            (reconnect, dupes, stops)
echo   [P6]  ML meta-label quality          (leakage, gates, features)
echo   [P7]  DST / timezone correctness     (session windows, DOW)
echo   [P8]  Config drift detection         (fail-open, hardcoded vals)
echo.
echo   DIFF REVIEW (review only changed lines — M2.5's sweet spot)
echo   -----------------------------------------------
echo   [D1]  Review uncommitted changes     (diff vs HEAD)
echo   [D2]  Review changes since ref       (diff vs main, HEAD~3, etc.)
echo.
echo   OTHER
echo   -----------------------------------------------
echo   [S]   Auto-scan changed files (since last commit)
echo   [T]   Auto-scan staged files only
echo   [V]   ML full integration audit
echo   [W]   Feature planning (4T orient/design/detail/validate)
echo   [X]   Custom prompt on any file(s)
echo   [B$]  Show API call budget
echo.
echo   [Q]   Quit
echo.
set /p "choice=  >>> "

if /i "%choice%"=="q" exit /b
if /i "%choice%"=="b$" (
    python scripts/tools/m25_audit.py --budget
    pause
    goto menu
)

REM --- Quick audit modes ---
if "%choice%"=="1" ( set "MODE=bugs" & goto ask_file )
if "%choice%"=="2" ( set "MODE=bias" & goto ask_file )
if "%choice%"=="3" ( set "MODE=joins" & goto ask_file )
if "%choice%"=="4" ( set "MODE=architect" & goto ask_file )
if "%choice%"=="5" ( set "MODE=discovery" & goto ask_file )
if "%choice%"=="6" ( set "MODE=improvements" & goto ask_file )
if "%choice%"=="7" ( set "MODE=general" & goto ask_file )
if "%choice%"=="8" goto single_deep

REM --- Deep dives ---
if /i "%choice%"=="a" goto dive_pipeline
if /i "%choice%"=="b" goto dive_trading
if /i "%choice%"=="c" goto dive_ml
if /i "%choice%"=="d" goto dive_live
if /i "%choice%"=="e" goto dive_guardrails
if /i "%choice%"=="f" goto dive_cost_config
if /i "%choice%"=="g" goto dive_research
if /i "%choice%"=="h" goto dive_full

REM --- Presets ---
if /i "%choice%"=="p1" goto preset_outcome
if /i "%choice%"=="p2" goto preset_validator
if /i "%choice%"=="p3" goto preset_pipeline
if /i "%choice%"=="p4" goto preset_cost
if /i "%choice%"=="p5" goto preset_live
if /i "%choice%"=="p6" goto preset_ml
if /i "%choice%"=="p7" goto preset_dst
if /i "%choice%"=="p8" goto preset_config

REM --- Diff review ---
if /i "%choice%"=="d1" goto diff_head
if /i "%choice%"=="d2" goto diff_ref

REM --- Other ---
if /i "%choice%"=="s" goto auto_audit
if /i "%choice%"=="t" goto auto_staged
if /i "%choice%"=="v" goto ml_audit
if /i "%choice%"=="w" goto plan_mode
if /i "%choice%"=="x" goto custom_prompt

echo  Invalid choice.
timeout /t 2 >nul
goto menu

REM ============================================================
REM  FILE PICKER (shared by quick audit modes)
REM ============================================================

:ask_file
echo.
echo  Popular targets:
echo   [1] trading_app/outcome_builder.py
echo   [2] trading_app/strategy_validator.py
echo   [3] trading_app/strategy_discovery.py
echo   [4] pipeline/build_daily_features.py
echo   [5] pipeline/build_bars_5m.py
echo   [6] pipeline/ingest_dbn.py
echo   [7] pipeline/cost_model.py
echo   [8] pipeline/dst.py
echo   [9] trading_app/ml/meta_label.py
echo  [10] trading_app/live/session_orchestrator.py
echo  [11] trading_app/live/data_feed.py
echo  [12] trading_app/config.py
echo  [13] pipeline/check_drift.py
echo  [14] pipeline/health_check.py
echo   ... or type any relative path
echo.
set /p "FPICK=  File (number or path): "
if "%FPICK%"=="" goto menu

set "FILE="
if "%FPICK%"=="1"  set "FILE=trading_app/outcome_builder.py"
if "%FPICK%"=="2"  set "FILE=trading_app/strategy_validator.py"
if "%FPICK%"=="3"  set "FILE=trading_app/strategy_discovery.py"
if "%FPICK%"=="4"  set "FILE=pipeline/build_daily_features.py"
if "%FPICK%"=="5"  set "FILE=pipeline/build_bars_5m.py"
if "%FPICK%"=="6"  set "FILE=pipeline/ingest_dbn.py"
if "%FPICK%"=="7"  set "FILE=pipeline/cost_model.py"
if "%FPICK%"=="8"  set "FILE=pipeline/dst.py"
if "%FPICK%"=="9"  set "FILE=trading_app/ml/meta_label.py"
if "%FPICK%"=="10" set "FILE=trading_app/live/session_orchestrator.py"
if "%FPICK%"=="11" set "FILE=trading_app/live/data_feed.py"
if "%FPICK%"=="12" set "FILE=trading_app/config.py"
if "%FPICK%"=="13" set "FILE=pipeline/check_drift.py"
if "%FPICK%"=="14" set "FILE=pipeline/health_check.py"
if not defined FILE set "FILE=%FPICK%"

echo.
echo  Running: m25_audit.py %FILE% --mode %MODE% --enrich --triage
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py %FILE% --mode %MODE% --enrich --triage
set "FILE="
goto done

:single_deep
echo.
echo  Deep review — 3-turn reasoning + enrichment + triage.
echo.
set "FILE="
echo  Popular targets:
echo   [1] trading_app/outcome_builder.py
echo   [2] trading_app/strategy_validator.py
echo   [3] pipeline/build_daily_features.py
echo   [4] trading_app/ml/meta_label.py
echo   [5] trading_app/live/session_orchestrator.py
echo   ... or type any path
echo.
set /p "FPICK=  File: "
if "%FPICK%"=="" goto menu

set "FILE="
if "%FPICK%"=="1" set "FILE=trading_app/outcome_builder.py"
if "%FPICK%"=="2" set "FILE=trading_app/strategy_validator.py"
if "%FPICK%"=="3" set "FILE=pipeline/build_daily_features.py"
if "%FPICK%"=="4" set "FILE=trading_app/ml/meta_label.py"
if "%FPICK%"=="5" set "FILE=trading_app/live/session_orchestrator.py"
if not defined FILE set "FILE=%FPICK%"

set "DMODE="
set /p "DMODE=  Lens [bugs/bias/joins/general/discovery/architect/improvements] (default general): "
if "%DMODE%"=="" set "DMODE=general"

echo.
echo  Running: m25_audit.py %FILE% --deep --enrich --triage --mode %DMODE%
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py %FILE% --deep --enrich --triage --mode %DMODE%
set "FILE="
goto done

REM ============================================================
REM  DIFF REVIEW — M2.5's sweet spot
REM ============================================================

:diff_head
echo.
echo  DIFF REVIEW: Uncommitted changes vs HEAD
echo  M2.5 is strongest reviewing small focused changes.
echo  -----------------------------------------------
echo.
echo  File(s) to review (space-separated, or blank for all changed):
set /p "DFILES=  >>> "
if "%DFILES%"=="" (
    REM Get list of changed files
    for /f "tokens=*" %%f in ('git diff --name-only HEAD -- "*.py" 2^>nul') do (
        set "DFILES=!DFILES! %%f"
    )
)
if "%DFILES%"=="" (
    echo  No changed .py files found.
    goto done
)
echo  Files: %DFILES%
echo.
set "DMODE="
set /p "DMODE=  Lens [bugs/bias/joins/general] (default bugs): "
if "%DMODE%"=="" set "DMODE=bugs"
echo.
echo  Running: m25_audit.py %DFILES% --diff HEAD --enrich --triage --mode %DMODE%
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py %DFILES% --diff HEAD --enrich --triage --mode %DMODE%
goto done

:diff_ref
echo.
echo  DIFF REVIEW: Changes since a git ref
echo  -----------------------------------------------
echo.
set /p "REF=  Git ref (e.g. HEAD~3, main, abc1234): "
if "%REF%"=="" goto menu
echo.
echo  File(s) to review (space-separated, or blank for all changed):
set /p "DFILES=  >>> "
if "%DFILES%"=="" (
    for /f "tokens=*" %%f in ('git diff --name-only %REF% -- "*.py" 2^>nul') do (
        set "DFILES=!DFILES! %%f"
    )
)
if "%DFILES%"=="" (
    echo  No changed .py files found vs %REF%.
    goto done
)
echo  Files: %DFILES%
echo.
set "DMODE="
set /p "DMODE=  Lens [bugs/bias/joins/general] (default bugs): "
if "%DMODE%"=="" set "DMODE=bugs"
echo.
echo  Running: m25_audit.py %DFILES% --diff %REF% --enrich --triage --mode %DMODE%
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py %DFILES% --diff %REF% --enrich --triage --mode %DMODE%
goto done

REM ============================================================
REM  DEEP DIVES (all use --deep --enrich --triage)
REM ============================================================

:dive_pipeline
echo.
echo  DEEP DIVE: Pipeline Core
echo  Files: ingest_dbn, build_bars_5m, build_daily_features, dst
echo  -----------------------------------------------
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/joins/general] (default joins): "
if "%DLENS%"=="" set "DLENS=joins"
echo.
python scripts/tools/m25_audit.py pipeline/ingest_dbn.py pipeline/build_bars_5m.py pipeline/build_daily_features.py pipeline/dst.py --deep --enrich --triage --mode %DLENS%
goto done

:dive_trading
echo.
echo  DEEP DIVE: Trading Engine
echo  Files: outcome_builder, strategy_discovery, strategy_validator
echo  -----------------------------------------------
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/bias/general] (default bias): "
if "%DLENS%"=="" set "DLENS=bias"
echo.
python scripts/tools/m25_audit.py trading_app/outcome_builder.py trading_app/strategy_discovery.py trading_app/strategy_validator.py --deep --enrich --triage --mode %DLENS%
goto done

:dive_ml
echo.
echo  DEEP DIVE: ML System
echo  -----------------------------------------------
echo.
set "ML_FILES="
for %%f in (trading_app\ml\*.py) do (
    if /i not "%%~nxf"=="__init__.py" (
        set "ML_FILES=!ML_FILES! trading_app/ml/%%~nxf"
        echo  Found: trading_app/ml/%%~nxf
    )
)
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/bias/general] (default bias): "
if "%DLENS%"=="" set "DLENS=bias"
echo.
python scripts/tools/m25_audit.py %ML_FILES% --deep --enrich --triage --mode %DLENS%
goto done

:dive_live
echo.
echo  DEEP DIVE: Live Trading
echo  -----------------------------------------------
echo.
set "LIVE_FILES="
for %%f in (trading_app\live\*.py) do (
    if /i not "%%~nxf"=="__init__.py" (
        set "LIVE_FILES=!LIVE_FILES! trading_app/live/%%~nxf"
        echo  Found: trading_app/live/%%~nxf
    )
)
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/general/architect] (default bugs): "
if "%DLENS%"=="" set "DLENS=bugs"
echo.
python scripts/tools/m25_audit.py %LIVE_FILES% --deep --enrich --triage --mode %DLENS%
goto done

:dive_guardrails
echo.
echo  DEEP DIVE: Guardrails
echo  Files: check_drift, health_check, audit_behavioral
echo  -----------------------------------------------
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/discovery/general] (default bugs): "
if "%DLENS%"=="" set "DLENS=bugs"
echo.
python scripts/tools/m25_audit.py pipeline/check_drift.py pipeline/health_check.py scripts/tools/audit_behavioral.py --deep --enrich --triage --mode %DLENS%
goto done

:dive_cost_config
echo.
echo  DEEP DIVE: Cost + Config
echo  Files: cost_model, config, asset_configs
echo  -----------------------------------------------
echo.
set "DLENS="
set /p "DLENS=  Lens [bugs/general/discovery] (default general): "
if "%DLENS%"=="" set "DLENS=general"
echo.
python scripts/tools/m25_audit.py pipeline/cost_model.py trading_app/config.py pipeline/asset_configs.py --deep --enrich --triage --mode %DLENS%
goto done

:dive_research
echo.
echo  DEEP DIVE: Research Scripts (bias scan)
echo  -----------------------------------------------
echo.
set "RES_FILES="
set "RES_COUNT=0"
for %%f in (research\research_*.py) do (
    set "RES_FILES=!RES_FILES! research/%%~nxf"
    set /a "RES_COUNT+=1"
    echo  Found: research/%%~nxf
)
echo.
echo  !RES_COUNT! files found. Large sets may hit token limits.
echo.
set /p "CONT=  Continue? (y/n): "
if /i not "%CONT%"=="y" goto menu
echo.
python scripts/tools/m25_audit.py %RES_FILES% --enrich --triage --mode bias
goto done

:dive_full
echo.
echo  DEEP DIVE: Full Stack (3 sequential audits)
echo  Pipeline + Trading + ML
echo  -----------------------------------------------
echo.
set /p "CONT=  This runs 3 audits (~9 API calls). Continue? (y/n): "
if /i not "%CONT%"=="y" goto menu

echo.
echo  [1/3] Pipeline core (joins lens)...
echo  -----------------------------------------------
python scripts/tools/m25_audit.py pipeline/ingest_dbn.py pipeline/build_bars_5m.py pipeline/build_daily_features.py --deep --enrich --triage --mode joins
echo.
echo  [2/3] Trading engine (bias lens)...
echo  -----------------------------------------------
python scripts/tools/m25_audit.py trading_app/outcome_builder.py trading_app/strategy_discovery.py trading_app/strategy_validator.py --deep --enrich --triage --mode bias
echo.
echo  [3/3] ML system (bias lens)...
echo  -----------------------------------------------
set "ML_FILES="
for %%f in (trading_app\ml\*.py) do (
    if /i not "%%~nxf"=="__init__.py" set "ML_FILES=!ML_FILES! trading_app/ml/%%~nxf"
)
python scripts/tools/m25_audit.py %ML_FILES% --deep --enrich --triage --mode bias
goto done

REM ============================================================
REM  AUTO SCANS
REM ============================================================

:auto_audit
echo.
echo  Scanning all changed files since last commit...
echo  -----------------------------------------------
echo.
python scripts/tools/m25_auto_audit.py
goto done

:auto_staged
echo.
echo  Scanning staged files only...
echo  -----------------------------------------------
echo.
python scripts/tools/m25_auto_audit.py --staged
goto done

:ml_audit
echo.
echo  Running full ML integration audit...
echo  -----------------------------------------------
echo.
python scripts/tools/m25_ml_audit.py
goto done

REM ============================================================
REM  PLANNING (4T — enriched)
REM ============================================================

:plan_mode
echo.
echo  4T Feature Planning (orient / design / detail / validate)
echo  Enriched with DB schema, config, tests, git history.
echo  -----------------------------------------------
echo.
echo  Describe the feature:
set /p "PLAN_DESC=  >>> "
if "%PLAN_DESC%"=="" goto menu
echo.
echo  Relevant files? (space-separated, or blank for auto-detect)
echo  Example: trading_app/entry_rules.py pipeline/build_daily_features.py
set /p "PLAN_FILES=  >>> "
echo.
if "%PLAN_FILES%"=="" (
    echo  Running: m25_audit.py --plan "..." --enrich
    echo  -----------------------------------------------
    python scripts/tools/m25_audit.py --plan "%PLAN_DESC%" --enrich
) else (
    echo  Running: m25_audit.py %PLAN_FILES% --plan "..." --enrich
    echo  -----------------------------------------------
    python scripts/tools/m25_audit.py %PLAN_FILES% --plan "%PLAN_DESC%" --enrich
)
goto done

REM ============================================================
REM  CUSTOM PROMPT (enriched + triaged)
REM ============================================================

:custom_prompt
echo.
echo  Custom Audit — your prompt, any file(s), enriched + triaged.
echo  -----------------------------------------------
echo.
echo  File(s) — space-separated:
set /p "FILE=  >>> "
if "%FILE%"=="" goto menu
echo.
echo  Your audit prompt (one line):
set /p "PROMPT=  >>> "
if "%PROMPT%"=="" goto menu
echo.
echo  Running: m25_audit.py %FILE% --prompt "..." --enrich --triage
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py %FILE% --prompt "%PROMPT%" --enrich --triage
goto done

REM ============================================================
REM  TAILORED PRESETS (enriched + triaged)
REM ============================================================

:preset_outcome
echo.
echo  PRESET: Outcome Builder Integrity (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py trading_app/outcome_builder.py --enrich --triage --prompt "You are a senior quant developer at a systematic futures fund auditing the ORB breakout outcome pre-computation engine. This file generates ~6.1M rows of trade outcomes across 4 micro-futures instruments (MGC, MNQ, MES, M2K), 3 ORB apertures (5/15/30 min), and entry models E1 (market-after-confirm) and E2 (stop-market at ORB boundary). You have been given the DB schema, current config, test files, and git history as runtime context since you cannot query these systems yourself. Audit with the rigour of a Bloomberg quant review: (1) pnl_r computation correctness — entry price, stop distance, target, R-multiple normalization. Is risk always positive? Can division-by-zero occur? (2) C8 session-close exit and C3 ORB-opposite-break exit — are these applied consistently across ALL entry models and apertures? Any path where an exit is skipped? (3) DELETE-then-INSERT idempotency — could concurrent access cause data loss? Is the transaction boundary correct? (4) Look-ahead bias — could ANY outcome be computed using information not available at trade entry time? Check every column used. (5) The orb_minutes JOIN with daily_features — must join on (trading_day, symbol, orb_minutes). Missing orb_minutes = 3x row inflation. (6) Cost deduction — is cost_r from cost_model.py applied to every outcome? Any path where costs are skipped? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING with specific line references."
goto done

:preset_validator
echo.
echo  PRESET: Strategy Validator Gates (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py trading_app/strategy_validator.py --enrich --triage --prompt "You are a senior quant developer at a systematic futures fund auditing the multi-phase strategy validation system. This validator is the statistical gatekeeper — it decides which of ~2,376 grid-searched strategy combinations survive to production. You have been given the DB schema, current config, test files, and git history as runtime context. Audit: (1) BH FDR (Benjamini-Hochberg False Discovery Rate) — is it applied correctly? Check: sorted p-values ascending, threshold = (rank/total)*q, step-up procedure from largest rank down. Any off-by-one? Is q=0.05? (2) Gate completeness — can ANY strategy reach validated_setups without passing ALL gates (FDR, min samples, year consistency, optional WF)? Trace every code path. (3) --no-regime-waivers flag — this enables min-years-positive-pct check (lives in an else branch). Is it correctly gated? (4) Walk-forward splits — are IS/OOS windows truly non-overlapping? Any data from OOS window leaking into IS? (5) REGIME classification (30-99 trades) — does it correctly prevent standalone portfolio inclusion? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING with specific line references."
goto done

:preset_pipeline
echo.
echo  PRESET: Pipeline Data Flow (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py pipeline/ingest_dbn.py pipeline/build_bars_5m.py pipeline/build_daily_features.py --enrich --triage --prompt "You are a senior quant developer auditing the 3-stage data pipeline for a futures ORB breakout system. You have runtime context (DB schema, config, tests, git history) since you cannot query these yourself. Stage 1: ingest_dbn.py ingests Databento .dbn.zst files into bars_1m. Stage 2: build_bars_5m.py aggregates 1m to 5m. Stage 3: build_daily_features.py computes ORB ranges, RSI, ATR. Invariants: All timestamps UTC. Trading day = 09:00 Brisbane to next 09:00. daily_features has 3 rows per (trading_day, symbol). Audit: (a) Trading day assignment — can any bar land on the wrong day? (b) 5m aggregation — deterministic across re-runs? Gap handling? (c) ORB computation — correct time window per session? DST handling? (d) Feature look-ahead — any features from data AFTER ORB close? (e) DELETE-then-INSERT date range correctness? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING with specific line references."
goto done

:preset_cost
echo.
echo  PRESET: Cost Model Accuracy (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py pipeline/cost_model.py --enrich --triage --prompt "You are a senior quant at a systematic futures fund auditing the transaction cost model. You have runtime context showing the actual COST_SPECS values. Audit: (1) Tick sizes correct for CME micros? MGC=0.10, MNQ=0.25, MES=0.25, M2K=0.10. (2) Spread realistic for micro futures during our sessions? (3) Slippage realistic for stop-market entries (E2)? (4) Commission per-side or round-trip — applied correctly? (5) cost_r = total_cost / (stop_distance * point_value) — can stop_distance be zero? (6) Costs applied to EVERY outcome path? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING with specific line references."
goto done

:preset_live
echo.
echo  PRESET: Live Trading Safety (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py trading_app/live/session_orchestrator.py trading_app/live/data_feed.py --enrich --triage --prompt "You are a senior systems engineer auditing live trading infrastructure. REAL MONEY at risk. Audit for PRODUCTION SAFETY: (1) WebSocket disconnect mid-session — reconnection? Can it miss a signal? (2) Duplicate orders — same signal fire twice? Dedup check? (3) Stop-loss guaranteed before/with entry? What if stop placement fails? (4) Crash recovery — state persisted? Knows about open position on restart? (5) State cleanup on session end/error? Stale state leaking to next session? (6) All time comparisons in UTC? Brisbane conversion correct? Flag CRITICAL (could-lose-money) issues first. For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING."
goto done

:preset_ml
echo.
echo  PRESET: ML Meta-Label Quality (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py trading_app/ml/meta_label.py --enrich --triage --prompt "You are a senior ML engineer auditing a Random Forest meta-labeling classifier for ORB breakout futures. V4: 12 per-aperture models. Predicts P(profitable) — TAKE or SKIP. Audit for ML INTEGRITY: (1) Temporal train/test split — truly walk-forward? No shuffling? No random split? (2) Feature look-ahead — trace each feature to data source. Any from after entry time? (3) 4-gate system (delta_r >= 0, CPCV AUC >= 0.50, test AUC > 0.52, skip <= 85%%) — all enforced? (4) -999.0 NaN sentinel — safely out-of-band for ALL features? (5) Target leakage — any feature encodes outcome by construction? (6) Expanding vs fixed window retraining? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING."
goto done

:preset_dst
echo.
echo  PRESET: DST / Timezone Correctness (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py pipeline/dst.py --enrich --triage --prompt "You are auditing DST and timezone handling. Brisbane = UTC+10 (no DST). US = EST/EDT. SESSION_CATALOG resolves per-day. You have the actual session times in runtime context. Audit: (1) US DST dates correct 2019-2026? 2nd Sunday March, 1st Sunday Nov. (2) Session times shift correctly when US changes? (3) NYSE_OPEN midnight crossing — trading_day assignment correct? (4) DOW alignment guard — Brisbane DOW vs exchange DOW for midnight sessions? (5) Edge cases — DST transition day, sessions near midnight, Feb 29? (6) Any remnant hardcoded times bypassing SESSION_CATALOG? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING."
goto done

:preset_config
echo.
echo  PRESET: Config Drift Detection (enriched + triaged)
echo  -----------------------------------------------
echo.
python scripts/tools/m25_audit.py trading_app/config.py pipeline/check_drift.py --enrich --triage --prompt "You are auditing the config and drift detection system. config.py = source of truth. check_drift.py = runtime guardian. You have actual config values in context. Audit: (1) Magic numbers traceable to research (@research-source)? (2) Any drift check that could pass when it should fail (fail-open)? Look for broad except, return True on error. (3) Check count dynamic (len(CHECKS)) not hardcoded? (4) Canonical source coverage — checks verify imports not hardcodes? (5) Metadata trust — any check verifying labels/comments instead of behavior? (6) Completeness gaps — what should be checked but isn't? For each finding, mark as TRUE, FALSE POSITIVE, or WORTH EXPLORING."
goto done

REM ============================================================

:done
echo.
echo  =====================================================================
echo   Audit complete.
echo  =====================================================================
echo.
set /p "again=  Run another? [y/n]: "
if /i "%again%"=="y" goto menu
