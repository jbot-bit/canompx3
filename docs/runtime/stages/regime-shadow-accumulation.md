---
slug: regime-shadow-accumulation
mode: IMPLEMENTATION (Stage 0 audit DONE; Stages 1-5 pending fresh context)
status: STAGE0_DONE_AWAITING_CLEAR_THEN_BUILD
created: 2026-06-03
design_doc: docs/plans/2026-06-03-regime-standalone-eligibility-monitor-design.md
capital_path: true
worktree: REQUIRED (isolated, off clean origin/main; START_WORKTREE.bat)
reason_paused: CTX 95% + peers actively committing to main. Build needs fresh context.
---

# REGIME Shadow Accumulation — revive/wire (Guardian mission)

## Mission (operator, 2026-06-03)
Accumulate every REGIME-tier would-have trade from NOW on, forward-only.
Live capital stays on valid DEPLOY lanes ONLY. Shadow = forward-monitoring
evidence, NOT discovery/tuning data.

## HARD RULES (do not violate)
- No REGIME live allocation. No live profile changes. No capital for shadow lanes.
- No OOS tuning. gold.db read-only EXCEPT writing shadow/live journal.
- Shadow results are forward-monitoring evidence, not discovery/tuning.
- Do not weaken account gates. Do not bypass SR / Chordia / c8.

## STAGE 0 AUDIT — DONE (findings, verified read-only 2026-06-03)
VERDICT: substrate EXISTS, runner MISSING. This is revive-and-wire, not build-fresh.

Existing (real):
- `trading_app/paper_trader.py`, `paper_trade_logger.py` — paper exec + logging.
- `paper_trades` DB table — append target; read by sr_monitor.
- `trading_app/sr_monitor.py` — SR drift detect; BOOTSTRAP at lines 171-206:
  paper_trades first, else canonical-forward fallback. Dissolves the N=100
  cold-start paradox — REGIME lanes accrue forward evidence to earn calibration.
- `scripts/tools/monitor_q4_band_shadow.py` — PRECEDENT shadow-monitor to mirror.
- `lane_allocator.py` gate chain (apply_chordia_gate -> apply_c8_gate ->
  apply_live_tradeability_gate) — DEPLOY-only filter, INTACT. Do not touch.

Dead/unverified (confirm next session): monitor_q4_band_shadow run-state
(.pyc only), bot_dashboard_legacy.html.

State writers: trade_journal.py / performance_monitor.py / db_manager.py ->
live_journal; paper_trade_logger.py -> paper_trades. VERIFY append-only
semantics before reuse.

WHY REGIME NOT ACCUMULATING NOW: no job enumerates REGIME-tier FIT/WATCH lanes
and drives them through paper_trader against live canonical bars. Pipe exists;
nothing pushes REGIME through it.

## STAGES 1-5 (build next session, fresh context, isolated worktree)
- S1 Universe: REGIME-tier (validated N 30-99) AND (FIT or WATCH) AND inst in
  {MNQ,MES,MGC} AND not graveyard/no-go AND not failing hard lookahead/account
  rules. Output report: strategy_id, instrument/session, include/exclude reason,
  current gates. READ-ONLY (gold-db MCP + lane scores).
- S2 Shadow ledger: REUSE paper_trader/paper_trades (do NOT build new infra).
  Append-only, mode=SHADOW, source=canonical-live-bars/replay, no broker order
  id. Fields: ts, strategy_id, would_trigger, direction, entry/stop/target,
  theoretical fill, pnl_r, pnl_dollars.
- S3 Monitor gates per shadow lane: rolling_N, rolling ExpR, rolling Sharpe,
  FIT/WATCH/DECAY/STALE (classify_fitness), sr_status (if calibrated),
  recency(60 trading days), drawdown, daily-loss-breach sim, account-survival
  sim. Params LITERATURE-GROUNDED: rolling_N floor 100 (Pepelyshev-Polunchenko
  first-100-trades), recency 60 trading days (P-P ARL~60d). Auto-pause uses
  EXISTING sr_status==ALARM — no new cooldown (avoid re-encoding canonical SR).
- S4 Separation proof: live DEPLOY lanes unchanged; no shadow lane can send
  broker order; no shadow lane in live_config; no allocation writes; no profile
  changes; no account-risk change.
- S5 Verify: targeted tests, py_compile, dry-run one shadow session, prove
  append-only writes only shadow state, prove live preflight sees only DEPLOY.

## scope_lock (build stages MAY touch)
- CREATE scripts/tools/regime_shadow_universe.py (S1, read-only)
- CREATE scripts/tools/regime_shadow_runner.py (S2, reuses paper_trader)
- CREATE tests/test_tools/test_regime_shadow_universe.py
- CREATE tests/test_tools/test_regime_shadow_runner.py
- CREATE docs/runtime/regime_shadow_universe.yaml (generated)
- REUSE (read/append only, NO logic change): paper_trader.py,
  paper_trade_logger.py, sr_monitor.py, classify_fitness.

## FORBIDDEN
- NO edit to lane_allocator.py / account_survival.py (peer-dirty) /
  strategy_fitness.py / prop_profiles.py / live_config / any profile.
- NO SR/Chordia/c8 bypass. NO deployment-threshold change. NO REGIME live alloc.

## Acceptance (each stage: show evidence)
S4 separation proof + S5 append-only + "live preflight sees only DEPLOY" are
the capital-safety gates — must show execution output, not claims.
