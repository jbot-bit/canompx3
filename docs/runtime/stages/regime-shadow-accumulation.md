---
task: REGIME shadow accumulation — universe builder + forward shadow runner (Stage 1)
slug: regime-shadow-accumulation
mode: IMPLEMENTATION
status: VERIFIED
created: 2026-06-03
capital_path: true
worktree: C:/Users/joshd/canompx3-regime-shadow (session/joshd-regime-shadow off origin/main 7bf8c40f)
design_doc: docs/plans/2026-06-03-regime-standalone-eligibility-monitor-design.md
---

# REGIME Shadow Accumulation — Stage 1 (universe + runner)

## Mission (operator, 2026-06-03)
Record EVERY REGIME-tier (validated sample_size 30-99) would-have trade
FORWARD (today onward), as forward-monitoring evidence. Live capital stays on
valid DEPLOY lanes ONLY. Shadow != discovery/tuning.

## Decisions locked this session
- Scope = SHADOW-RECORDER ONLY (not the four-gate eligibility predicate).
- Trade source = forward-only from canonical `orb_outcomes` (NOT bar-replay).
- Forward-start boundary = today (2026-06-03); nothing earlier written as shadow.
- Separation sentinel = `paper_trades.execution_source = 'shadow'` (NEW value;
  every existing consumer filters 'live' or 'backfill' — shadow is invisible).

## Operator decisions (RESOLVED 2026-06-03)
1. RECORD-ALL: every active REGIME lane (62: MNQ 25, MES 24, MGC 13) is recorded
   unconditionally. `fitness_status` is an ATTRIBUTE per lane, NEVER an inclusion
   gate. DECAY/STALE lanes are the most informative for a regime turn, so they
   stay in the stream. (Dropped the FIT/WATCH `included` gate and `--include-watch`.)
2. TRIGGER = inside `CanonMPX_DailyRefresh` (single-writer, off-peak) — resolves
   the G1 multi-writer hazard by construction (DailyRefresh owns gold.db
   exclusively). Stage 1 proves `--sync` manually; the refresh-job step is the
   small documented follow-up. Runner ALSO keeps a fail-closed live-session lock
   guard (defense-in-depth) so a manual run during a live session refuses.
3. ORPHAN PRUNING (G5): report-only by default; `--prune-orphans` flag exists,
   defaults OFF (never silent).

## Substrate audit (verified read-only 2026-06-03, this session)
- `paper_trades` table EXISTS (792 rows). Has `execution_source` column with
  values 'backfill'/'live'. NO 'shadow' value yet — clean sentinel.
- `paper_trade_logger.py` is a PROFILE-SCOPED BACKFILL (build_lanes ->
  effective_daily_lanes), NOT a forward REGIME recorder. Reuse its
  filter-application pattern (_query_outcomes / _load_features /
  _inject_cross_asset_atrs / matches_row), NOT its lane universe.
- `paper_trader.py` = full bar-replay engine (ExecutionEngine+RiskManager).
  NOT used — orb_outcomes are pre-computed canonical; replaying re-encodes
  canonical logic (institutional-rigor §4).
- REGIME tier = sample_size 30-99 (RESEARCH_RULES.md:55, ARCHITECTURE.md:112).
- `classify_fitness(rolling_exp_r, rolling_sample, recent_sharpe_30)` canonical.
- Live REGIME universe: 62 active (MNQ 25, MES 24, MGC 13).

## HARD RULES (do not violate)
- No REGIME live allocation. No live profile / live_config changes. No capital.
- gold.db read-only EXCEPT appending paper_trades rows with source='shadow'.
- Forward-only: ValueError if a row earlier than the boundary is about to write.
- Do not weaken account gates. Do not bypass SR / Chordia / c8.
- Delegate to canonical sources; never re-encode fitness/filter/session logic.

## scope_lock
- scripts/tools/regime_shadow_universe.py
- scripts/tools/regime_shadow_runner.py
- tests/test_tools/test_regime_shadow_universe.py
- tests/test_tools/test_regime_shadow_runner.py
- tests/test_tools/test_paper_trades_schema_migration.py
- tests/test_tools/test_regime_shadow_sr_noncontamination.py
- docs/runtime/regime_shadow_universe.yaml
- docs/runtime/stages/regime-shadow-accumulation.md
- trading_app/db_manager.py  # SCOPE EXPANSION (Tier B, user-approved 2026-06-03)
# SCOPE EXPANSION 2 (Tier B, user-approved 2026-06-03) — consumer shadow-guards.
# Adversarial review FALSIFIED the "shadow is structurally invisible" claim: the
# self-review only audited 3 consumers and the per-strategy_id disjointness
# argument does NOT protect AGGREGATE reads. 5 consumers had unfiltered
# COUNT/SUM/GROUP-BY reads that would mix shadow rows into live reports. Guarded
# each with `execution_source != 'shadow'`:
- trading_app/weekly_review.py            # CRIT: §1 forward-perf, §4 discipline, §6 MGC progress
- trading_app/derived_state.py            # HIGH: DB-identity hash churned C11/C12 readiness
- scripts/tools/paper_trade_summary.py    # HIGH: per-lane + portfolio totals
- trading_app/pre_session_check.py        # MED: session-start row count
- trading_app/live/bot_dashboard.py       # MED: live trade list + total count
- tests/test_tools/test_regime_shadow_consumer_invisibility.py  # NEW regression test
# NOTE: consistency_tracker.py:109 was an auditor FALSE POSITIVE — it filters
# `WHERE pnl_dollar IS NOT NULL` and the runner never writes pnl_dollar for shadow
# rows (NULL), so shadow rows are already excluded. Verified, no edit needed.

## SCOPE EXPANSION — paper_trades schema-drift root fix (2026-06-03)
AUDIT FINDING (verified by execution, not memory):
- Live gold.db paper_trades has 3 cols absent from init_trading_app_schema base
  DDL (db_manager.py:361-382): execution_source VARCHAR DEFAULT 'backfill',
  pnl_dollar DOUBLE, notes VARCHAR DEFAULT ''.
- init_trading_app_schema has 27 ALTER TABLE ADD COLUMN IF NOT EXISTS migration
  blocks (validated_setups/daily_features/orb_outcomes/...) — NONE for
  paper_trades. The 3 cols were added to the DB by an uncommitted ad-hoc ALTER.
- verify_trading_app_schema checks paper_trades EXISTENCE only, not columns —
  so the drift is invisible to parity checks.
- IMPACT: a fresh DB rebuild (CanonMPX_DailyRefresh) would break BOTH
  log_trade.py (live trade logging, writes execution_source='live') AND shadow
  accumulation. This is a pre-existing latent capital-path bug.
GROUNDING: CLAUDE.md § Source-of-Truth Chain Rule (audit upstream, never patch
downstream to compensate for upstream corruption) + institutional-rigor §4/§5.
The downstream band-aid (runner self-ALTERs) is FORBIDDEN by these rules.
FIX: add idempotent paper_trades migration to init_trading_app_schema mirroring
the existing pattern; extend verify_trading_app_schema to assert the 3 columns
(close the detection gap — institutional-rigor §6 no-silent-failures).

## Blast Radius
- regime_shadow_universe.py — NEW, read-only; reads validated_setups +
  orb_outcomes (RO), calls classify_fitness/parse_strategy_id. Zero callers.
- regime_shadow_runner.py — NEW; reads orb_outcomes (RO) + daily_features (RO),
  WRITES paper_trades rows ONLY with execution_source='shadow'. Forward-only
  boundary. Reuses ALL_FILTERS[...].matches_row. Zero callers of live paths.
- paper_trades table — WRITE shadow rows only. Existing consumers filter
  'live'/'backfill' (log_trade.py, pre_session_check.py, weekly_review.py) so
  shadow rows are structurally invisible to live/monitoring. SEPARATION PROOF.
- NO edits to lane_allocator.py / account_survival.py / prop_profiles.py /
  strategy_fitness.py / live_config / any profile. Live allocation untouched.

## Adversarial-self-review gaps (closed this session)
- G1 (multi-writer lock): `assert_no_live_session()` reads canonical
  trading_app.live.instance_lock files; fails closed if a live PID holds a
  bot_*.lock before opening the gold.db write connection. Delegates PID-liveness
  to instance_lock.is_pid_alive (no re-encode). Tests: live-lock-held raises +
  writes nothing; dead-PID lock does NOT block.
- G2 (boundary durability): `write_universe_yaml` preserves an already-persisted
  forward_start (never advances it); `resolve_forward_start` re-derives the
  boundary from MIN(shadow trading_day) if the YAML is missing but shadow rows
  exist (never resets to today). Tests: preserve-on-refresh + re-derive-on-delete.
- G3 (automation): runner built + manual `--sync` proven; DailyRefresh step is
  the documented follow-up (decision 2).
- G4 (sr_monitor contamination): defended by tier-disjointness — REGIME (30-99)
  vs CORE deploy (>=100) are disjoint per classify_strategy, so no strategy_id
  carries both shadow and live rows in sr_monitor's un-filtered paper_trades read.
  Test asserts disjointness (live DB) + classifier non-overlap.
- G5 (stale orphans): runner reports orphan shadow strategy_ids each run; prunes
  only under --prune-orphans (default OFF). Tests: report-not-prune + prune-only-
  shadow.
- G6 (gold-plating): the parity guard + schema regression test are scoped to
  making the EXISTING/intended schema verifiable (institutional-rigor §6), not
  new behavior. No four-gate predicate, no live wiring, no sizing.

## Acceptance — EVIDENCE (2026-06-03)
- ROOT FIX idempotency on a COPY of live gold.db: two inits -> 22 cols incl.
  execution_source/pnl_dollar/notes; 792 legacy rows execution_source='backfill';
  second init no-op; verify_trading_app_schema all_valid=True.
- Universe (live DB, read-only): 62 lanes, all included (MNQ 25/MES 24/MGC 13);
  fitness mix FIT 55 / WATCH 6 / STALE 1 (recorded as attribute).
- Runner dry-run (live DB): forward_start=2026-06-03, 62 lanes, 0 would-append
  today (pure boundary effect — today's bars not yet in DB), 0 errored, 0 orphans.
- Tests: 25 passed (14 runner + 5 universe + 4 schema regression + 2 sr-
  noncontamination).
- Generated artifact: docs/runtime/regime_shadow_universe.yaml (immutable
  forward_start=2026-06-03, 62 lanes).

## REMAINING (follow-up, not this stage)
- Wire `--sync` into CanonMPX_DailyRefresh (operator launcher domain; decision 2).
- First real `--sync` accrues rows once 2026-06-03+ bars land in gold.db.
