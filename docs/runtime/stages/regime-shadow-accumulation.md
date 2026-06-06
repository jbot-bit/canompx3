---
task: REGIME shadow accumulation — Stage 3 DONE (DELETE predicate parity #1 + DailyRefresh wiring #3); Stage 2 R1/F1/F3 prior
slug: regime-shadow-accumulation
mode: IMPLEMENTATION
status: DONE
created: 2026-06-03
reopened: 2026-06-04
stage3_committed: 3fe3bbf6  # pushed origin/session/joshd-regime-shadow 2026-06-04
capital_path: true
worktree: C:/Users/joshd/canompx3-regime-shadow (session/joshd-regime-shadow off origin/main 7bf8c40f)
design_doc: docs/plans/2026-06-03-regime-standalone-eligibility-monitor-design.md
---

# REGIME Shadow Accumulation — Stage 2 (adversarial-audit fix)

## Stage 2 mission (2026-06-04)
Close the 4 findings from the Codex adversarial review (`7bf8c40f..HEAD`,
needs-attention / no-ship). Root-cause analysis (institutional-rigor §3) collapses
4 findings into 3 root-grounded work items — NOT 4 inline patches:

- **R1 (root — kills the F2/F4/whack-a-mole class):** `paper_trades` is a shared
  table discriminated by `execution_source`; invisibility was maintained by every
  reader REMEMBERING to add a predicate (a vigilance contract that leaks by
  construction — 6→7 unguarded readers already proven). Fix structurally: a
  shadow-safe VIEW `live_paper_trades` (`execution_source IN ('live','backfill')`)
  in `init_trading_app_schema` + a STATIC drift check
  `check_paper_trades_reads_are_shadow_safe` that FAILs any `FROM paper_trades`
  read not targeting the VIEW or carrying an `execution_source` predicate. This is
  what makes an 8th leak impossible — CI catches it, not a future reviewer.
  Migration depth = **predicate now, VIEW later** (operator-confirmed 2026-06-04):
  add inline `execution_source` predicates to the 7 unguarded readers so the drift
  check passes; the VIEW is defined for the later cleanup. Subsumes F2 + F4.
- **F1 (separate, active, evidenced ×9 lanes):** one GLOBAL `forward_start` pins
  every lane's monitoring-start to one shared date; a late-joiner inherits the
  global boundary, not its own first-eligible date. Add per-lane `first_seen`;
  boundary = `max(forward_start, lane.first_seen)` threaded into BOTH the insert
  `since=` AND the per-lane DELETE. Back-derive `first_seen=forward_start` for
  existing lanes → provable no-op (0 shadow rows exist today, verified).
- **F3 (separate concurrency concern, D1 operator-confirmed):** N per-lane commits
  → ONE transaction over the whole loop; re-assert `assert_no_live_session()`
  immediately before write-open (narrow the TOCTOU window).

## Honest severity ledger (no inflation — institutional-rigor §12 bias defense)
- **F1 = ACTIVE, evidenced** (9 active strategies in the 90-110 sample band straddle
  REGIME/CORE=100; verified live this session). 0 shadow rows today → fix lands
  before first accrual.
- **F4 = real-in-code, operator-reachability NOT active.** `log_trade` has ZERO
  functional callers / launchers (verified: only a path-string in an improvement
  tool + 2 comments in db_manager). It IS the canonical manual-CLI live-writer
  (db_manager:1008), so NOT dead — covered by R1's VIEW, not deleted, not inflated.
- **F2, F3 = currently SAFE by invariant** (tier-disjointness: smallest active
  profile lane N=427, verified zero active sub-100 lanes / DailyRefresh
  single-writer). Fixes are defense-in-depth hardening, NOT active bug-stops.
- **R1 justification is STRUCTURAL** ("end the leak class"), not "stop an active
  leak". Honest and sufficient — a vigilance contract on a capital-adjacent table
  WILL leak again.

## Reader inventory (verified this session — 13 `FROM paper_trades` readers)
GUARDED (6, drift check passes as-is): weekly_review ×5, pre_session_check ×2,
bot_dashboard ×2, derived_state, paper_trade_summary ×4, consistency_tracker ×3
(implicit: `pnl_dollar IS NOT NULL`; shadow rows have NULL pnl_dollar).
UNGUARDED (7, get inline predicate under R1): paper_trade_logger (MAX/summary),
log_trade (stat), sr_monitor, sprt_monitor, prop_portfolio, project_pulse.
DANGER NOTE: paper_trade_logger's DELETE (:260-267) must NEVER gain a predicate
that would delete shadow rows — its DELETE is keyed on strategy_id (CORE lanes);
shadow rows are disjoint and must stay untouched.

---

# REGIME Shadow Accumulation — Stage 1 (universe + runner) [SHIPPED 7d9889a4]

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

## Stage 2 — VERIFICATION EVIDENCE (2026-06-04)
- Tests: 248 passed (Stage-2 + all touched-consumer regression suites) + 126
  passed (adjacent: bot_dashboard/lane_allocator/paper_trader/schema-migration).
  Zero failures. The 6 consumer-suite regressions were UNREALISTIC fixtures
  (hand-rolled paper_trades without execution_source); fixed by adding the column
  the real schema always has — NOT by weakening the production predicate.
- Drift: `NO DRIFT DETECTED: 176 checks passed, 0 skipped, 22 advisory`. New
  Check 196 (`paper_trades reads exclude shadow rows`) PASSED.
- Drift known-violation injection: an injected unguarded `FROM paper_trades`
  SELECT is CAUGHT; removed → passes (mutation-proof, integrity-guardian §7).
- R1 VIEW: idempotent, no-op on live-DB copy (792==792, 0 shadow rows), excludes
  a seeded shadow+live collision (sum drops by exactly the shadow pnl_r).
- F1 no-op proof (live DB): all 62 lanes resolve per-lane boundary == forward_start
  == 2026-06-03 (`{'2026-06-03': 62}`); dry-run 62 scanned / 0 errored / 0 append.
- ruff: all changed files clean.
- Dead-code sweep: every new symbol referenced; `PAPER_TRADES_SHADOW_SOURCE` was
  initially dead → wired (runner's `SHADOW_SOURCE` now delegates to it, removing a
  re-encoded literal, institutional-rigor §4).

## Adversarial-audit gate (institutional-rigor §2 / adversarial-audit-gate.md)
Independent evidence-auditor pass on the fix diff: **CONDITIONAL**. Claims 1,2,5,6,7
PROVEN from code; claim 3 (the 2026-06-03 date) proven live this session; claim 4
CONDITIONALLY PROVEN with one residual gap below.

### RESOLVED in Stage 3 (`3fe3bbf6`) — was the deferred finding below
**Fix shipped:** both DELETE paths in `paper_trade_logger.py` now carry
`AND execution_source != 'shadow'` (predicate parity with the read paths at
`_query_outcomes` MAX() and the post-sync summary). Shadow rows survive a
shared-strategy_id resync structurally — chosen over the disjointness drift
check below because it is the root fix (R1 philosophy) and a strictly smaller
diff that does not leave the unsafe DELETE in place. Regression test:
`test_paper_trade_logger_delete_preserves_shadow_rows` (both DELETE paths +
load-bearing unguarded assertion). Drift NO DRIFT (176), 11/11 shadow tests.

**Optional remaining defense-in-depth (NOT required — the root fix above closes
the exposure):** a `check_shadow_universe_disjoint_from_core_lanes` drift check
asserting shadow-YAML strategy_ids ∩ prop_profiles CORE lanes = ∅ would catch a
REGIME→CORE promotion-without-YAML-removal at CI time. Belt-and-suspenders only;
the DELETE is already shadow-safe regardless of disjointness.

### (historical) DEFERRED finding — superseded by the Stage-3 fix above
**Gap:** `paper_trade_logger.py` DELETE (~:260-267) keys on `strategy_id` ALONE
(no execution_source predicate — correctly, it owns CORE rows). IF a strategy were
promoted REGIME→CORE while STILL present in the shadow universe YAML, a
`paper_trade_logger` full-resync would DELETE that strategy's shadow rows
(shared strategy_id). **Not an active bug:** tier-disjointness holds TODAY
(verified live: smallest active profile lane N=427; zero active sub-100 lanes; 0
shadow rows exist), and the universe build-time tripwire (`regime_shadow_universe.py`
classify_strategy != REGIME → ValueError) blocks a CORE lane ENTERING the shadow
universe. The uncovered case is a lane that was REGIME when added and is promoted
later without YAML removal — a promotion-workflow race, not a code defect.
**Deferral justification:** closing it requires a NEW drift/runtime guard asserting
shadow-YAML strategy_ids ∩ prop_profiles CORE lanes = ∅ — a separate, clean,
adversarial-audit-gated follow-up. It is NOT in Stage-2 scope (the 4 review
findings), would expand the diff into prop_profiles read paths, and the race
cannot fire while disjointness holds. **Follow-up:** add
`check_shadow_universe_disjoint_from_core_lanes` to check_drift.py (next stage).

## Commit note — pre-commit hook BYPASSED (--no-verify), justified
The pre-commit hook BLOCKED on `TestCollectWorktrees::test_detects_worktree` — a
PRE-EXISTING flaky test that asserts on the LIVE machine worktree count
(`32 open worktrees` from concurrent sessions). It PASSES in isolation; the
Stage-2 diff does NOT touch it (verified `git diff --cached` — no worktree-test
lines). The hook's own gates were satisfied MANUALLY this session: drift NO DRIFT
(176 checks, Check 196 PASSED), 248+126 tests pass, ruff clean. Committed with
`--no-verify` (operator-approved 2026-06-04) — branch-only, no push to main, on
the correct branch (branch-flip guard moot). Follow-up (separate, trivial-tier):
make the worktree-count test environment-independent (mock the enumeration).

## scope_lock
- scripts/tools/regime_shadow_universe.py        # F1: first_seen per-lane boundary
- scripts/tools/regime_shadow_runner.py          # F1 boundary thread + F3 atomicity
- tests/test_tools/test_regime_shadow_universe.py
- tests/test_tools/test_regime_shadow_runner.py
- tests/test_tools/test_paper_trades_schema_migration.py
- tests/test_tools/test_regime_shadow_sr_noncontamination.py
- tests/test_tools/test_regime_shadow_consumer_invisibility.py  # R1 VIEW + collision tests
- docs/runtime/regime_shadow_universe.yaml
- docs/runtime/stages/regime-shadow-accumulation.md
- trading_app/db_manager.py  # R1: VIEW live_paper_trades (Tier B, user-approved)
# STAGE 2 scope (R1 structural fix — predicate-now/VIEW-later, operator-confirmed 2026-06-04):
- pipeline/check_drift.py                         # R1: check_paper_trades_reads_are_shadow_safe
- tests/test_pipeline/test_check_drift_paper_trades_shadow_safe.py  # R1 drift known-violation injection
- trading_app/paper_trade_logger.py              # R1: execution_source predicate on MAX/summary reads
- trading_app/log_trade.py                       # R1: execution_source predicate on post-trade stat
- trading_app/sr_monitor.py                      # R1: execution_source predicate (DiD)
- trading_app/sprt_monitor.py                    # R1: execution_source predicate (DiD)
- trading_app/prop_portfolio.py                  # R1: execution_source predicate (DiD)
- scripts/tools/project_pulse.py                 # R1: execution_source predicate (DiD)
- scripts/reports/monitor_lane_correlation_rolling.py  # R1: 8th reader (NOT in plan inventory — drift check surfaced it)
- pipeline/db_contracts.py                        # R1: canonical VIEW + live-source constants
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

## REMAINING (follow-up)
- ~~Wire the shadow runner into CanonMPX_DailyRefresh~~ DONE Stage 3 (`3fe3bbf6`):
  `daily_refresh.bat` now runs `python -m scripts.tools.regime_shadow_runner`
  after CORE `paper_trade_logger --sync`. NOTE: the runner has NO `--sync` flag —
  it accumulates by default (`sync_shadow(...)` unconditional); the earlier
  "`--sync`" wording was wrong.
- First real accrual happens on the next scheduled refresh once 2026-06-03+ bars
  land in gold.db (shadow rows currently 0 — built, now wired, not yet fired).
- (Optional) disjointness drift check — see "Optional remaining defense-in-depth"
  above. Not required; the DELETE is shadow-safe at the root.
