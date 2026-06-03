# Plan v2 Current Execution

Date: 2026-06-03
Profile: `topstep_50k_mnq_auto`
Checkout: `main` at `7315ccbc`
Scope: current Windows working tree, not clean `origin/main`.

## Disconfirming Checks First

- MEASURED: `gold.db` exists at `C:\Users\joshd\canompx3\gold.db` and opened read-only.
- MEASURED: the worktree is dirty and `main` is ahead 3 / behind 9 versus `origin/main`; this is not clean-main proof.
- MEASURED: `.venv` is the active Windows interpreter; `.venv-wsl` is absent in this checkout.
- MEASURED: `session_preflight.py` warns on dirty tree and 19 active stage files.
- MEASURED: `workflow_doctor.py status --json` initially reported `dirty_tree=BLOCK`, `peer_lease=BLOCK`, `dashboard_stale=WARN`, and `stage_bloat=WARN`.
- MEASURED: later in the run, `workflow_doctor.py db --json` reported `dirty_tree=BLOCK`, `stage_bloat=WARN`, dashboard heartbeat fresh, and DB read-only open OK.
- MEASURED: the peer lease state is volatile/stale-sidecar territory; this run did not release or mutate the lease.

## Live-Readiness Blocker Taxonomy

### `BLOCKED_CRITERION_11`

MEASURED.

Command:
`python -m trading_app.account_survival --profile topstep_50k_mnq_auto`

Result:
- Exit: 2.
- Gate: FAIL at 70%.
- Horizon: 90d, 10,000 paths, source days 2048.
- DD survival / operational pass: 73.3%.
- Daily-loss breach probability: 26.5%.
- Historical daily-loss breach days: 7 (`2022-05-12`, `2025-04-04`, `2025-04-07`, `2025-04-09`, `2025-04-23`, `2026-02-20`, `2026-05-19`).
- Historical max observed 90d DD: `$2,788`.
- Effective strict DD budget: `$1,600`.
- Prop-account path safety: FAIL.
- Final deployability gate: FAIL.

Conclusion:
Do not rank or promote allocation candidates while this blocker is active.

### `CRITERION_12_VALID`

MEASURED.

Command:
`python scripts\tools\live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`

Result:
- Exit: 1 because C11 is blocked.
- `criterion12.latest_verdict=valid`.
- `criterion12.age_days=1`.
- Three active lanes all report SR `CONTINUE`.
- `unsupported_or_missing_evidence=[]`.
- Strict blockers: `Criterion 11 gate not OK`.

### `TELEMETRY_ADVISORY`

MEASURED.

Readiness report:
- Verdict: `UNVERIFIED_INSUFFICIENT_TELEMETRY`.
- Profile-scoped days: 9/30.
- Qualifying records: 57/119.
- Current report surfaces this as advisory for the express/funded profile.

### `PAUSED_UNSPECIFIED`

MEASURED as cleared.

Readiness report:
- Paused lanes: 845.
- Stale lanes: 0.
- Top reasons:
  - 781 strict live gate: SR status UNKNOWN is not CONTINUE.
  - 35 session regime COLD.
  - 12 E2 `PD_CLEAR_LONG` live-tradeability gate.
  - 5 E2 `PD_GO_LONG` live-tradeability gate.
  - 2 session regime COLD.
  - 2 E2 `PD_DISPLACE_LONG` live-tradeability gate.

Direct allocation JSON parse:
- `docs/runtime/lane_allocation/topstep_50k_mnq_auto.json`: `lanes=3`, `paused=845`, `stale=0`, `displaced=0`.
- `docs/runtime/lane_allocation/topstep_50k_mnq_auto.c11-fail-dry-run.json`: `lanes=3`, `paused=845`, `stale=0`, `displaced=0`.

Conclusion:
The prior silence concern is no longer true in the current report surface.

## Allocation Dry-Run Gate

MEASURED decision, not a ranking run.

Because Criterion 11 failed strict account diagnostics, Plan v2's anti-tunnel rule applies:
- Do not rank candidates for promotion.
- Do not convert allocation output into an implicit promotion exercise.
- Treat allocation output as blocker taxonomy only.

## Execution Attribution

MEASURED and not cleared.

Command:
`python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`

Result:
- Exit: 1 on initial run and retry.
- Root cause class: `paper_trade_logger --dry-run` still calls `init_trading_app_schema(...)`, which opens `gold.db` in write mode before dry-run insertion logic.
- Current blocker: active external readers hold `gold.db`.
- Observed lock holders included:
  - PID 9568: `pipeline/check_drift.py`.
  - PID 47988: separate C11/account-survival remediation Python command.
  - PID 91932: transient Python holder reported by DuckDB before inspection.

Command:
`python scripts\tools\project_pulse.py --fast --format json`

Result:
- Exit: 1 because pulse has broken items.
- Counts: `broken=3`, `decaying=4`, `ready=1`, `unactioned=25`, `paused=6`.
- `live_journal_status=unreadable:IOException` because `live_journal.db` is held by PID 81448 / PID 15088: `scripts\run_live_session.py --profile topstep_50k_mnq_auto --signal-only`.
- `missing_count=0`.
- `stale_execution_strategy_ids=['MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15']`.
- Per-lane attribution:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`: paper 75, live 0, last `2026-05-28`, age 6.
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`: paper 53, live 0, last `2026-05-19`, age 15.
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`: paper 73, live 0, last `2026-05-28`, age 6.

Conclusion:
Execution attribution is still stale/blocked. A no-op backfill result was not reproduced in the proper rerun; the supported command currently cannot acquire the write handle.

## Dashboard

MEASURED static smoke:
`python -m pytest tests\test_trading_app\test_bot_dashboard.py -q`

Result:
- Exit: 0.
- 42 passed.
- 3 pytest config warnings for unknown timeout options.

MEASURED runtime GET-only smoke:
- `GET http://127.0.0.1:8080/` returned 200 and title `ORB Bot Dashboard`.
- `GET http://127.0.0.1:8080/api/status` returned 200 with `mode=STOPPED`, `is_running=false`.
- `workflow_doctor.py dashboard --json` reported dashboard heartbeat fresh and planned `SIGNAL` mode for `topstep_50k_mnq_auto`.

SKIPPED:
- No POST live-start, kill, flatten, broker, refresh, or preflight actions were sent.
- No browser visual smoke was run in this pass.

## Phase 7 Live Audit

MEASURED:
`python scripts\audits\run_all.py --phase 7`

Result:
- Exit: 0.
- `PHASE 7 PASSED: 11 checks clean`.

Conclusion:
Live wiring parity is clear on this pass. It does not override the Criterion 11 account-risk block.

## Drift

MEASURED:
`python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory`

Result:
- Exit: 0.
- `SUMMARY: clean passed=137 advisory=15`.

MEASURED profiler gap:
`python scripts\tools\profile_check_drift.py`

Result:
- Timed out after a bounded 90s child-process run.
- The spawned child PID was killed.
- The profiler path emitted a Windows `UnicodeDecodeError` while reading subprocess output under cp1252.
- A subsequent read-only DuckDB open succeeded.

Conclusion:
Fast drift closeout is clean. Full profiling remains a timeout/tooling gap in this Windows run and should not be reported as completed.

## ASX Anti-Bias Guardrails

MEASURED:

Command:
session catalog probe against `pipeline.dst.SESSION_CATALOG`

Result:
- `SESSION_CATALOG` has 13 sessions.
- No ASX / Australia-Sydney / Sydney / cash-open session label was found.

RESEARCH METHOD REVIEW:
- Verdict: UNVERIFIED for any ASX claim.
- No ASX prereg, scan, session-catalog addition, or result claim was created.
- Minimum next evidence: prereg first; declare K; prove ASX cash open is distinct from existing event sessions; ban 2026 selection/ranking; apply costs, BH/FDR, era splits, and holdout discipline.

## Literature Grounding

MEASURED:
`python scripts\tools\check_literature_coverage.py`

Result:
- Resources indexed: 6.
- With curated extract: 4.
- Without curated extract: 2.
- Resource files present locally: 3.
- Resource files missing locally: 3.
- Missing curated extracts: `projectx_api_spec_2026_05_16.md`, `prop-firm-official-rules.md`.

Local methodology anchors used:
- `RESEARCH_RULES.md`.
- `docs/institutional/pre_registered_criteria.md`.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`.
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`.
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`.
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`.
- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`.

## Live-Risk Audit

Scope: `topstep_50k_mnq_auto` live readiness and dashboard runtime surface.

Capital impact: account-risk / deploy-readiness.

Routing note:
- MEASURED: `.claude/agents/live-risk-auditor.md` and `.claude/agents/research-methodologist.md` exist and were applied inline.
- MEASURED: `.claude/agents/canompx3_reviewer.md` is not present in this checkout; `.codex/WORKFLOWS.md` references `canompx3_reviewer`, but no callable/repo agent file was found.

Findings:
- HIGH MEASURED: Criterion 11 strict diagnostics fail.
  - Premise: strict account path safety must not be silently treated as live-ready.
  - Trace: `account_survival` and `live_readiness_report`.
  - Evidence: 7 historical daily-loss breach days; max observed 90d DD `$2,788` versus `$1,600` strict budget.
  - Verdict: BLOCK live promotion/start under strict readiness.
- HIGH MEASURED: execution attribution is not clear.
  - Evidence: paper backfill dry-run is blocked by active DB readers; pulse reports one stale lane; live journal read is blocked by an active signal-only run.
  - Verdict: VERIFY_MORE before treating execution attribution as fresh.
- MEDIUM MEASURED: runtime dashboard is reachable but API status is stopped.
  - Evidence: GET `/api/status` reports `mode=STOPPED`, `is_running=false`.
  - Verdict: no live session was proven running by API smoke.

Skipped checks:
- Broker fills, flatten/kill, and live POST preflight were not exercised because this pass is audit/read-only and C11 is already blocked.
- Mutating paper-trade sync was not run because the dry-run could not acquire DB write access.

Decision: BLOCK.

## Post-Result Sanity Pass

- reality: partial.
- focus: correct.
- fresh_eyes_findings:
  - MEASURED: Plan v2 ran far enough to classify live readiness, dashboard, phase 7, fast drift, literature, and ASX guardrails.
  - MEASURED: It did not clear execution attribution; active DB/live-journal locks block the supported dry-run and pulse live-journal read.
  - MEASURED: C11 is the main blocker; allocation ranking remains the wrong next move while C11 fails.
  - INFERRED: the proper next operational step is lock-aware attribution/C11 remediation in an isolated session, not more strategy discovery.
- bottom_line:
  - Plan v2 executed properly as a blocker taxonomy; live promotion remains blocked.
- next_action:
  - Resolve or wait out active DB/live-journal holders, then rerun `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run` and `python scripts\tools\project_pulse.py --fast --format json` before any live-readiness closeout.

## Final Classification

- `NEED_DB_EVIDENCE`: CLEARED for read-only evidence; BLOCKED for paper-trade write-handle dry-run while active DB readers exist.
- `BLOCKED_CRITERION_11`: ACTIVE.
- `BLOCKED_CRITERION_12`: CLEARED.
- `BLOCKED_TELEMETRY`: ADVISORY under current policy, still 9/30.
- `PAUSED_UNSPECIFIED`: CLEARED as report-surface issue; reasons are explicit.
- `CONFIG_DISCONNECTED`: NOT FOUND in this pass.
- `EXECUTION_ATTRIBUTION`: BLOCKED / STALE. Dry-run cannot acquire DB write access; pulse flags one stale lane and cannot read live journal while signal-only run holds it.
- `DASHBOARD_STATIC`: CLEAR.
- `DASHBOARD_RUNTIME_READONLY`: GET-only clear, API stopped; live actions not exercised.
- `PHASE_7_LIVE_AUDIT`: CLEAR.
- `DRIFT_FAST`: CLEAR.
- `DRIFT_PROFILER`: BLOCKED/TIMEOUT plus Windows decode issue.
- `ASX`: PREREG REQUIRED; no scan or claim.
