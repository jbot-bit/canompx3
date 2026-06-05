# Institutional Full Audit + Tasked Plan - 2026-06-03

Verdict: FIX_REQUIRED before new capital expansion; CONTINUE only on the current 3-lane signal/live-readiness path after daily control-state refresh.

Scope: full repo/runtime audit pass from isolated worktree `C:\Users\joshd\canompx3\.worktrees\tasks\codex\institutional-audit-2026-06-03`, branch `wt-codex-institutional-audit-2026-06-03`, HEAD `1e53f3260ba073fdc5e1f58145408db165b8fd20`. Main checkout was dirty, so no production edits were made from main.

Evidence labels:
- MEASURED: directly read from repo, DB, or command output in this audit.
- INFERRED: direct evidence supports the risk, but the exact root cause still needs a focused fix pass.
- UNSUPPORTED: not proven by this audit.

## Executive Finding

The highest-EV bottleneck is not another broad strategy scan. It is operational/data readiness around allocation evidence: 848 active validated setups exist, the current profile deploys 3 lanes, and 845 lanes are intentionally paused. The largest pause reason is not slot capacity; it is missing/failed fresh SR evidence (`UNKNOWN` for 782 paused lanes, `ALARM` for 2). Current live readiness can be made strict-green for the existing 3 lanes after `account_survival` + `refresh_control_state`, but attribution sync and drift are blocked by concurrent DB/process locks, and canonical outcome layers lag the newest 1m bars.

## Source Of Truth Read

Files/rules read for authority:
- `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `HANDOFF.md`
- `TRADING_RULES.md`, `RESEARCH_RULES.md`
- `.claude/rules/institutional-rigor.md`
- `.claude/rules/backtesting-methodology.md`
- `docs/STRATEGY_BLUEPRINT.md`
- `docs/institutional/pre_registered_criteria.md`
- `docs/institutional/mechanism_priors.md`
- `docs/institutional/literature/aronson_2007_ebta_data_snooping.md`
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`
- `docs/runtime/lane_allocation/topstep_50k_mnq_auto.json`

Literature grounding: Aronson/Bailey local extracts support the audit posture that many validated/backtested lanes are not automatically deployable after broad testing; allocation must require fresh SR, holdout discipline, account survival, and telemetry evidence.

## Preflight

MEASURED:
- Main checkout `C:\Users\joshd\canompx3` was dirty before work (`HANDOFF.md`, `docs/audit/results/2026-06-02-mnq-single-leg-account-fit-replacement-v1.csv`).
- Audit used isolated worktree `C:\Users\joshd\canompx3\.worktrees\tasks\codex\institutional-audit-2026-06-03`.
- `scripts/tools/session_preflight.py` reported branch `wt-codex-institutional-audit-2026-06-03`, HEAD `1e53f3260ba073fdc5e1f58145408db165b8fd20`, `.venv=yes`, `.venv-wsl=no`, interpreter `C:\Users\joshd\canompx3\.venv\Scripts\python.exe`.
- `scripts/tools/system_context.py` reported canonical root `C:\Users\joshd\canompx3`, active profile `topstep_50k_mnq_auto`, active instruments `MES`, `MGC`, `MNQ`, and clean audit worktree state.

## Opportunity Map

| rank | lane | claim | evidence source | expected EV | risk | effort | source-of-truth chain | smallest safe diff | action_now? |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Allocation/control-state readiness | 845 active validated lanes are paused; 782 are blocked by missing fresh SR `UNKNOWN`, not slot capacity | `lane_allocation/topstep_50k_mnq_auto.json`; live readiness report | Very high | Capital misallocation if treated as deployable without SR | Medium | `validated_setups` -> allocator JSON -> SR state -> live readiness | Lock-aware attribution/SR refresh, then strict allocator dry-run | YES |
| 2 | DB/process lock + drift drag | Attribution dry-run and direct drift are blocked by concurrent Python/check_drift processes | `paper_trade_logger --sync --dry-run`; process list | High | Stale paper/journal evidence hides execution truth | Medium | process table -> `gold.db` lock -> attribution sync | Add/execute lock-aware operator gate; no force-kill live peer by default | YES |
| 3 | Canonical data freshness | `bars_1m` reaches 2026-06-03, but `bars_5m`/features/outcomes lag | DuckDB read-only query; `audit_integrity.py` | High | Research/deploy claims on newer bars are unsupported | Medium | `bars_1m`, `bars_5m`, `daily_features`, `orb_outcomes` | Scoped rebuild plan after DB lock clears | YES |
| 4 | Research architecture leak | `strategy_discovery.py` imports validation helper from `strategy_validator.py` | phase 9; `trading_app/strategy_discovery.py:1664` | Medium-high | Discovery/validation coupling can weaken anti-leak architecture | Low-medium | discovery -> stats helper -> validator | Move BH helper to neutral stats module with tests | YES |
| 5 | Repo/doc drift | `REPO_MAP.md` stale; phase 3 times out at 30s, direct check fails at 37s | `gen_repo_map.py --check` | Medium | Source-of-truth routing stale | Low | repo files -> generated map | Regenerate map and/or raise/check timeout | YES |
| 6 | Dashboard/live UI readiness | Dashboard control tests pass; live backend gates remain strict | 111 dashboard tests | Medium | UI less likely current blocker | Low | dashboard routes -> backend preflight -> live readiness | No UI rewrite now | NO |
| 7 | ASX cash open | Structural gap remains unproven | no prereg/test run in this audit | Low now | New discovery while ops/data are stale | Medium | prereg -> `pipeline/dst.py` -> outcomes rebuild | Park until readiness/data blockers close | NO |

## Findings

### F1 - Allocation Is Evidence-Blocked, Not Merely Under-Allocated

MEASURED:
- `validated_setups`: 848 active, 23 retired.
- Active by instrument: MNQ 787, MES 48, MGC 13.
- Current profile allocation JSON: rebalance date 2026-05-30, lanes 3, paused 845, stale 0, all_scores_count 848.
- Paused reasons: 782 `strict live gate: SR status UNKNOWN is not CONTINUE for fresh live allocation`; 2 `SR status ALARM`; 35 `Session regime COLD`; 19 live tradeability unsafe filters; remaining single-digit cold variants.
- Current active lanes:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`

Interpretation: active validated count is not deployable count. The next institutional action is to restore/refresh SR + attribution evidence and re-run allocation dry-run, not manually increase live allocations.

### F2 - Existing 3-Lane Profile Can Be Strict-Green After Control Refresh

MEASURED:
- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto` passed Criterion 11: operational survival 95.2%, trailing DD breach 4.8%, daily loss 0.0%, scaling 0.0%.
- First strict readiness run before refresh had Criterion 12 invalid.
- `python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto --force` made Criterion 12 valid with 0 alarms.
- Final `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn` exited 0: deployed 3, validated_active 848, deployed_not_validated 0, strict green true, blockers 0, warnings 1.
- Remaining warning: telemetry not mature, `UNVERIFIED_INSUFFICIENT_TELEMETRY`, 9/30 trading days, 24 files, 119 records, 57 qualifying.

Interpretation: code/config path for the current 3-lane profile is green after refresh, but telemetry is still advisory/immature and should not be described as institutional-grade execution evidence.

### F3 - Attribution Sync Is Blocked By A Real DB Lock

MEASURED:
- `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run` exited 1.
- DuckDB error: worktree `gold.db` cannot open because another process is using it.
- Process list showed `canompx3-stale-work-radar\.venv\Scripts\python.exe pipeline/check_drift.py --skip-crg-advisory --skip-advisory` PIDs 84644 and 91796; earlier lock report also named PID 70196.

Interpretation: attribution/journal freshness cannot be proven until the concurrent drift process releases the DB or the sync tooling gains a lock-aware read-only/retry path. No force-kill was performed.

### F4 - Canonical Data Layers Are Not Equally Fresh

MEASURED:
- DuckDB read-only latest dates:
  - `bars_1m`: 2026-06-03
  - `bars_5m`: 2026-06-01
  - `daily_features`: 2026-06-01
  - `orb_outcomes`: 2026-05-31
- `audit_integrity.py` row counts: `bars_1m` 20,566,951; `bars_5m` 5,300,473; `daily_features` 35,424; `orb_outcomes` 8,949,570.
- Outcome ranges: MES to 2026-05-31, MGC to 2026-05-28, MNQ to 2026-05-31.

Interpretation: anything depending on outcomes after 2026-05-31 is NEED EVIDENCE until scoped rebuild catches up. This is a data freshness blocker, not a reason to run new ASX/open research.

### F5 - Research Discovery Imports Validation Helper

MEASURED:
- `scripts/audits/run_all.py --phase 9` exited 1 with critical architecture finding.
- Direct source: `trading_app/strategy_discovery.py:1664` imports `from trading_app.strategy_validator import benjamini_hochberg`.

Interpretation: the import is a small but real anti-leak architecture smell. The fix should move `benjamini_hochberg` to a neutral statistics module and make both discovery and validation import that module.

### F6 - Repo Map Is Stale

MEASURED:
- `python scripts/tools/gen_repo_map.py --check` exited 1 after about 37s: `REPO_MAP.md is stale. Run: python scripts/tools/gen_repo_map.py`.
- `scripts/audits/run_all.py --phase 3` timed out its repo-map subcheck at 30s.

Interpretation: the map is stale and the audit wrapper timeout is too tight for current repo size. This is doc/source-of-truth drift, not trading logic drift.

### F7 - Some Audit Findings Look Like Stale Heuristics

MEASURED:
- Phase 2 reported `SESSION_CONFIG_DRIFT` against `DYNAMIC_ORB_RESOLVERS`, but `pipeline/dst.py:573-575` builds `DYNAMIC_ORB_RESOLVERS` from `SESSION_CATALOG`.
- Phase 2 reported DOW misalignment for `NYSE_PREOPEN`/`NYSE_OPEN`, but `pipeline/dst.py:232-235` explicitly registers both in `DOW_MISALIGNED_SESSIONS`.
- Phase 4 reported `NYSE_PREOPEN` missing from `ORB_DURATION_MINUTES`; `trading_app/config.py:4216-4229` indeed omits `NYSE_PREOPEN` while other code/tests treat NYSE_PREOPEN as canonical.

Interpretation: DYNAMIC/DOW findings are likely audit false positives and need focused patching in the audit heuristic. `NYSE_PREOPEN` duration omission is a real config parity item unless it is intentionally excluded from the live app config.

## Verification Run

Passed:
- `python scripts/tools/audit_integrity.py` - all 10 checks clean.
- `python scripts/tools/audit_behavioral.py` - all 7 checks clean.
- `python scripts/audits/run_all.py --phase 7` - live trading readiness authority checks passed.
- `python -m pytest tests/test_trading_app/test_bot_dashboard.py tests/test_trading_app/test_bot_dashboard_routes.py tests/test_trading_app/test_bot_dashboard_csrf.py tests/test_trading_app/test_bot_dashboard_holdtokill.py tests/test_trading_app/test_bot_dashboard_signals_recent.py tests/test_trading_app/test_bot_dashboard_sse.py -q` - 111 passed.
- `python -m pytest tests/test_tools/test_live_readiness_report.py tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_sr_monitor.py tests/test_scripts/test_run_live_session_preflight.py -q` - 104 passed.
- `python -m pytest tests/test_tools/test_worktree_guard.py tests/test_tools/test_worktree_launch_preflight.py tests/test_tools/test_worktree_manager.py tests/test_hooks/test_branch_state.py tests/test_hooks/test_branch_flip_guard.py tests/test_hooks/test_mcp_git_guard.py tests/test_hooks/test_head_flip_guard.py tests/test_hooks/test_active_sibling_guard.py -q` - 133 passed.

Failed / blocked:
- `python scripts/audits/run_all.py --quick` - exit 137 around 96s.
- `python scripts/audits/run_all.py --phase 1` - exit 137 around 73s.
- `python scripts/audits/run_all.py --phase 3` - critical repo-map timeout/staleness.
- `python scripts/audits/run_all.py --phase 9` - critical discovery imports validator helper.
- `python pipeline/check_drift.py --fast --quiet` - timed out at 120s; local audit process was stopped, but external stale-work-radar drift processes were left untouched.
- `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run` - blocked by DuckDB file lock.

## Tasked Plan

1. Close DB lock and attribution evidence.
   - Exact next command after the stale-work-radar drift processes finish:
     `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`
   - If it still locks, implement a lock-aware dry-run/report path before any DB write.

2. Refresh canonical data layers before new research or allocation expansion.
   - Confirm scoped rebuild path for `bars_5m`, `daily_features`, and `orb_outcomes`.
   - Do not rank or select candidates on post-2026-05-31 outcomes until this is current.

3. Re-run strict allocation dry-run only after F1/F3/F4 are clean.
   - Command shape:
     `python scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto --strict-live-clean --output docs/runtime/lane_allocation/topstep_50k_mnq_auto.dry-run.json`
   - Expected decision output: selected/rejected reasons, SR state, cold/unsafe filters, account survival implication, and no live allocation write.

4. Fix the research architecture import.
   - Move `benjamini_hochberg` out of `trading_app/strategy_validator.py` into a neutral stats/helper module.
   - Update discovery and validator imports.
   - Verify with phase 9 and targeted strategy discovery/validator tests.

5. Repair repo-map drift.
   - Run `python scripts/tools/gen_repo_map.py`.
   - Re-run `python scripts/tools/gen_repo_map.py --check`.
   - If phase 3 still times out, adjust the phase wrapper timeout or make the check cheaper without hiding staleness.

6. Patch audit false positives and config parity.
   - Add/adjust tests for `DYNAMIC_ORB_RESOLVERS` and `DOW_MISALIGNED_SESSIONS` parity.
   - Decide whether `NYSE_PREOPEN` belongs in `trading_app.config.ORB_DURATION_MINUTES`; if yes, add it with targeted tests. If no, document the boundary and update the audit check.

7. Keep ASX cash open parked.
   - No prereg or scan until current allocation/data/attribution blockers are closed.

## Next Command

`python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`

## Next-Phase Execution Addendum

Verdict: CONTINUE on the current 3-lane profile only; FIX_REQUIRED before capital expansion or new research.

Actions completed in this next phase:
- Converted the ad hoc work capsule into the canonical `pipeline.work_capsule` TOML-front-matter format and added matching stage file. `python scripts/tools/work_capsule.py` now reports `Current work capsule matched`.
- Moved BH/FDR logic to neutral `trading_app/fdr.py`; `strategy_discovery.py` and `strategy_validator.py` now import the shared helper; phase 9 now passes.
- Added `NYSE_PREOPEN` to `trading_app.config.ORB_DURATION_MINUTES`.
- Patched phase 2 false positives for dynamic session resolver parity and DOW misalignment expectations; phase 2 now passes.
- Raised the phase 3 repo-map checker timeout from 30s to 90s and regenerated `REPO_MAP.md`; direct `gen_repo_map.py --check` and phase 3 now pass.
- Ran strict-live-clean allocation dry-run to `docs/runtime/lane_allocation/topstep_50k_mnq_auto.next-phase-dry-run.json`; selected lanes are unchanged.

Fresh live/readiness evidence:
- `account_survival --profile topstep_50k_mnq_auto`: PASS, operational survival 95.2%, trailing DD 4.8%, daily loss 0.0%, scaling 0.0%, 10,000 paths.
- `refresh_control_state --profile topstep_50k_mnq_auto --force`: Criterion 12 valid, 0 alarms, all 3 active lane SR states `CONTINUE`.
- `live_readiness_report --profile topstep_50k_mnq_auto --strict-zero-warn`: exit 0, strict green true, blockers 0, warnings 1; telemetry remains `UNVERIFIED_INSUFFICIENT_TELEMETRY` at 9/30 trading days.
- `rebalance_lanes --profile topstep_50k_mnq_auto --strict-live-clean`: scored 848, deployable 3, paused 845, stale 0, SR liveness `{'UNKNOWN': 845, 'CONTINUE': 3}`, no changes vs current lanes.
- `project_pulse --fast --format json`: broken 0, decaying 4, ready 1, unactioned 22, paused 5. Capsule blocker is closed; remaining decays are MES/MGC/MNQ rebuild staleness plus one stale attribution lane.

Changed files:
- `HANDOFF.md`
- `REPO_MAP.md`
- `docs/audit/results/2026-06-03-institutional-full-audit-tasked-plan.md`
- `docs/runtime/capsules/institutional-audit-2026-06-03.md`
- `docs/runtime/stages/institutional-audit-2026-06-03.md`
- `docs/runtime/lane_allocation/topstep_50k_mnq_auto.next-phase-dry-run.json`
- `scripts/audits/phase_2_infra_config.py`
- `scripts/audits/phase_3_docs.py`
- `tests/test_trading_app/test_strategy_discovery_architecture.py`
- `trading_app/config.py`
- `trading_app/fdr.py`
- `trading_app/strategy_discovery.py`
- `trading_app/strategy_validator.py`

Verification passed:
- `python scripts/tools/work_capsule.py`
- `python scripts/tools/gen_repo_map.py --check`
- `python scripts/audits/run_all.py --phase 0`
- `python scripts/audits/run_all.py --phase 2`
- `python scripts/audits/run_all.py --phase 3`
- `python scripts/audits/run_all.py --phase 4`
- `python scripts/audits/run_all.py --phase 5` (reported findings but wrapper scorecard PASS)
- `python scripts/audits/run_all.py --phase 6` (reported MNQ build-chain finding but wrapper scorecard PASS)
- `python scripts/audits/run_all.py --phase 7`
- `python scripts/audits/run_all.py --phase 8` (reported stale-test findings but wrapper scorecard PASS)
- `python scripts/audits/run_all.py --phase 9`
- `python scripts/audits/run_all.py --phase 10`
- `python -m pytest tests/test_trading_app/test_strategy_discovery_architecture.py tests/test_trading_app/test_strategy_validator.py::TestBenjaminiHochberg -q` - 11 passed
- `python -m py_compile trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py`
- `ruff check trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py tests/test_trading_app/test_strategy_discovery_architecture.py`
- `ruff format --check trading_app/fdr.py trading_app/strategy_discovery.py trading_app/strategy_validator.py trading_app/config.py scripts/audits/phase_2_infra_config.py scripts/audits/phase_3_docs.py tests/test_trading_app/test_strategy_discovery_architecture.py`

Still blocked / not closed:
- Non-dry `paper_trade_logger --sync` is blocked by active `gold.db` writers. Current lock evidence includes PID 69500 running `C:/Users/joshd/canompx3/.worktrees/quick-gate-reliability/scripts/migrations/backfill_validated_trade_windows.py`; earlier active drift/check processes also held the DB. No process was killed.
- `run_all.py --phase 1` still exits 137 on Windows.
- Phase 5/6 findings remain real: MES/MGC/MNQ data freshness/build-chain lag needs a separate scoped rebuild stage. `scripts/tools/run_rebuild_with_sync.sh` exists but includes Pinecone sync and broad rebuild side effects, so it was not run inside this live-readiness capsule.
- Phase 8 findings remain real: stale `E0`/old-session references in tests are measured but outside this capsule's minimal-diff lane.
