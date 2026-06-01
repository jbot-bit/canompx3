# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## This Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13.
- **⚠ Truth note (Claude, 2026-05-30, verified):** the `--strict-live-clean` flag and the lane_allocator `feature_cache` opt described above are **NOT committed on any branch** (`git log --all -S` = 0 hits). They live only in `stash@{0}` + `docs/runtime/rescued/2026-05-30-*` (see RESCUE-MANIFEST). Capital-path, Codex-owned — must be committed by its owner, not dropped. The regenerated `lane_allocation*.json` is the OUTPUT of that un-landed code, so canonical `docs/runtime/lane_allocation.json` on HEAD does NOT reflect it.
- **✅ Drift CLEAN — RETRACTION of an earlier false claim (Claude, 2026-05-30, execution-verified):** an earlier version of this note (commit `82721bcc`) claimed `check_active_native_trade_windows_match_provenance` was FAILING on lane `MNQ_COMEX_SETTLE_...OVNRNG_100`. **That was wrong** — I wrote it from a stale memory note without executing, violating Rule 11 (never trust metadata). Direct call returns `VIOLATIONS: 0`; full `check_drift.py --skip-crg-advisory` = **NO DRIFT DETECTED, 170 passed, 0 failed** (incl. Check 191 cold-recheck PASSED). The COMEX_SETTLE lane IS present in canonical `lane_allocation.json` (one of the 3 active lanes). `backfill_validated_trade_windows.py` (live write) = `inspected=848 drifted=0 updated=0`. Trade-window provenance is canonical. No action owed.

## Current Codex Follow-up - ORB Execution Research
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Implemented the bias-hardened ORB execution variants runner requested by the user. New prereg `docs/audit/hypotheses/2026-06-01-orb-execution-variants-v1.yaml` locks K=114 selectable cells over MNQ NYSE_OPEN, US_DATA_1000, and secondary CME_PRECLOSE O5 E2 CB1 RR {1.0,1.5,2.0}. Runner `research/orb_execution_variants_v1.py` uses only `bars_1m`, `daily_features`, and `orb_outcomes`, rejects E2 lookahead predictors, accounts policy EV per original parent opportunity, and emits non-selectable shuffled-date/random-window controls.
- **Result truth:** Full canonical run completed read-only against shared `C:\Users\joshd\canompx3\gold.db`. Outputs: `docs/audit/results/2026-06-01-orb-execution-variants-v1.md` and `docs/audit/results/2026-06-01-orb-execution-variants-v1-cells.csv`. Verdicts: 114 selectable cells = 105 KILL, 9 NARROW, 0 CONTINUE; 18 non-selectable controls = PARK. Primary same-direction re-entry answer is KILL; best same-direction cell (`same_dir_reentry__CME_PRECLOSE__rr2__wait5`) delta +0.0334R but BH-family 0.0525, DSR ~0, era unstable, and 2026 descriptive delta -0.0278R. No priority additions.
- **Bias note:** The shuffled-date control printed large positive deltas, so the report explicitly treats it as construction-sensitivity warning rather than evidence. Opposite-direction fakeout reversal produced NARROW rows but is not the user's same-direction failure mode and is not deployment-ready.
- **Verification:** `python -m py_compile research/orb_execution_variants_v1.py` passed. `python -m pytest tests/test_research/test_orb_execution_variants_v1.py -q` passed 8 tests. Full runner command `python research\orb_execution_variants_v1.py` completed and wrote artifacts.

## Current Codex Follow-up - Best Own ORB Candidate
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** After the user corrected against tunnel vision, added reproducible exploratory runner `research/best_own_strategy_scan_v1.py` and data-first result doc `docs/audit/results/2026-06-01-best-own-strategy-scan-v1.md`. The runner scans MNQ enabled sessions over O{5,15,30}, RR{1,1.5,2}, E2 CB1, and 15 pre-entry-safe filters using only `orb_outcomes` + `daily_features` with the same 2026 holdout discipline. This is explicitly exploratory/post-selection, not a preregistered validation run.
- **Result truth:** Full run wrote 1,755 cell rows to `docs/audit/results/2026-06-01-best-own-strategy-scan-v1-cells.csv` and 2 book rows to `docs/audit/results/2026-06-01-best-own-strategy-scan-v1-portfolio.csv`. Strict full-K exploratory passes = 0; research shortlist = 159. Best practical next hypothesis is **not** same-direction re-entry; it is the MNQ O15/E2/RR2 NYSE_OPEN + US_DATA_1000 book. COST_LT10 book metrics after code-review fix: N=1704, mean=+0.2641R/day, annual=+67.59R, t=5.54, p=3.54e-08, DD=23.42R, 2026 descriptive mean=+0.3707R/day, leg corr=0.218. NO_FILTER book is slightly higher annual (+67.69R) but higher DD (24.42R).
- **Interpretation:** `NARROW`, not deploy. Next formal work should preregister a small book validation using prior survivor context for MNQ NYSE_OPEN and US_DATA_1000, and declare upfront whether annual R (NO_FILTER) or DD/t-stat (COST_LT10) ranks first. Park high in-sample `NYSE_PREOPEN` O30 rows because 2026 descriptive monitoring is negative.
- **Verification:** `python -m py_compile research\best_own_strategy_scan_v1.py` passed. `python research\best_own_strategy_scan_v1.py` completed read-only against shared `C:\Users\joshd\canompx3\gold.db`. Follow-up code review patched COST_LT to delegate to canonical `CostRatioFilter`, patched DSR to use per-cell skew/kurtosis, grounded report in `resources/INDEX.md` + local literature extracts, and added `tests/test_research/test_best_own_strategy_scan_v1.py`.

## Current Codex Follow-up
- **Tool:** Codex
- **Date:** 2026-05-31
- **Summary:** Live-pilot readiness rest pass on `main`: `topstep_50k_mnq_auto` is a 3-lane MNQ single-account pilot (`--copies 1`), NYSE_OPEN SR-alarm lane is paused/parked, funded telemetry maturity remains advisory, strict readiness is green, ProjectX preflight passed 14/14, phase 7 passed, targeted live readiness/preflight tests passed. Also fixed docs/audit drift: CLAUDE sample thresholds, TRADING_RULES `NYSE_PREOPEN`, `.claude/rules/large-file-reads.md` false old-session hit, regenerated `REPO_MAP.md`, and fixed Phase 3's REPO_MAP checker to call `gen_repo_map.py --check`. No live launch was started.
- **Operator launcher update (Codex, 2026-05-31):** `START_BOT.bat` is now the single operator entrypoint and defaults to signal-only control-room startup. The separate `START_LIVE_PILOT.bat` / `scripts/tools/start_topstep_live_pilot.py` path was removed. Dashboard live start pins `topstep_50k_mnq_auto` / `MNQ` / `--copies 1`, runs live-mode preflight with that effective config, and uses the hold-to-confirm UI as the operator gate before `--auto-confirm` is passed to the runner.
- **Dashboard smoke polish (Codex, 2026-06-01):** Rendered dashboard smoke used a mocked localhost API surface to avoid broker auth/live side effects. Fixed first-viewport pilot visibility, broker-account pending copy, mobile topbar wrapping, and stale enabled HOLD TO GO LIVE state after operator blockers load. Live-safe checks remain: `live_readiness_report --copies 1 --strict-zero-warn` green; live preflight is expected to fail while the branch is dirty and should be rerun after commit.
- **Peer-parity EV proof pack (Codex, 2026-06-01):** Implemented on isolated branch `codex/ev-proof-pack-harness`: EV-1 bootstrap health artifact CLI (`scripts/tools/bootstrap_health_proof.py`), EV-2 `live_readiness_report.py` profile proof-pack schema/markdown section, and EV-3 bounded benchmark harness (`scripts/tools/bounded_benchmark_harness.py`). Durable artifacts are under `docs/audit/results/2026-06-01-*`.
- **Runtime truth:** Current worktree preflight is not green: `codex-wsl` expected interpreter `.venv-wsl/bin/python` is missing, the tree is dirty before commit, and fast pulse reports 3 broken items. `live_readiness_report --strict-zero-warn` exits nonzero because current Criterion 11/12 evidence is not green. These are recorded in the committed proof JSONs instead of hidden.
- **Verification:** `python -m pytest tests/test_tools/test_session_preflight.py tests/test_tools/test_project_pulse.py tests/test_tools/test_bootstrap_health_proof.py tests/test_tools/test_bounded_benchmark_harness.py tests/test_tools/test_live_readiness_report.py -q` passed 150 tests. `ruff check ... --quiet` and `git diff --check` passed.
- **CI follow-up:** PR #350 first CI run timed out in `Tests with coverage (tools and research)` while `tests/test_tools/test_fast_lane_status.py` was running. Fixed `fast_lane_status` to extract `scope.strategy_id` and `metadata.template_version` with narrow scalar scans instead of full PyYAML loads; live status build dropped from ~24s to ~0.5s locally. Targeted EV/status tests passed 56 tests.
- **CI follow-up 2:** Second PR #350 CI run completed the tools/research shard but failed `tests/test_tools/test_git_hooks_env.py::test_pre_commit_prefers_wsl_venv_before_windows_venv_on_posix_shells` and ended at the 10-minute shard cap. Fixed pre-commit commit-lock Python selection to prefer `.venv-wsl` on POSIX and raised only the tools/research shard timeout to 15 minutes. `python -m pytest tests/test_tools/test_git_hooks_env.py tests/test_tools/test_fast_lane_status.py -q` passed 24 tests.
- **CI follow-up 3:** Third PR #350 CI run passed tools/research and failed fast-lane drift shard on `test_drift_check_fails_on_unrevoked_pooling_artifact`. Fixed `check_fast_lane_promote_orphans()` to flag any `pooling_artifact` with no revocation sidecar regardless of scanner terminal status. Exact failing test now passes locally.
- **CI follow-up 4:** Fourth PR #350 CI run passed the dedicated fast-lane drift shard, then timed out in `pipeline core` because that shard duplicated `test_check_drift_fast_lane*.py`. Updated CI pipeline-core shard to ignore the fast-lane files already covered by the dedicated shard.
- **Dashboard main-merge follow-up (Codex, 2026-06-01):** Merged `origin/main` into the dashboard live-pilot branch in an isolated worktree, kept the retired standalone live-pilot script/test deleted, and preserved the dashboard as the operator path.

## Last Session
- **Tool:** Unknown
- **Date:** 2026-06-01
- **Commit:** 060fc2d8 — style(live): smoke-polish dashboard pilot UX
- **Files changed:** 3 files
  - `HANDOFF.md`
  - `tests/test_trading_app/test_bot_dashboard.py`
  - `trading_app/live/bot_dashboard.html`

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
