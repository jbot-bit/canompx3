# HANDOFF.md - Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done - update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Claude Session — C11 Secure & Hold (2026-06-04)
- **Tool:** Claude Code
- **Verdict:** `topstep_50k_mnq_auto` C11 is a **measured capital NO-GO. Live NOT armed.**
- **What landed (`572499d6`, rebased onto peer `5dab3861`; pushed in `b0d43c0f`):** committed the stranded C11 analysis (4 audit-result docs + corrected `c11-80pt-cap-wiring.md` stage + `.gitignore` for the dry-run scratch json). Docs/config only; no production logic, no schema, no capital lever moved.
- **The decisive math:** the ~80pt ORB cap clears C11's operational-survival (≥70%) and zero-breach-day gates, but **NOT** the drawdown-magnitude gate — strict observed 90d DD at the current 0.75 stop = **$2,038**, exceeding both the $1,600 strict budget AND the full $2,000 Topstep MLL. A *wider* stop (1.0) makes DD worse; only a **tighter** stop (≤0.50) closes the gap (cap+0.50 → $1,142). Cap is necessary, not sufficient.
- **Path to GO (research, not "go live ASAP"):** pre-registered cap+stop≤0.50 remediation (RULE 10) → re-run `account_survival` proving BOTH C11 gates → DSR/era checks → **independent live-path audit** (incl. the bracket-risk-parity fix `9b3fc530`, whose adversarial-audit gate is still OPEN — no independent reviewer yet) → only then live arming (separate operator GO).
- **Already safe:** parity plumbing `9b3fc530` and preserve branch `c11-orb-cap-preserve-2026-06-04` (`48071955`) both on origin.
- **Deferred (not done this session):** REGIME design docs (owned by `canompx3-regime-shadow` worktree — committed from there, not here); cap value selection; the cap+stop remediation research.

## This Session
- **Tool:** Codex
- **Date:** 2026-06-03
- **Summary:** Completed the bounded MNQ single-leg account-fit replacement audit for `topstep_50k_mnq_auto`. Restored the prereg after the prior prereg-only commit was reverted, added `research/mnq_single_leg_account_fit_replacement_v1.py`, focused tests, and result artifacts under `docs/audit/results/2026-06-02-mnq-single-leg-account-fit-replacement-v1.*`. Scope stayed locked at five candidates x three incumbent replacement slots (K=15), using structured allocator fields, canonical `orb_outcomes` + `daily_features`, canonical filter delegation, and read-only DB access.
- **Result truth:** All 15 replacement scenarios are `KILL`. Best ranked scenario was `R02_C05`, replacing `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` with `MNQ_CME_PRECLOSE_O15_E2_RR2_COST_LT10`; it improved annual dollars/maxDD but failed account-safety gates due daily-loss breach evidence. No scenario is live-valid, validated, deployable, or an allocation change from this artifact.
- **Verification:** YAML parse passed; `prereg_front_door.py --format text` passed; `python -m py_compile research\mnq_single_leg_account_fit_replacement_v1.py` passed; focused pytest passed 8 tests; runner execution wrote 15 scenarios; scoped ruff passed; `git diff --check` passed; `project_pulse.py --fast` reported broken=0; `pipeline\check_drift.py --fast --quiet` passed with advisories only.

## Current Codex Follow-up - Workflow Reliability And Stage Ownership
- **Tool:** Codex
- **Date:** 2026-06-03
- **Summary:** Actioned the workflow-control-plane cleanup as a reliability job, not a stage-count cleanup. Verified the main-worktree lease holder PID/PPID were absent and `peer_live=false`, then released the stale lease. Archived only the approved Batch A stage `2026-05-29-drift-cache-proof-of-honesty.md`; no contested/live/drift/capital/unverifiable stages were moved. Added durable phased plan `docs/plans/active/2026-06/2026-06-03-workflow-reliability-stage-ownership.md` grounding the remaining work in worktree/branch/DB/port/dashboard/drift/lease ownership.
- **Current blockers:** main remains dirty until this workflow-tooling diff is committed; dashboard port 8080 is open from `C:\Users\joshd\canompx3-live-launch-tokyo` with stale heartbeat; `gold.db` read-only probe is OK but hardlink count is 2; drift/precommit work remains owned by dirty peer `codex/precommit-drift-speed`.
- **Verification:** `python -m pytest tests\test_tools\test_workflow_doctor.py tests\test_tools\test_stage_reaper_audit.py -q` passed 34 tests; scoped `ruff check` passed; `git diff --check` passed; `workflow_doctor.py status` showed no live peer lease block but did show dirty tree/dashboard stale/stage bloat; `stage_reaper_audit.py` reported `DONE_SAFE=0 LIVE_OR_CONTESTED=24 UNVERIFIABLE=8 CLOSED=23 total=55`.

## Previous Session
- **Tool:** Codex
- **Date:** 2026-05-30
- **Summary:** Cleared the `topstep_50k_mnq_auto` live-validity blocker in both this Codex worktree and canonical `C:\Users\joshd\canompx3`. Root cause was strict live allocation using SR state for the current book as if it covered all candidates, allowing `UNKNOWN` SR candidates and old SR-alarm lanes to rotate back in. `rebalance_lanes.py --strict-live-clean` now requires current SR `CONTINUE` evidence, computes correlation only after hard gates, and the allocator caches feature rows so rebalance stays bounded. Canonical allocation is now 3 lanes: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`, `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`. Canonical C11/C12 refreshed green, `live_readiness_report --strict-zero-warn` green with only telemetry maturity advisory, and canonical signal-only preflight passed 13/13.
- **⚠ Truth note (Claude, 2026-05-30, verified):** the `--strict-live-clean` flag and the lane_allocator `feature_cache` opt described above are **NOT committed on any branch** (`git log --all -S` = 0 hits). They live only in `stash@{0}` + `docs/runtime/rescued/2026-05-30-*` (see RESCUE-MANIFEST). Capital-path, Codex-owned — must be committed by its owner, not dropped. The regenerated `lane_allocation*.json` is the OUTPUT of that un-landed code, so canonical `docs/runtime/lane_allocation.json` on HEAD does NOT reflect it.
- **✅ Drift CLEAN — RETRACTION of an earlier false claim (Claude, 2026-05-30, execution-verified):** an earlier version of this note (commit `82721bcc`) claimed `check_active_native_trade_windows_match_provenance` was FAILING on lane `MNQ_COMEX_SETTLE_...OVNRNG_100`. **That was wrong** — I wrote it from a stale memory note without executing, violating Rule 11 (never trust metadata). Direct call returns `VIOLATIONS: 0`; full `check_drift.py --skip-crg-advisory` = **NO DRIFT DETECTED, 170 passed, 0 failed** (incl. Check 191 cold-recheck PASSED). The COMEX_SETTLE lane IS present in canonical `lane_allocation.json` (one of the 3 active lanes). `backfill_validated_trade_windows.py` (live write) = `inspected=848 drifted=0 updated=0`. Trade-window provenance is canonical. No action owed.

## Current Codex Follow-up - Rescue Cleanup
- **Tool:** Codex
- **Date:** 2026-06-03
- **Summary:** Continued rescue review from clean branch `review/rescue-2026-06-03` after `origin/main` advanced to `f505b69a`. Landed only connected low-risk documentation: `docs/plans/active/2026-05/2026-05-23-slack-control-room-design.md`, an observer-only Slack control-room plan with active-plan lifecycle frontmatter. Excluded stale rescue `HANDOFF.md` changes; no Slack install/posting, no live trading, no DB writes, and no allocator/profile/broker/journal/webhook changes.
- **Remaining:** Rescue refs remain undeleted. Live-capital, DB/MCP, allocator, webhook, journal/profile, broad hook/drift, and research/OOS branches stay parked for separate focused review.

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

## Current Codex Follow-up - MNQ Open Book Validation
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Added post-exploratory prereg `docs/audit/hypotheses/2026-06-01-mnq-open-book-validation-v1.yaml` plus runner `research/mnq_open_book_validation_v1.py` to answer whether the prior O15/E2/RR2 NYSE_OPEN + US_DATA_1000 book was too narrow. The runner keeps the current two-book confirmation separate from a capped open challenger scan and filter diagnostic layer.
- **Result truth:** Canonical read-only run through `prereg_front_door --execute` wrote `docs/audit/results/2026-06-01-mnq-open-book-validation-v1.md`, `*-books.csv`, `*-pool.csv`, and `*-filter-diagnostics.csv`. The prereg is no-theory (`theory_grant=false`), so the runner uses the stricter t>=3.79 gate. Current COST_LT10 book remains the risk-aware winner over NO_FILTER (annual 67.59R, DD 23.42R, objective 2.885 vs NO_FILTER annual 67.69R, DD 24.42R, objective 2.771), but both are only `NARROW` because inherited broad-scan DSR remains low. Open challenger scan tested 50 pair books from a capped 11-cell pool; top risk-adjusted challenger is `NYSE_OPEN O15 RR2 COST_LT10 + US_DATA_1000 O15 RR1 COST_LT12` (annual 57.34R, DD 13.95R, objective 4.109, 2026 descriptive mean 0.419) but also remains `NARROW` under inherited DSR. Annual-only challenger is `NYSE_OPEN O15 RR2 COST_LT10 + NYSE_PREOPEN O30 RR2 COST_LT15` (annual 72.60R, DD 24.30R, 2026 descriptive mean 0.110), not a replacement.
- **Filter read:** Existing COST_LT filters are the only broad-positive family in this pass: COST_LT12/10/15/08 show positive median annual deltas and lower median DD versus matched NO_FILTER parents. Direction, ATR, ORB_SIZE, ORB_VOL, and ATR_VEL diagnostics are median-negative. Do not invent order-flow/absorption filters under current 1m OHLCV; park as new-data work.
- **Interpretation:** Current book is not too specific as a current practical baseline, but the best next hypothesis should test US_DATA_1000 RR1/RR1.5 as a book leg against RR2 under a fresh prereg, not silently alter the candidate. No deployment claim.
- **Verification:** `python -m py_compile research\mnq_open_book_validation_v1.py`, `python -m pytest tests\test_research\test_mnq_open_book_validation_v1.py -q`, and `python research\mnq_open_book_validation_v1.py` passed/completed against shared canonical `C:\Users\joshd\canompx3\gold.db`.

## Current Codex Follow-up - MNQ US_DATA RR Leg Choice
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Added no-theory conditional-role prereg `docs/audit/hypotheses/2026-06-01-mnq-usdata-rr-leg-choice-v1.yaml` plus runner `research/mnq_usdata_rr_leg_choice_v1.py`. This fixes the NYSE_OPEN anchor leg at `MNQ_NYSE_OPEN_O15_E2_RR2_COST_LT10` and only varies the US_DATA_1000 O15 E2 leg over RR {1.0, 1.5, 2.0} x filters {NO_FILTER, COST_LT08, COST_LT10, COST_LT12, COST_LT15}; K=15, 2026 frozen descriptive only.
- **Result truth:** Canonical read-only front-door run wrote `docs/audit/results/2026-06-01-mnq-usdata-rr-leg-choice-v1.md` and `docs/audit/results/2026-06-01-mnq-usdata-rr-leg-choice-v1-books.csv`. Risk-adjusted winner is `NYOPEN_USDATA_RR1_NO_FILTER` (annual 58.52R, DD 13.95R, objective 4.193, t=5.70, WFE=1.115, 2026 descriptive mean 0.419). Annual-only winner is `NYOPEN_USDATA_RR2_NO_FILTER` (annual 68.50R, DD 24.42R, objective 2.805). Current comparison row `NYOPEN_USDATA_RR2_COST_LT10` remains annual 67.59R, DD 23.42R, objective 2.885.
- **Interpretation:** The data says the next capital-aware book design should compare drawdown-budgeted RR1/RR1.5 US_DATA legs against the current RR2 leg, not deploy a silent replacement. All 15 rows are `NARROW`, 0 `CONTINUE`, 0 `KILL`; inherited broad-scan DSR remains the blocker despite family BH/DSR being strong.
- **Verification:** Front door accepted and executed. `python -m pytest tests\test_research\test_mnq_usdata_rr_leg_choice_v1.py tests\test_research\test_mnq_open_book_validation_v1.py tests\test_research\test_best_own_strategy_scan_v1.py tests\test_research\test_orb_execution_variants_v1.py -q` passed 20 tests. Ruff, claim hygiene, and `git diff --check` passed.

## Current Codex Follow-up - MNQ US_DATA Capital Fit
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Added allocator/account-fit prereg `docs/audit/hypotheses/2026-06-01-mnq-usdata-capital-fit-v1.yaml` plus runner `research/mnq_usdata_capital_fit_v1.py`. It fixes the same 15 US_DATA leg-choice books and maps 1-10 MNQ contracts per leg through `topstep_50k_mnq_auto` constraints using repo-owned profile data and Topstep scaling math.
- **Result truth:** Canonical read-only front-door run wrote `docs/audit/results/2026-06-01-mnq-usdata-capital-fit-v1.md`, `*-books.csv`, and `*-sizing.csv`. All 15 raw two-leg books are `KILL` under the active profile: no contract size from 1-10 is profile-safe. Even the lower-DD RR1/no-filter row has 1-contract annual $6,799 but max DD $2,711 and 55 historical daily-belt breaches; current RR2/COST_LT10 has 1-contract annual $7,947 but max DD $3,549 and 62 daily-belt breaches. Raw annual-dollar winner is RR2/no-filter ($7,962) but it is not profile-safe.
- **Interpretation:** Account constraints, not RR choice, are now the binding issue. Do not promote the raw two-leg book. The next honest hypothesis is a risk-overlay family: profile stop_multiplier=0.75, max ORB/risk cap, sequential one-loss/daily-belt throttle, or lower-risk lane replacement.
- **Verification:** Front door accepted and executed. `python -m pytest tests\test_research\test_mnq_usdata_capital_fit_v1.py -q` passed 4 tests. Ruff, py_compile, claim hygiene, and `git diff --check` passed.

## Current Codex Follow-up - MNQ Open Book Risk Overlay
- **Tool:** Codex
- **Date:** 2026-06-01
- **Summary:** Added strict K=28 conditional-role prereg `docs/audit/hypotheses/2026-06-01-mnq-open-book-risk-overlay-v1.yaml` plus runner `research/mnq_open_book_risk_overlay_v1.py`. It tests four fixed NYSE_OPEN+US_DATA book shapes against seven structural overlays: raw, stop 0.75, risk caps at $225/$300, stop+cap combinations, and realized-loss throttle. Grounding is local: conditional-edge framework, Criterion 11 account survival, Lopez de Prado finite-data risk/bet-sizing framing, and `TRADING_RULES.md` stop/cap doctrine.
- **Result truth:** Canonical read-only front-door run wrote `docs/audit/results/2026-06-01-mnq-open-book-risk-overlay-v1.md` and `*-candidates.csv`. All 28 candidates are `KILL`; no overlay is profile-safe at one MNQ contract per leg. Highest annual candidate is `RAW_ANNUAL_RR2_NO_FILTER__RISK_CAP_300` (annual $8,257, DD $3,250, 25 daily-belt breaches, survival 0.223). Best near-miss risk control is `LOW_DD_RR1_NO_FILTER__RISK_CAP_225` (annual $6,947, DD $2,079, 0 daily-belt breaches, survival 0.895), but it still exceeds the 80% max-loss DD budget ($1,600).
- **Interpretation:** Risk caps help the daily-belt problem but do not solve drawdown. Stop 0.75 worsens drawdown in this two-leg book under the canonical MAE simulation. Realized-loss throttle is mostly ineffective because losses are not known before many later entries. The next honest path is lower-risk lane replacement / single-leg allocation, not more two-leg sizing.
- **Verification:** Front door accepted and executed. `python -m pytest tests\test_research\test_mnq_open_book_risk_overlay_v1.py -q` passed 3 tests. Ruff passed after fixing the loader to include `symbol` for canonical COST_LT filters.

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
- **Tool:** Claude Code
- **Date:** 2026-06-04
- **Commit:** b73271f4 — fix(hooks): resolve real git-dir in post-commit HANDOFF-skip guard (worktree footgun)
- **Files changed:** 2 files
  - `.githooks/post-commit`
  - `.githooks/pre-commit`

## Current Codex Follow-up - Live Readiness And Drift Fast Closeout
- **Tool:** Codex
- **Date:** 2026-06-03
- **Summary:** Completed the remaining high-EV live-readiness/dashboard/drift closeout from the operator prompt. Fresh `account_survival` for `topstep_50k_mnq_auto` passed Criterion 11 at 95.2% operational pass. Fresh `live_readiness_report --strict-zero-warn --proof-pack-only` showed C11 pass age 0d, C12 valid age 1d, three active lanes, zero stale lanes, no missing evidence, and only telemetry maturity advisory (9/30 days). Dashboard test smoke passed 42 tests. Phase 7 live audit passed 11 checks.
- **Drift fix:** Root-caused the fast drift timeout to stale fast-skip coverage plus repeated relative-volume enrichment in `StrategyTradeWindowResolver`. Added resolver caching for same `(instrument, orb_minutes, orb_label, lookback_days)` relative-volume enrichment and added a focused regression test. Updated `SLOW_CHECK_LABELS` with slow labels measured in this session so `--fast` skips slow checks while full pre-commit/CI still retain coverage.
- **Verification:** `python -m pytest tests\test_pipeline\test_check_drift_slow_labels.py tests\test_trading_app\test_validation_provenance.py tests\test_pipeline\test_check_drift_db.py::TestActiveMicroOnlyFiltersAfterMicroLaunch -q` passed 8 tests; `python -m pytest tests\test_trading_app\test_bot_dashboard.py -q` passed 42 tests; scoped `ruff check` and `py_compile` passed; `python -u pipeline\check_drift.py --fast --quiet --skip-crg-advisory` completed with `SUMMARY: clean passed=137 advisory=15`; `python scripts\audits\run_all.py --phase 7` passed. Known residual: pytest emitted ignored Windows temp cleanup `PermissionError` after successful runs.

## Worktree-Guard Self-DOS Message Fix (Claude, 2026-06-03)
- **Tool:** Claude Code
- **Commit (pre-rebase):** 5e3d6dc3 / 71056eb2 — fix(hooks): name START_WORKTREE.bat in guard BLOCK messages (self-DOS fix)
- **Files changed:** 4 files
  - `.claude/hooks/branch-flip-guard.py`
  - `.claude/hooks/mcp-git-guard.py`
  - `.claude/hooks/worktree_guard.py`
  - `docs/runtime/stages/worktree-guard-selfdos-message-fix.md`
- **Summary:** Guard BLOCK messages told the operator to run `scripts/tools/new_session.sh` through the very Bash tool the guard blocks (self-DOS). Messages now lead with `START_WORKTREE.bat` (the Windows launcher outside the blocked Bash surface). Message-string-only edits — no logic/exit-code/matcher changes. Rebased onto `origin/main` `c7a6ac05`; worktree_guard.py resolution KEEPS main's `2e8f3b59` cwd-scoping/mutation-detection logic AND this branch's START_WORKTREE.bat message.

## F2-A Landing — self_funded contract-cap leak fix (Claude, 2026-06-03)
- **Tool:** Claude Code
- **Summary:** Landed the F2-A capital-path fix in isolated worktree `canompx3-f2a-land` (branch `session/joshd-f2a-land` off `origin/main` `fa98bf86`). Merged `origin/session/joshd-f2a-self-funded-sizing` (was 43 behind / 4 ahead). `prop_portfolio.select_for_profile` now makes `contract_budget` firm-aware: `None` for `self_funded` (prop micro-cap no longer gates a personal-capital book — risk/DD/slot budgets still bind), `tier.max_contracts_micro` for prop firms (unchanged). Honors `.claude/rules/self-funded-sizing-doctrine.md`. No schema/trading-logic change beyond the scoped cap-leak fix; gold.db read-only.
- **Conflicts resolved:** HANDOFF.md (kept current main baton, appended this note); `tests/test_scripts/test_start_topstep_live_pilot.py` (accepted main's delete — dead START_LIVE_PILOT path, not resurrected).
- **Status:** NOT merged to main — awaiting operator approval.

## Current Codex Follow-up - Dashboard Live CTA Visibility
- **Tool:** Codex
- **Date:** 2026-06-02
- **Summary:** Fixed the dashboard focus-mode gap where the only usable `HOLD TO GO LIVE` control lived inside profile/account cards hidden behind SHOW ALL/drawer views. Added a first-viewport topbar `HOLD TO GO LIVE` control for the pinned `topstep_50k_mnq_auto` MNQ pilot. It stays hidden while a session is running, disabled with a surfaced blocker while gates/operator state are not ready, and uses the same 2-second hold-to-confirm path as the existing profile-card live control.
- **Safety note:** No backend live-launch relaxation. The button still calls the existing dashboard `launchSession(..., "live", {skipConfirm: true})` only after the hold gesture; server-side `/api/action/start?mode=live` still runs the live preflight and strict live gating before starting `scripts.run_live_session --live --auto-confirm`.
- **Verification:** Inline dashboard JS parse check passed. `python -m pytest tests\test_trading_app\test_bot_dashboard.py -q` passed 39 tests. Mocked browser render at `127.0.0.1:8093` showed a visible enabled topbar live button in the first viewport; helper server was stopped after the check. `git diff --check` passed.

## Current Codex Follow-up - F4-A/F4-B Closeout
- **Tool:** Codex
- **Date:** 2026-06-02
- **Summary:** Merged the previously pushed F4-A branch `origin/session/joshd-f4a-branch-flip-fix` into `main`. The branch scopes `branch-flip-guard.py`, `mcp-git-guard.py`, and `head-flip-guard.py` to the PostToolUse payload `cwd` via `_branch_state.invoking_cwd(event)`, so the guards inspect the worktree where the tool actually ran instead of the hook process cwd in the main checkout. PR #348 / F4-B was already merged before this closeout, so the stale-MCP-restart warning path is present on main.
- **Verification:** `python -m pytest tests/test_hooks/test_branch_state.py tests/test_hooks/test_branch_flip_guard.py tests/test_hooks/test_mcp_git_guard.py tests/test_hooks/test_head_flip_guard.py -q` passed 50 tests. Scoped `ruff check` on the changed hook/test files passed after mechanical import/f-string cleanup. `git diff --check` passed. `python scripts/tools/audit_behavioral.py` and `python scripts/tools/audit_integrity.py` passed. `python scripts/tools/project_pulse.py --fast --format json` reported `broken=0`.
- **Residual gap:** `python pipeline/check_drift.py`, `python pipeline/check_drift.py --skip-crg-advisory --quiet`, and `python pipeline/check_drift.py --fast --quiet` all timed out locally before a summary. A 60s unbuffered probe showed fast drift progressing through `Active micro-only filters run only on real-micro instruments` before stalling on the next drift check; this appears unrelated to the F4-A hook merge but remains unclosed.

## Current Codex Follow-up - Highest-Risk Commit Review
- **Tool:** Codex
- **Date:** 2026-06-03
- **Summary:** Reviewed the recent highest-risk work surfaces (self-funded contract-cap fix, live journal-lock diagnostics, MNQ single-leg replacement research, and Slack control-room design). Fixed the research-monitoring bug in `research/mnq_single_leg_account_fit_replacement_v1.py`: replacement verdict gates now remain strictly pre-2026 in-sample while the full locked calendar is still passed through scoring so `mean_2026_*` monitoring fields are populated instead of silently NaN. Added a regression test proving 2026 holdout losses are reported but do not affect account-safe/verdict gates. Follow-up cleanup renamed the scoring inputs from misleading `book_is`/`trades_is` to `book`/`trades` so the API matches the full-calendar monitoring split. Second review pass fixed the report surface so Markdown rankings also expose `mean_2026_dollars` and `mean_2026_r`, removed transient `HANDOFF.md` conflict markers, and regenerated the MNQ result doc/CSV against `C:\Users\joshd\canompx3\gold.db`.
- **Verification:** `python -m pytest tests/test_research/test_mnq_single_leg_account_fit_replacement_v1.py -q` passed 10 tests with known pytest config warnings and an ignored Windows temp cleanup `PermissionError` after the pass. `python research\mnq_single_leg_account_fit_replacement_v1.py` wrote 15 scenarios and the report against canonical `gold.db`; result CSV has 15 `KILL` rows, zero `mean_2026_*` nulls, and zero account-safe rows. `ruff check`, `ruff format --check`, `python -m py_compile`, prereg front-door text route, `project_pulse.py --fast --format json` (`broken=0`), and `git diff --check` passed. `python pipeline\check_drift.py --fast --quiet` timed out after 184s before a summary, so full fast-drift remains unclosed in this local run.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`

## Current Codex Follow-up - Maximise No-Tunnel-Vision Sprint
- **Tool:** Codex
- **Date:** 2026-06-03
- **Branch:** `session/joshd-maximise-no-tunnel-vision`
- **Summary:** Built the opportunity map across allocation/live-readiness, ASX-open research, worktree/hook friction, dashboard readiness, drift speed, and stale-doc risk. Fresh DB-backed allocation verification is blocked because canonical `/workspace/canompx3/gold.db` is absent in this WSL checkout. Chosen action was the smallest high-EV operational fix: `workflow_doctor` now recommends opening an isolated worktree (`START_WORKTREE.bat <descriptor>` or `scripts/tools/new_session.sh <descriptor>`) for a live `peer_lease` instead of only inspecting the holder. This keeps live peer leases intact and removes the ambiguous force-release temptation.
- **Verification:** `python -m pytest tests/test_tools/test_workflow_doctor.py -q` passed 22 tests with known pytest config warnings. `python -m pytest tests/test_tools/test_worktree_guard.py tests/test_tools/test_worktree_launch_preflight.py tests/test_tools/test_workflow_doctor.py -q` passed 73 tests / 1 skipped with the same config warnings. `ruff check`, `ruff format --check`, `python -m py_compile scripts/tools/workflow_doctor.py`, and `git diff --check` passed on the changed code paths. A mistaken probe for nonexistent `tests/test_hooks/test_worktree_guard_hook.py` failed with pytest exit 4 before collecting tests; reran the correct guard/launcher/workflow-doctor set successfully.

## 2026-06-04 Codex plan update — live app operability/readiness

- Created `docs/plans/2026-06-04-live-app-operability-readiness-plan.md`.
- Plan separates dashboard-open, signal/read-only usability, and live-execution gates.
- Key direction: dashboard should fail open/degraded; live launch must remain fail-closed.
- Workstreams: dashboard startup, clean runtime worktree/git lease, DB-safe snapshot refresh, preflight split, C11/C12 clarity, lane/account frontier, recent-commit integration audit, operator runbook boundaries.
- No database, broker, or live runtime state was inspected. Current decision remains NO-GO until measured gates pass.
- 2026-06-04 follow-up revision: plan now explicitly prioritizes smallest useful diffs first: doc clarification, report metadata, dashboard render-only blocker cards, worktree diagnostics, then one atomic snapshot before any runtime-worktree/scheduler/lane changes.
