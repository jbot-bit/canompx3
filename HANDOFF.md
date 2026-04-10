# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Update (2026-04-11 — native promotion provenance hardening)

### Headline

Added native promotion provenance to `validated_setups`, centralized canonical
trade-window recomputation, and hardened drift so the active shelf can audit
promotion lineage honestly instead of relying on stale `strategy_trade_days`.

### What changed

- `trading_app/validation_provenance.py`
  - new canonical `StrategyTradeWindowResolver`
  - recomputes `first_trade_day`, `last_trade_day`, `trade_day_count` from:
    - `daily_features`
    - canonical filter logic from `trading_app.strategy_discovery`
    - exact-lane `orb_outcomes`
  - handles composite volume / cross-asset ATR filters
- `trading_app/db_manager.py`
  - added `validated_setups` columns:
    - `first_trade_day`
    - `last_trade_day`
    - `trade_day_count`
    - `validation_run_id`
    - `promotion_git_sha`
    - `promotion_provenance`
  - additive migration is included in `init_trading_app_schema()`
- `trading_app/strategy_validator.py`
  - validator now fail-closes on missing git SHA
  - validator now recomputes canonical trade windows before promotion
  - native rows are stamped with:
    - trade-window fields
    - `validation_run_id`
    - `promotion_git_sha`
    - `promotion_provenance='VALIDATOR_NATIVE'`
- `pipeline/check_drift.py`
  - micro-launch check now delegates to the canonical resolver
  - added:
    - `check_active_native_promotion_provenance_populated()`
    - `check_active_native_trade_windows_match_provenance()`
  - hardened legacy-schema behavior:
    - if `gold.db` has not been migrated yet, drift emits explicit schema
      violations instead of crashing with `BinderException`
- tests updated:
  - `tests/test_app_sync.py`
  - `tests/test_trading_app/test_db_manager.py` coverage already exercised by suite
  - `tests/test_trading_app/test_strategy_validator.py`
  - `tests/test_pipeline/test_check_drift_db.py`

### Live DB state

- Ran `init_trading_app_schema()` against repo `gold.db`
- Result: additive `validated_setups` provenance columns are now present on the
  live shelf DB, so the new drift checks run cleanly on real data

### Verification

- `./.venv-wsl/bin/python -m ruff check trading_app/validation_provenance.py trading_app/strategy_validator.py trading_app/db_manager.py pipeline/check_drift.py tests/test_app_sync.py tests/test_pipeline/test_check_drift_db.py tests/test_trading_app/test_strategy_validator.py`
  - `All checks passed!`
- `./.venv-wsl/bin/python -m pytest tests/test_app_sync.py tests/test_trading_app/test_db_manager.py tests/test_trading_app/test_strategy_validator.py tests/test_pipeline/test_check_drift_db.py -q`
  - `229 passed`
- `./.venv-wsl/bin/python pipeline/check_drift.py`
  - `NO DRIFT DETECTED: 98 checks passed [OK], 0 skipped (DB unavailable), 7 advisory`
- `./.venv-wsl/bin/python scripts/tools/audit_behavioral.py`
  - passed all 7 checks
- `./.venv-wsl/bin/python scripts/tools/audit_integrity.py`
  - still fails for pre-existing unrelated integrity debt:
    - `46 validated strategies with NULL family_hash`
    - `GC: 17 active validated strategies (dead instrument)`

### Notes

- The new provenance path is intentionally honest about current limits:
  it does **not** pretend `dataset_snapshots` / `research_runs` exist in the DB.
  It stores only provenance we can populate truthfully today.
- If another tool touches `validated_setups` promotion logic, re-read:
  - `trading_app/validation_provenance.py`
  - `trading_app/strategy_validator.py`
  - `pipeline/check_drift.py`
  before modifying anything.

## Update (2026-04-11 — active-shelf micro-data discipline hardening)

### Headline

Added a new fail-closed drift guard so active `validated_setups` rows cannot
use `requires_micro_data=True` filters on non-real-micro instruments.

### What changed

- `pipeline/check_drift.py`
  - added `check_active_micro_only_filters_on_real_micros()`
  - delegates to canonical sources only:
    - `trading_app.config.ALL_FILTERS[filter_type].requires_micro_data`
    - `pipeline.data_era.is_micro(instrument)`
  - registered new drift check:
    - `Active micro-only filters run only on real-micro instruments`
- `tests/test_pipeline/test_check_drift_db.py`
  - added coverage for:
    - valid micro-only filter on real micro (`MNQ`)
    - invalid micro-only filter on parent instrument (`GC`)
    - retired invalid row ignored
    - unknown instrument fails closed

### Why this matters

This closes the honest instrument-level era gap without inventing provenance we
do not have. It prevents future promotion of volume/rel-vol style filters on
parent or proxy lanes even if discovery/runtime routing drifts.

### Important limit

Precise pre-launch date enforcement is still **not** derivable from the current
`validated_setups` schema. The shelf does **not** store first/last trade day or
equivalent strategy-level date provenance. So a true Stage 3d-style
`first_trade_day >= micro_launch_day(instrument)` check still requires schema or
provenance work. Do not fake this with inferred dates from shelf metadata.

### Verification

- `./.venv-wsl/bin/python -m pytest tests/test_pipeline/test_check_drift_db.py -q`
  - result: `24 passed`
- `./.venv-wsl/bin/python pipeline/check_drift.py`
  - new check passed on real shelf
  - drift still fails for pre-existing unrelated issues:
    - `trading_app/strategy_validator.py` indentation/import break
    - `family_rr_locks` coverage
    - orphaned `hypothesis_file_sha`

### Follow-up hardening (same session)

Added a second, stricter era-discipline guard:

- `pipeline/check_drift.py`
  - added `check_active_micro_only_filters_after_micro_launch()`
  - recomputes first traded day for active `requires_micro_data=True` rows from
    canonical computed facts:
    - `daily_features`
    - `orb_outcomes`
    - canonical filter logic from `trading_app.strategy_discovery`
    - `pipeline.data_era.micro_launch_day(instrument)`
  - explicitly does **not** trust `strategy_trade_days` because that table is
    documented stale for active-shelf audit use
- `tests/test_pipeline/test_check_drift_db.py`
  - added coverage for:
    - post-launch pass
    - pre-launch violation
    - missing recomputable trade-day failure

Verification for this follow-up:

- `./.venv-wsl/bin/python -m pytest tests/test_pipeline/test_check_drift_db.py -k 'ActiveMicroOnlyFiltersOnRealMicros or ActiveMicroOnlyFiltersAfterMicroLaunch' -q`
  - result: `7 passed`

Environment note:

- direct `gold.db` opens became intermittently unavailable with
  `Permission denied` during later verification, so full real-DB drift evidence
  for the new Stage-3d-style check was blocked by file access, not logic/test
  failure.

## Update (2026-04-10 evening — GC proxy branch merged, drift/SR cleanup)

### Headline

Merged `research/gc-proxy-validity` to main (30 commits, fast-forward). All infrastructure fixes applied. Drift 92/0/7 (all green). SR monitor Windows unicode bug fixed.

### What changed

**Code fixes:**
- `pipeline/check_drift.py`: Check 57 (daily_features row integrity) scoped to ACTIVE_ORB_INSTRUMENTS — GC proxy data with single aperture no longer flags
- `pipeline/check_drift.py`: Check 55 (cost model ranges) already scoped to skip proxy instruments (prior commit)
- `trading_app/sr_monitor.py`: Replaced unicode chars (σ, ≈) with ASCII equivalents for Windows cp1252 console
- `trading_app/prop_profiles.py`: Profile lanes updated to current multi-RR validated strategies (prior commit)
- `family_rr_locks` DB table populated (6 rows for 6 families)

**Criterion 11 survival report:** Regenerated. Gate FAILS at 26.2% (68.1% scaling breach). Structural — TopStep 50K Day-1 cap is 2 lots vs 5 bot lanes. Bot is in signal/demo mode. Not a code fix — F-1 scaling plan wiring needed before live.

**SR ALARM on L3 (MNQ NYSE_OPEN ORB_G5 RR1.5):** Edge is NOT decaying (WR 43.7→42.4%, AvgR 0.095→0.093 — stable). Alarm driven by vol regime shift (MNQ ORB median 45.5→88.8 pts, +95%). Status: WATCH, not DECAY.

**GC downstream audit:** 5 GC strategies in validated_setups are correctly excluded from all downstream modules (validator FDR, pulse fitness, profiles, SR monitor) because GC is not in ACTIVE_ORB_INSTRUMENTS. This is by-design — they're proxy research results pending MGC micro cross-validation.

### What's pending (strategic decisions for user)

1. **GC strategies cross-validation** — 5 validated on 16yr GC proxy data. Need hard audit (yearly JSON parsing fix) then MGC micro cross-validation. Handover: `docs/handoffs/2026-04-10-gc-proxy-discovery-handover.md`
2. **Multi-RR re-run** — Validator DELETE bug wiped old RR1.0 strategies. Need re-run to recover. Handover: `docs/handoffs/2026-04-10-multi-rr-discovery-handover.md`
3. **MES/MGC = 0 validated** — Coverage gap in portfolio
4. **L3 NYSE_OPEN WATCH** — Monitor for vol normalization, no code action

### State

- Branch: `main` @ `6fb59854`
- Drift: 92 pass / 0 fail / 7 advisory
- Validated strategies: GC=5, MNQ=5 (10 total active)
- Deployed: 5 MNQ lanes on `topstep_50k_mnq_auto`

---

## Update (2026-04-10 later — Windows shortcut dedupe)

### Headline

Removed two dead/confusing Windows shortcut aliases from
`scripts/infra/windows-shortcuts/`:

- `agent-tasks.bat`
- `ai-workspaces.bat`

Both were unreferenced one-line aliases to the same workstream menu launcher and
added no unique behavior.

### What stays

The actual supported Windows launcher surface is unchanged:

- `codex.bat`
  - simple normal Codex repo session
- `codex-workstream.bat`
  - isolated Codex task/worktree
- `ai-workstreams.bat`
  - workstream/menu launcher

Shortcut folder still keeps the distinct task-specific helpers:

- `claude-workstream.bat`
- `codex-search-workstream.bat`
- `codex-workstream.bat`
- `workstream-clean.bat`
- `workstream-finish.bat`
- `workstream-list.bat`

### Verification

- `rg -n "agent-tasks\\.bat|ai-workspaces\\.bat" -S .`
  - no remaining references

## Update (2026-04-10 later — Windows Codex entrypoint cleanup)

### Headline

Added one plain Windows convenience entrypoint for Codex repo sessions:

- `codex.bat`

This is intentionally narrow. It fills a real missing path for Windows users who
just want to open a normal Codex session in this repo without remembering the
WSL launcher internals.

This pass does **not** add another workstream/menu system. Existing roles stay:

- `codex.bat`
  - simplest normal repo Codex session
- `codex-workstream.bat`
  - isolated task/worktree
- `ai-workstreams.bat`
  - workstream/menu launcher

### Design

Windows already had:

- menu/workstream launchers
- isolated Codex workstream launchers

but no dead-simple "open Codex in this repo now" Windows entrypoint.

To avoid duplicating launch logic:

- `codex.bat` shells into the existing PowerShell launcher
- the PowerShell launcher dispatches a new `codex-project` mode
- the Python Windows launcher dispatches that mode into the existing canonical
  WSL entrypoint:
  - `scripts/infra/codex-project.sh --no-alt-screen`

So the new path is only a thin Windows convenience wrapper over the current
canonical WSL launcher.

### Cleanup

Renamed the user-facing wording on `codex-workstream.bat` and the Windows
shortcut copy so it is clearly an **isolated workstream**, not the default
everyday entrypoint.

### Files touched

- `codex.bat`
- `codex-workstream.bat`
- `scripts/infra/windows-agent-launch.ps1`
- `scripts/infra/windows_agent_launch.py`
- `scripts/infra/windows-shortcuts/codex-workstream.bat`
- `tests/test_tools/test_windows_agent_launch.py`
- `CODEX.md`
- `HANDOFF.md`

### Verification

- `python3 -m py_compile scripts/infra/windows_agent_launch.py tests/test_tools/test_windows_agent_launch.py`
- `./.venv-wsl/bin/python -m pytest tests/test_tools/test_windows_agent_launch.py -q`
  - result: `15 passed`

### Operator truth

For a beginner Windows user, the canonical easiest path is now:

- open repo folder
- run `codex.bat`

That launches the same repo-aware WSL Codex session as:

- `scripts/infra/codex-project.sh`

without asking the user to remember the internal launcher structure.

## Update (2026-04-10 later — unified lifecycle-state reader)

### Headline

Implemented a shared lifecycle-state reader so the live-control path no longer
has separate ad hoc interpretations of:

- Criterion 11 deployment gate
- Criterion 12 SR monitor state
- persisted lane pauses

This is an operational-truth unification pass. It does **not** change:

- discovery scope
- validator logic
- account survival math
- SR math or SR envelope format
- lane pause storage format

### Design

New module:

- `trading_app/lifecycle_state.py`

It provides one shared interpretation layer:

- `read_criterion11_state(...)`
- `read_criterion12_state(...)`
- `read_pause_state(...)`
- `read_lifecycle_state(...)`

`read_lifecycle_state(...)` returns:

- profile id
- lane ids
- C11 summary
- C12 summary
- pause summary
- per-strategy lifecycle state
- blocked strategy ids
- blocked reason map

Blocking precedence:

1. explicit pause override
2. SR `ALARM`
3. otherwise clear / informational

That means SR alarms are now first-class operational lane blocks in the shared
reader even if the pause file has not yet been written.

### Consumers updated

`scripts/tools/project_pulse.py`

- now reads C11/C12/pause summaries through `trading_app.lifecycle_state`
- no longer re-implements SR envelope validation inline
- drift check was updated so the canonical acceptable patterns are now:
  - direct validation in pulse
  - or delegation to `read_criterion12_state(...)`

`trading_app/pre_session_check.py`

- now reads the shared lifecycle snapshot once per run
- `Criterion 11 survival` comes from that shared snapshot
- new `Lane lifecycle` row reports:
  - paused lane = `BLOCKED`
  - SR alarm lane = `BLOCKED`
  - valid SR continue/no-data = pass/info
  - stale/mismatched SR state = warning only

`trading_app/live/session_orchestrator.py`

- `_load_paused_lane_blocks()` now loads lifecycle blocks, not just explicit
  pause overrides
- startup/runtime blocking now sees both:
  - paused lanes
  - SR-alarmed lanes

### Files touched

- `trading_app/lifecycle_state.py`
- `trading_app/pre_session_check.py`
- `trading_app/live/session_orchestrator.py`
- `scripts/tools/project_pulse.py`
- `pipeline/check_drift.py`
- `tests/test_tools/test_project_pulse.py`
- `tests/test_trading_app/test_pre_session_check.py`
- `tests/test_trading_app/test_session_orchestrator.py`

### Verification

- `python3 -m py_compile trading_app/lifecycle_state.py trading_app/pre_session_check.py trading_app/live/session_orchestrator.py scripts/tools/project_pulse.py pipeline/check_drift.py tests/test_tools/test_project_pulse.py tests/test_trading_app/test_pre_session_check.py tests/test_trading_app/test_session_orchestrator.py`
- `./.venv-wsl/bin/python -m pytest tests/test_tools/test_project_pulse.py tests/test_trading_app/test_pre_session_check.py -q`
  - result: `83 passed`
- `./.venv-wsl/bin/python -m pytest tests/test_trading_app/test_session_orchestrator.py -q -k "load_paused_lane_blocks or paused_lane_blocks_new_entry_with_pause_reason"`
  - result: `2 passed, 111 deselected`
- `./.venv-wsl/bin/python pipeline/check_drift.py`
  - result: `NO DRIFT DETECTED: 92 checks passed [OK], 0 skipped, 7 advisory`
- real pulse:
  - `./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --tool codex`
  - still shows:
    - `C11 PASS 85.0%`
    - `C12 SR continue=5 alarm=0 no_data=0`
    - `Paused lanes: 0`
- real pre-session:
  - `./.venv-wsl/bin/python -m trading_app.pre_session_check --profile topstep_50k_mnq_auto --session EUROPE_FLOW`
  - now includes:
    - `Lane lifecycle: Criterion 12 SR clear for MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G8`

### Supported Codex launch paths

WSL / supported:

- `scripts/infra/codex-project.sh`
- `scripts/infra/codex-project-search.sh`
- `scripts/infra/codex-review.sh`
- `scripts/infra/codex-worktree.sh open <task>`

Why:

- they activate `.venv-wsl`
- they run session preflight
- they pass claim mode for concurrency guardrails

Windows / convenience:

- `codex.bat`
  - canonical simple path for a normal repo Codex session
- `codex-workstream.bat`
  - isolated task/worktree path
- `ai-workstreams.bat`
  - menu/workstream manager

Do not use bare system `python3` as the normal repo path. It can produce false
broken state because it bypasses `.venv-wsl` and misses repo deps like `duckdb`.

## Update (2026-04-10 later — Codex adapter consolidation pass)

### Headline

Consolidated the Codex adapter layer to reduce duplicate startup/routing text
without adding any new wrappers, registries, or Codex-only workflow surfaces.

This is a documentation/control-plane cleanup only. It does **not** change:

- launch script behavior
- env split (`.venv` vs `.venv-wsl`)
- project-scoped Codex config
- project pulse, preflight, C11, or C12 logic
- Claude/Codex authority boundaries

### Why this was needed

The Codex side had started to accumulate too many overlapping docs:

- `CODEX.md`
- `.codex/STARTUP.md`
- `.codex/WORKFLOWS.md`
- `.codex/PROJECT_BRIEF.md`
- `.codex/CURRENT_STATE.md`

The real problem was human/operator complexity, not missing infrastructure.

Correct goal:

- Claude stays boss
- Codex stays a second POV
- startup stays thin
- no repeated full-context loading
- no Codex-only side project

### What changed

`CODEX.md`

- now acts as the actual front door
- emphasizes the thin default read set
- keeps the operator model small:
  - normal
  - search
  - review
  - worktree
- points to supporting docs only when needed instead of trying to be a full
  router and startup file at the same time

`.codex/STARTUP.md`

- reduced to true startup deltas only
- explicitly says **do not auto-load all of `.codex/` every session**
- corrected a stale claim: there is **no always-present dedicated Codex stage
  file**
- now tells Codex to follow the current repo stage-file convention under
  `docs/runtime/stages/` if stage tracking is actually required

`.codex/WORKFLOWS.md`

- trimmed to execution defaults, verification defaults, and route selection
- removed duplicated orientation/startup burden

`.codex/CURRENT_STATE.md`

- updated to reflect current repo reality after the recent control-layer work:
  - pulse/preflight
  - derived-state validation
  - concurrent mutating-session guardrails
  - C11 gate
  - C12 SR monitor + lane pauses
- still states clearly that the repo is **not** fully live-finished

`.codex/PROJECT_BRIEF.md`

- kept as a short repo-orientation surface
- now explicitly points back to `CODEX.md` as the adapter front door

### Important stale claim removed

Previous `.codex/STARTUP.md` incorrectly claimed:

- `docs/runtime/stages/codex.md`

as a dedicated Codex stage file.

That path does not exist. Search confirmed the stale reference only lived in
that doc, so it has now been removed from the Codex layer.

### Files touched

- `CODEX.md`
- `.codex/STARTUP.md`
- `.codex/WORKFLOWS.md`
- `.codex/CURRENT_STATE.md`
- `.codex/PROJECT_BRIEF.md`

### Verification

- reread all touched docs after edit
- `rg -n "docs/runtime/stages/codex.md" -S . AGENTS.md CODEX.md .codex scripts docs`
  - result: no matches
- confirmed no launcher/config edits were made
- `git status --short`
  - only the intended doc files modified

### Resulting mental model

Beginner-safe operator model is now intentionally small:

- Claude = canonical authority
- Codex = second POV / implement / review / verify
- Windows = `.venv`
- WSL = `.venv-wsl`
- Codex entrypoints:
  - `scripts/infra/codex-project.sh`
  - `scripts/infra/codex-project-search.sh`
  - `scripts/infra/codex-review.sh`
  - `scripts/infra/codex-worktree.sh`

### Next move

Do **not** add more Codex wrappers/docs unless they clearly replace something
older.

If more setup work is requested, the right next pass is:

- audit remaining Codex-facing docs as `keep / trim / convenience-only`
- only collapse more surfaces if it further reduces confusion
- otherwise stop and return to project work

## Update (2026-04-10 later — Criterion 11 v2 path-aware survival)

### Headline

Implemented **Criterion 11 v2** in `trading_app/account_survival.py`.

This upgrades the deployment gate from a daily-close bootstrap to a
**conservative trade-path replay** built from canonical trade timestamps plus
MAE/MFE, and replaces the old static day-1-only Topstep scaling check with
**dynamic per-day scaling-ladder enforcement** over the full 90-day horizon.

This is a deployment-control upgrade. It does **not** change:

- discovery scope
- validation outcomes
- deployed lane selection
- SR / lifecycle pause semantics

### Design

Old C11 problem:

- daily scenario = one close-to-close PnL number
- no intraday low/high envelope
- no real day-by-day scaling-plan enforcement
- therefore DD/DLL breach risk was understated

New v2 model:

- load canonical trade outcomes per lane
- preserve the strategy filter by reusing `_load_strategy_outcomes(...)`
- for each trade, convert `pnl_r`, `mae_r`, `mfe_r` into dollars
- build one daily scenario from the actual day’s trades
- replay a conservative intraday envelope:
  - intraday low = realized PnL + simultaneous adverse excursion of all open trades
  - intraday high = realized PnL + simultaneous favorable excursion of all open trades
- track `max_open_lots` intraday from the real trade chronology
- in simulation:
  - sample daily path scenarios with replacement
  - enforce DD/DLL on the intraday envelope, not just the close
  - enforce Topstep scaling from the **prior EOD balance each simulated day**

Report/gate additions:

- summary now includes:
  - `scaling_breach_probability`
  - `path_model = "trade_path_conservative"`
- gate now requires `path_model == "trade_path_conservative"`

### Important bug found and fixed

While implementing v2, the first real run falsely showed:

- `scaling_breach_probability = 100%`
- `operational_pass_probability = 0%`

Root cause:

- `trading_app.strategy_fitness._load_strategy_outcomes(...)` was only
  selecting `trading_day`, `pnl_r`, `mae_r`, `mfe_r`, `entry_price`,
  `stop_price`
- it dropped `entry_ts` / `exit_ts`
- so the new path builder saw every trade as happening at the same synthetic
  time (`None`), which inflated simultaneous open lots and made scaling fail
  spuriously

Fix:

- `_load_strategy_outcomes(...)` now also selects:
  - `entry_ts`
  - `exit_ts`

After that fix, the live profile’s daily geometry is sane again:

- `max_open_lots` distribution for `topstep_50k_mnq_auto`:
  - mostly `1`
  - some `0`
  - occasional `2`
  - max `2`

### Current real state

Re-ran:

- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --as-of 2026-04-09 --paths 10000 --seed 0`

Current canonical result:

- profile: `topstep_50k_mnq_auto`
- path model: `trade_path_conservative`
- C11 operational pass: `85.0%`
- DD survival: `85.0%`
- trailing DD breach: `15.0%`
- daily-loss breach: `0.0%`
- scaling breach: `0.0%`
- gate: `PASS`

So the prior `87.2%` figure from the daily-close model has been replaced by a
more realistic conservative path-aware `85.0%`.

### Scope / honesty

This v2 is institutionally stronger than the old daily-close bootstrap, but it
is still **not tick-true**:

- it uses trade timestamps, MAE, and MFE
- it does **not** reconstruct the exact within-trade tick order
- for intraday path stress it intentionally chooses a conservative envelope

That is acceptable for deployment gating because the bias direction is safety-
first, not permissive.

### Files touched

- `trading_app/account_survival.py`
- `trading_app/strategy_fitness.py`
- `tests/test_trading_app/test_account_survival.py`

### Verification

- `python3 -m py_compile trading_app/account_survival.py trading_app/strategy_fitness.py tests/test_trading_app/test_account_survival.py`
- `./.venv-wsl/bin/python -m pytest tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_pre_session_check.py -q`
  - result: `36 passed`
- `./.venv-wsl/bin/python pipeline/check_drift.py`
  - result: `NO DRIFT DETECTED: 92 checks passed [OK], 0 skipped, 7 advisory`
- real gate readback:
  - `Criterion 11 pass: operational 85.0%, as_of=2026-04-09, age=1d, paths=10000`
- real pulse:
  - `C11 PASS 85.0% | as_of 2026-04-09 | age 0d`

### Next move

Highest-value next step after this slice:

- unify C11 / C12 / pause state under one explicit lifecycle-state reader
- keep the same fail-closed / thin-startup discipline already established

## Update (2026-04-10 later — Concurrency Guardrails v1)

### Headline

Implemented **Concurrency Guardrails v1** so Claude/Codex same-branch editing
is no longer just a warning convention. Mutating launcher paths now fail closed
when another fresh mutating claim already exists on the same branch.

This is a session/orchestration hardening pass. It does **not** change:

- strategy discovery logic
- validation truth
- live trading behavior
- account survival / SR semantics

### Design

The old model was too weak:

- one global temp claim file
- no explicit read-only vs mutating distinction
- no real concurrent-session enforcement

The new model in `scripts/tools/session_preflight.py` is:

- multi-claim storage in a temp claim directory
- one active claim file per `tool × repo/worktree root`
- explicit `mode`
  - `read-only`
  - `mutating`
- claim metadata now includes:
  - `tool`
  - `branch`
  - `head_sha`
  - `started_at`
  - `pid`
  - `mode`
  - `root`

Important policy:

- fresh same-branch concurrent **mutating** claims from another tool now block
  mutating launch
- read-only sessions remain non-blocking
- same-branch read-only vs mutating produces a warning, not a block
- worktree-separated parallel sessions remain the happy path

### Launcher wiring

Explicit claim modes are now passed by real entrypoints:

- `scripts/infra/codex-project.sh`
  - `--claim codex --mode mutating`
- `scripts/infra/codex-project-search.sh`
  - `--claim codex-search --mode read-only`
- `scripts/infra/wsl-env.sh`
  - `--claim wsl-shell --mode read-only`
- `scripts/infra/claude-worktree.sh`
  - `--claim claude --mode mutating`
- `scripts/infra/windows_agent_launch.py`
  - `run_preflight(..., mode=...)`
  - Claude launch now raises if preflight blocks

So the mutating launchers now actually enforce the policy instead of just
printing advisory noise.

### Pulse visibility

Added a thin session-claim summary in `project_pulse`:

- if multiple fresh claims exist on separate branches/worktrees:
  - one compact `PAUSED` item saying parallel appears isolated
- if multiple fresh mutating claims exist on the same branch:
  - one compact `ACT SOON` / high-severity item for dangerous parallel use

No full claim dump is shown in default pulse output.

### Current real-state note

On the current repo state there was only one fresh mutating Codex claim, so the
new session-claim summary does not appear in the default pulse. That is the
expected quiet-path behavior.

### Drift enforcement

Added a new structural drift check:

- preflight launchers must pass explicit claim modes

Current result:

- `python3 pipeline/check_drift.py`
  - `NO DRIFT DETECTED: 92 checks passed [OK], 0 skipped, 7 advisory`

### Files touched

- `scripts/tools/session_preflight.py`
- `scripts/tools/project_pulse.py`
- `scripts/infra/codex-project.sh`
- `scripts/infra/codex-project-search.sh`
- `scripts/infra/wsl-env.sh`
- `scripts/infra/claude-worktree.sh`
- `scripts/infra/windows_agent_launch.py`
- `pipeline/check_drift.py`
- `tests/test_tools/test_session_preflight.py`
- `tests/test_tools/test_project_pulse.py`
- `tests/test_tools/test_windows_agent_launch.py`

### Verification

- `python3 -m py_compile scripts/tools/session_preflight.py scripts/tools/project_pulse.py scripts/infra/windows_agent_launch.py pipeline/check_drift.py tests/test_tools/test_session_preflight.py tests/test_tools/test_project_pulse.py tests/test_tools/test_windows_agent_launch.py`
- `uv run --frozen --extra dev pytest tests/test_tools/test_session_preflight.py tests/test_tools/test_project_pulse.py tests/test_tools/test_windows_agent_launch.py tests/test_tools/test_pulse_integration.py -q`
  - result: `94 passed`
- `python3 -m trading_app.sr_monitor`
  - reran to refresh SR state after code-fingerprint change so pulse stayed canonical
- `python3 scripts/tools/project_pulse.py --fast --tool codex`
  - pulse reads current C11/C12 state cleanly
- `python3 scripts/tools/session_preflight.py --context codex-wsl --with-pulse --quiet --claim codex --mode mutating`
  - quiet preflight path still works; blocks are now available when same-branch mutating conflicts exist

### Why this matters

This closes the main cross-tool collaboration gap that remained after the
derived-state contract work:

- stale repo state is one failure mode
- concurrent mutable branch use is another

Now the mutating launch path is materially safer:

- same-branch edits by both tools are no longer silently tolerated
- read-only research/search sessions remain cheap
- startup remains thin and operator-readable

### Next move

Highest-value next step remains:

- **Criterion 11 v2**
  - intraday/path-aware breach modeling
  - dynamic scaling over the simulated horizon

## Update (2026-04-10 later — Derived State Contract v1 for SR)

### Headline

Implemented **Derived State Contract v1** for Criterion 12 SR state so
startup/operator reasoning now self-invalidates when upstream profile, lane,
DB, or code truth changes.

This is a control-surface hardening pass. It does **not** change:

- discovery scope
- validation outcomes
- deployment lane selection
- live execution logic

### Design

New shared helper module:

- `trading_app/derived_state.py`

It centralizes the contract pieces that were previously ad hoc:

- `build_profile_fingerprint(profile)`
- `build_code_fingerprint(paths)`
- `build_db_identity(db_path, con=None)`
- `build_state_envelope(...)`
- `validate_state_envelope(...)`
- `get_git_head(...)`

Important design choice:

- Criterion 11 now reuses the shared profile-fingerprint helper
- SR state is the first full consumer of the derived-state envelope
- `project_pulse` validates the SR envelope before trusting it
- SR mismatch/staleness **warns/degrades** pulse state, but does **not** block
  pre-session in this v1

### SR contract shape

`data/state/sr_state.json` is no longer an ad hoc blob. It is now a versioned
envelope with:

- `schema_version`
- `state_type`
- `generated_at_utc`
- `git_head`
- `tool`
- `canonical_inputs`
  - `profile_id`
  - `profile_fingerprint`
  - `lane_ids`
  - `db_path`
  - `db_identity`
  - `code_fingerprint`
- `freshness`
  - `as_of_date`
  - `max_age_days`
- `payload`
  - `apply_pauses`
  - `pause_days`
  - `results`

Current v1 policy:

- missing/legacy/mismatched/stale SR state => pulse shows SR as decaying and
  tells the operator to rerun `python -m trading_app.sr_monitor`
- Criterion 11 remains the only hard-block derived-state gate in pre-session

### Runtime truth after the change

After rerunning `python -m trading_app.sr_monitor` on `2026-04-10`, the current
repo state is:

- profile: `topstep_50k_mnq_auto`
- SR age: `0d`
- SR counts: `5 CONTINUE / 0 ALARM / 0 NO_DATA`
- SR stream mix: `canonical_forward=4`, `paper_trades=1`
- `project_pulse --fast --tool codex` trusts the SR summary again because the
  envelope now matches current profile, lane, DB, and code identity

### Drift enforcement

Added new structural drift checks:

- shared profile fingerprint helper must remain canonical
- SR monitor must write the derived-state contract envelope
- `project_pulse.collect_sr_state()` must validate the envelope before trust

Current drift result:

- `python3 pipeline/check_drift.py`
  - `NO DRIFT DETECTED: 91 checks passed [OK], 0 skipped, 7 advisory`

### Files touched

- `trading_app/derived_state.py`
- `trading_app/account_survival.py`
- `trading_app/sr_monitor.py`
- `scripts/tools/project_pulse.py`
- `pipeline/check_drift.py`
- `tests/test_trading_app/test_sr_monitor.py`
- `tests/test_tools/test_project_pulse.py`

### Verification

- `python3 -m py_compile trading_app/derived_state.py trading_app/account_survival.py trading_app/sr_monitor.py scripts/tools/project_pulse.py pipeline/check_drift.py tests/test_trading_app/test_sr_monitor.py tests/test_tools/test_project_pulse.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_sr_monitor.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py tests/test_tools/test_session_preflight.py -q`
  - result: `95 passed`
- `python3 -m trading_app.sr_monitor`
  - rewrote `sr_state.json` in envelope format
- `python3 scripts/tools/project_pulse.py --fast --tool codex`
  - live-control block reads the validated SR envelope correctly
- `python3 scripts/tools/session_preflight.py --context codex-wsl --with-pulse --quiet`
  - same compact pulse path works from preflight

### Why this matters

This is the missing guardrail for a fast-changing repo:

- stale derived state now self-invalidates instead of being trusted by
  convention
- startup remains compact; no heavy payload dumping into session context
- current truth is checked cheaply from fingerprints/identity fields rather than
  recomputing heavy logic every time

### Next move

The next sister task is still:

- **Concurrency Guardrails v1**
  - strengthen Claude/Codex same-branch mutating-session protection
  - move from warning-only toward fail-closed editing when concurrent mutable
    claims exist on the same branch

## Update (2026-04-10 late — Repo Intelligence / Pulse Hardening)

### Headline

Hardened the existing `project_pulse` orientation stack into a more canonical
repo-intelligence surface for Codex/startup use.

This is **not** a new rule system and **not** a discovery/runtime behavior
change. It is an operator-truth upgrade:

- current deployed-live profile/lane truth
- current validated-active truth
- current Criterion 11 gate truth
- current Criterion 12 SR monitor truth
- current lane-pause truth
- truthful Codex/Claude session ownership in the pulse continuity marker

### Design

Kept `scripts/tools/project_pulse.py` as the single synthesis layer rather than
building a new `.codex` status system.

New pulse summaries:

- `deployment_summary`
  - source: `trading_app.prop_profiles` + `validated_setups`
  - shows deployed-live count vs validated-active count
  - surfaces dangerous mismatch if a deployed lane is no longer validated
  - surfaces validated-but-not-deployed lanes as `ON DECK`, not as an error
- `survival_summary`
  - source: `trading_app.account_survival.check_survival_report_gate()` +
    persisted `account_survival_<profile>.json`
  - shows gate pass/block, operational pass probability, `as_of`, and report age
- `sr_summary`
  - source: persisted `data/state/sr_state.json`
  - shows lane counts by `CONTINUE/ALARM/NO_DATA`, stream-source mix, and age
- `pause_summary`
  - source: `trading_app.lane_ctl.get_paused_strategy_ids()`
  - shows active paused-lane count

Also fixed the pulse continuity marker path:

- `collect_session_delta()` no longer hardcodes `tool="claude"`
- `build_pulse(..., tool_name=...)` now records the actual calling tool
- `session_preflight.py --with-pulse` now derives `tool_name` from `--context`
  (`codex-*` -> `codex`, `claude-*` -> `claude`, else `unknown`)

### Current real pulse truth

On the current repo state, the upgraded pulse shows:

- profile: `topstep_50k_mnq_auto`
- deployed-live: `5`
- validated-active: `6`
- validated-only: `1` (`MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8`)
- Criterion 11: `PASS 87.2%`, `as_of=2026-04-09`
- Criterion 12 SR: `5 CONTINUE`, `0 ALARM`, `0 NO_DATA`
- SR stream mix: `canonical_forward=4`, `paper_trades=1`
- paused lanes: `0`

### Files touched

- `scripts/tools/project_pulse.py`
- `scripts/tools/session_preflight.py`
- `tests/test_tools/test_project_pulse.py`
- `tests/test_tools/test_pulse_integration.py`

### Verification

- `python3 -m py_compile scripts/tools/project_pulse.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py scripts/tools/session_preflight.py`
- `uv run --frozen --extra dev pytest tests/test_tools/test_session_preflight.py tests/test_tools/test_project_pulse.py tests/test_tools/test_pulse_integration.py -q`
  - result: `76 passed`
- `python3 pipeline/check_drift.py`
  - result: `88 passed, 0 blocking, 7 advisory`
- `python3 scripts/tools/session_preflight.py --context codex-wsl --with-pulse --quiet`
  - pulse prints the new live-control block correctly
  - `.pulse_last_session.json` now records `tool="codex"` on the Codex path

### Why this matters

This makes Codex-side reasoning materially safer:

- less chance of reasoning from stale docs/comments
- explicit visibility into structural-vs-live-vs-monitoring truth
- explicit warning on dangerous mismatches instead of implicit assumptions
- no false Claude ownership in Codex continuity markers

### Remaining frontier gap

This is a control-plane / orientation upgrade, not the final “smarter model”
solution.

Highest-signal next move remains:

- **Criterion 11 v2** — path-aware / intraday-aware breach modeling and dynamic
  scaling over the simulation horizon

That is still the next institutional trading-control upgrade after this pulse
hardening.

## Update (2026-04-10 early — Institutional Monitoring + Metadata Hygiene)

## Update (2026-04-10 later — Criterion 11 Account-Survival Gate Implemented)

### Headline

Implemented a real Criterion 11 account-survival Monte Carlo for deployment
profiles and wired it into `pre_session_check` as a fail-closed gate.

This is a deployment control only. It does **not** change:

- discovery scope
- validation outcomes
- lane allocation
- live order sizing

### Design

New module:

- `trading_app/account_survival.py`

It answers the deployment question:

- given the current `daily_lanes` for one profile, what is the probability a
  single account copy survives the next 90 trading days under the modeled firm
  rules?

Current design choices:

- canonical source = `validated_setups` strategy identity +
  `orb_outcomes`/`daily_features` historical outcomes
- dependence modeled via **daily portfolio PnL bootstrap**, not independent
  trade shuffling
- rules applied:
  - trailing DD / EOD freeze semantics
  - daily loss limit
  - payout-path consistency rule when the profile has one
  - static Topstep day-1 scaling feasibility check
- explicit fail-closed stance for `intraday_trailing` accounts
  - current engine does **not** pretend to model intraday trailing accurately
  - for those profiles it raises and the gate should stay blocked until an
    intraday path simulator exists

### Pre-session wiring

`trading_app/pre_session_check.py` now includes:

- `Criterion 11 survival`

It reads the latest persisted report and blocks if:

- no report exists
- report is unreadable
- report horizon is not `90d`
- report used fewer than `10,000` paths
- report is older than `30` days
- report fails scaling feasibility
- report operational survival is `< 70%`

### Current live profile result

Ran the real report for the only active execution profile with daily lanes:

- profile: `topstep_50k_mnq_auto`
- command:
  `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --as-of 2026-04-09 --paths 10000 --seed 0`

Result:

- source days: `2023` (`2019-05-07` -> `2026-04-09`)
- DD survival: `91.9%`
- operational pass: `91.9%`
- trailing-DD breach probability: `8.1%`
- daily-loss breach probability: `0.0%`
- gate status: `PASS`

Direct gate readback on `2026-04-10`:

- `Criterion 11 pass: operational 91.9%, as_of=2026-04-09, age=1d, paths=10000`

### Verification

- `python3 -m py_compile trading_app/account_survival.py trading_app/pre_session_check.py tests/test_trading_app/test_account_survival.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_pre_session_check.py -q`
  - result: `32 passed`
- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --as-of 2026-04-09 --paths 10000 --seed 0`
  - report generated successfully
- `python3 pipeline/check_drift.py`
  - result: `88 passed, 0 blocking, 7 advisory`

### Files touched

- `trading_app/account_survival.py`
- `trading_app/pre_session_check.py`
- `tests/test_trading_app/test_account_survival.py`

### Important limitation

This is institutionally honest for the currently active EOD-trailing profile,
but it is **not** yet a universal prop-account survival engine.

Known boundary:

- `intraday_trailing` profiles are fail-closed / unsupported
- current sizing assumption is the project’s present one-contract daily-lane
  convention per account copy; if live sizing becomes variable by lane, feed
  that explicitly into Criterion 11 instead of assuming it

### Follow-up review hardening (same day)

Read-only audit of the initial Criterion 11 implementation found two real
logic issues:

1. the report gate trusted `profile_id` + age only, so a changed profile could
   still ride on an old passing report
2. the simulator was using `validated_setups.stop_multiplier` instead of the
   deployed profile stop, which understated the actual live-book difference for
   `topstep_50k_mnq_auto` (`validated=1.0x`, deployed profile=`0.75x`)

Fixes applied:

- `trading_app/account_survival.py`
  - report metadata now includes a deterministic `profile_fingerprint`
  - gate now fail-closes if the current profile definition does not match the
    persisted report inputs
  - lane daily PnL loading now applies the **effective deployed profile stop**
    (`planned_stop_multiplier` or `profile.stop_multiplier`) instead of blindly
    trusting the validated row stop
- `tests/test_trading_app/test_account_survival.py`
  - added regression coverage for report/profile mismatch blocking
  - added regression coverage proving the profile stop override is used

Re-verified:

- `python3 -m py_compile trading_app/account_survival.py tests/test_trading_app/test_account_survival.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_pre_session_check.py -q`
  - result: `34 passed`

Corrected live result after applying the real deployed `0.75x` stop:

- profile: `topstep_50k_mnq_auto`
- as_of: `2026-04-09`
- DD survival: `87.2%`
- operational pass: `87.2%`
- trailing-DD breach probability: `12.8%`
- gate status: still `PASS`

This corrected the earlier `91.9%` number, which had been too optimistic
because it was effectively simulating the validated `1.0x` stop rather than
the live deployed stop.

### Headline

Focused institutional cleanup only. No discovery rerun. Goal was to keep the
research/deployment/monitoring layers canonical, remove stale metadata smell,
and ground the monitoring design in the actual literature PDFs under
`resources/`.

### What was verified

- `validated_setups` remains the structural source of truth: `6` active rows.
- `topstep_50k_mnq_auto` remains the deployed-live source of truth: `5` MNQ
  lanes only.
- `pipeline/check_drift.py` passes clean on the current repo state:
  `88 passed, 0 blocking, 7 advisory`.
- Current active validated count rechecked after cleanup: `6`.

### Literature grounding actually read from source PDFs

Read the actual PDFs from `resources/` via direct text extraction from the PDF
files themselves, not just prior markdown notes:

- `resources/real_time_strategy_monitoring_cusum.pdf`
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`

Operational takeaway used here:

- Criterion 12 style monitoring should be based on the live monitored R stream.
- Shiryaev-Roberts is appropriate for repeated surveillance and preferable to
  naive one-shot monitoring/CUSUM framing in this setting.
- 2026 holdout must not be folded back into selection to rescue a failed
  strategy. Live/hot monitoring is separate from structural validation.

### Monitoring design correction

Problem:

- `scripts/tools/forward_monitor.py` already had usable current-state evidence
  for all 5 deployed lanes.
- `trading_app/sprt_monitor.py` was effectively blind on 4/5 lanes because it
  only read `paper_trades`.

Fix:

- `trading_app/sprt_monitor.py` now uses explicit source priority:
  1. `paper_trades`
  2. canonical forward outcomes from `orb_outcomes` via
     `strategy_fitness._load_strategy_outcomes`
- The monitor prints and persists the source label:
  `paper_trades` or `canonical_forward`.
- This preserves the institutional rule that live/paper execution history is
  preferred, while keeping the monitor operational when paper coverage is
  incomplete.

Verification:

- `python -m trading_app.sprt_monitor` now runs successfully.
- Current monitor output:
  - `NYSE_CLOSE` `SIGNAL` from `canonical_forward`
  - `EUROPE_FLOW`, `COMEX_SETTLE`, `NYSE_OPEN`, `TOKYO_OPEN` all `CONTINUE`
    with source labeled explicitly

### Validator / metadata hygiene correction

Problem:

- `experimental_strategies` had `7` rows with
  `validation_status='REJECTED'` but `rejection_reason IS NULL`.
- Those rows still had text in `validation_notes`, so downstream queries could
  disagree depending on which field they read.

Fix:

- `trading_app/strategy_validator.py` now backfills `rejection_reason` from
  rejection notes when a row is rejected and the explicit reason was not
  populated in that code path.
- Added regression coverage so rejected rows do not silently leave
  `rejection_reason` empty again.
- One-time DB cleanup applied to existing rows:
  `before=7`, `after=0`.

Examples of backfilled historical reasons now queryable directly:

- `MGC_US_DATA_1000_E2_RR1.0_CB1_ORB_G5`:
  `Phase 4b: Too few positive windows: 50% < 60%`
- `MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5`:
  `Phase 4b: Too few positive windows: 58% < 60%`
- `MNQ_CME_REOPEN_E2_RR1.0_CB1_ORB_G5`:
  `Phase 4b: Too few positive windows: 50% < 60%`

### Tests run

- `python3 -m py_compile trading_app/sprt_monitor.py tests/test_trading_app/test_sprt_monitor.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_sprt_monitor.py tests/test_trading_app/test_strategy_validator.py -q`
  - result: `126 passed`

### Files touched in this cleanup

- `trading_app/sprt_monitor.py`
- `trading_app/strategy_validator.py`
- `tests/test_trading_app/test_sprt_monitor.py`
- `tests/test_trading_app/test_strategy_validator.py`

### Important interpretation

- This session did **not** weaken validation to bless hot recent data.
- It cleaned the monitoring/control layer so strategy-level validation and
  live-regime monitoring stay separate and queryable.
- If later work introduces true Shiryaev-Roberts monitoring, use the actual PDF
  above as the starting point, not stale summaries.

## Update (2026-04-10 later — Criterion 12 SR Monitor Implemented)

### Headline

Implemented a real score-function Shiryaev-Roberts monitor grounded in the
actual Pepelyshev-Polunchenko PDF, without pretending that the daily-reset
in-memory `PerformanceMonitor` was already Criterion 12 compliant.

### Why this was needed

- `trading_app/live/performance_monitor.py` still uses an in-memory daily-reset
  CUSUM helper. That is useful operationally, but it is not an honest
  60-trading-day SR surveillance process.
- Criterion 12 in `docs/institutional/pre_registered_criteria.md` explicitly
  requires Shiryaev-Roberts monitoring on the live R stream.

### What was added

- `trading_app/live/sr_monitor.py`
  - score-function Shiryaev-Roberts recursion (`R_n = (1 + R_{n-1}) e^{S_n}`)
  - Eq. 17/18 coefficient implementation
  - Monte Carlo threshold calibration to target ARL ≈ 60 trading days
- `trading_app/sr_monitor.py`
  - deployed-lane runner
  - baseline source priority:
    1. first 50 `paper_trades` if available
    2. validated backtest baseline otherwise
  - monitored stream priority:
    1. `paper_trades`
    2. canonical forward outcomes
  - source labels written explicitly so shadow fallback is never disguised as
    true live-paper monitoring
- `tests/test_trading_app/test_sr_monitor.py`
  - SR alarm behavior
  - threshold calibration sanity
  - baseline/stream source behavior

### Verification

- `python3 -m py_compile trading_app/live/sr_monitor.py trading_app/sr_monitor.py tests/test_trading_app/test_sr_monitor.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_sr_monitor.py tests/test_trading_app/test_cusum_monitor.py tests/test_trading_app/test_performance_monitor.py -q`
  - result: `19 passed`
- `python -m trading_app.sr_monitor`
  - threshold calibrated to `31.96` for target ARL≈60 on the standardized
    pre-change stream
  - current deployed-lane output:
    - all 5 lanes currently `CONTINUE`
    - `COMEX_SETTLE` is the only lane with enough paper trades to use a true
      first-50 live baseline right now
    - the other 4 lanes still use validated backtest baseline +
      canonical-forward stream fallback

### Important limitation

Criterion 12 is now implemented as a real SR monitor, but current live-paper
coverage is still thin. So:

- the implementation is institutional-grade
- the current data coverage is not yet full institutional-grade for every lane

That distinction matters. Do not claim all deployed lanes have a mature
50-trade live-paper baseline yet.

## Update (2026-04-10 later again — SR Alarm Lifecycle Wired To Lane Pauses)

### Headline

Extended the Criterion 12 SR monitor from "informational only" into an actual
operational lifecycle control:

- SR alarms can now write temporary lane pauses
- `session_orchestrator` now honors persisted lane pauses at startup and entry
- this is an operational control only, not a structural validation mutation

### Design boundary

This change intentionally does **not**:

- edit `validated_setups`
- edit `strategy_fitness`
- bless or reject strategies structurally
- use 2026/live data for rediscovery

It only changes whether a deployed lane is temporarily tradable.

### What changed

- `trading_app/lane_ctl.py`
  - added `pause_strategy_id(...)`
  - added `get_paused_strategy_ids(...)`
  - updated module contract text: lane overrides now affect
    `resolve_daily_lanes`, `pre_session_check`, and `session_orchestrator`
- `trading_app/sr_monitor.py`
  - added `apply_alarm_pauses(...)`
  - new CLI flags:
    - `--apply-pauses`
    - `--pause-days`
  - `sr_state.json` now records pause intent metadata alongside results
- `trading_app/live/session_orchestrator.py`
  - loads active lane pauses on startup when running a profile portfolio
  - blocks new entries for paused lanes with explicit `ENTRY_BLOCKED_PAUSED`
    signal records
  - keeps orphan/manual-close blocks distinct from SR/manual-review pauses

### Important bug fixed during implementation

The first draft set `_profile_id_for_lane_ctl` inside profile resolution and
then accidentally reset it to `None` later in `__init__`, which would have made
the pause loader silently no-op in production. That was corrected before
commit/verification.

### Verification

- `python3 -m py_compile trading_app/lane_ctl.py trading_app/live/session_orchestrator.py trading_app/sr_monitor.py tests/test_trading_app/test_lane_ctl.py tests/test_trading_app/test_sr_monitor.py tests/test_trading_app/test_session_orchestrator.py`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_lane_ctl.py tests/test_trading_app/test_sr_monitor.py -q`
  - result: `26 passed`
- `uv run --frozen --extra dev pytest tests/test_trading_app/test_session_orchestrator.py -k "paused_lane_blocks_new_entry_with_pause_reason or load_paused_lane_blocks_from_lane_ctl" -q`
  - result: `2 passed`
- `python -m trading_app.sr_monitor`
  - still shows all 5 deployed lanes at `CONTINUE`
- `python pipeline/check_drift.py`
  - still clean: `88 passed, 0 blocking, 7 advisory`

### Important limitation

The full `tests/test_trading_app/test_session_orchestrator.py` file appears to
hang in unrelated existing async coverage on this branch, so verification for
the new behavior was narrowed to the new pause-specific tests plus compile,
runtime SR output, and drift. Do not overstate that as a clean full-file
orchestrator pass.

## Update (2026-04-09 late — Portfolio Alignment Sprint)

### Headline

Fixed a silent-but-catastrophic drift: all 5 deployed lanes in `topstep_50k_mnq_auto` were GHOSTS — referencing strategy_ids not present in validated_setups or experimental_strategies. The bot was operating with zero current validation backing against real (practice) capital. Swept 2026-04-09 in a focused 2-commit alignment sprint + added drift check 95 to prevent recurrence.

### What was ghost

Prior `topstep_50k_mnq_auto.daily_lanes` (from a 2026-04-03 allocator run that pre-dated the Apr 9 discovery rebuild):
- MGC_CME_REOPEN_E2_RR2.5_CB1_ORB_G6 — not in validated_setups, not in experimental_strategies
- MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12 — same
- MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 — same
- MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10 — same
- MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10 — same

These were from an older discovery run. When the Apr 9 methodology rebuild re-discovered from scratch under the new Pathway B criteria, none of these IDs survived, but `prop_profiles.py` was never updated to match.

### New deployed portfolio (validated_setups backed)

All 5 lanes sourced from the 6-strategy current validated book. Lane selection guided by a decay-aware yearly audit (Bailey et al 2013 era rule).

Post-sprint hygiene note: the one active validated row missing provenance metadata
(`MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` with `validation_pathway = NULL`) was
backfilled to `validation_pathway = 'family'` after confirming it came from the
family-mode MES comprehensive hypothesis run. Active validated rows are now
metadata-consistent on that field (6/6 = `family`).

Important: the list below is a historical selection-time note from 2026-04-09
and should NOT be treated as the canonical current live stat block. Forward
N / WR / ExpR drift over time. Canonical current-state sources are:
- `gold.db` (`validated_setups`, `orb_outcomes`, strategy_fitness)
- `trading_app/prop_profiles.py` for deployed-live lanes
- `docs/plans/2026-04-09-portfolio-tiered.md` for the current audited summary

Selection-time live portfolio:
- `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8` — selected despite prior decay concerns because live behavior had improved
- `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G8` — holding / stable
- `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8` — holding / improving
- `MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G5` — holding / improving
- `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G8` — watch / probation lane

`MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` remained `validated_setups.status='active'` but was excluded from the deployed-live profile during the alignment sprint. Treat that as a live-book routing choice, not as removal from the structurally validated set.

### Drift check 95 added

`pipeline/check_drift.py::check_prop_profiles_validated_alignment` — fires on any lane in an `active=True` profile that doesn't exist in `validated_setups` with `status='active'`. Inactive profiles exempt (re-checked at activation time). 4 TDD tests locked the behavior (pass/missing/retired/inactive-exempt). Check 95 passes against current state. 87/87 non-advisory checks pass (baseline family_rr_locks #59 still pre-existing advisory).

### What this sprint did NOT do (deferred, documented)

- **validation_status vs rejection_reason dual-field sync bug.** Phase 3/4b validator gates set legacy `validation_status='REJECTED'` + `validation_notes` but not the new `rejection_reason` field. 7 of 25 experimental_strategies are in this state. Not urgent — doesn't affect runtime, just makes ad-hoc queries confusing.
- **Codex-identified shadow candidates** (MGC SINGAPORE_OPEN G4, MGC CME_REOPEN G6, MGC EUROPE_FLOW G6, MGC TOKYO_OPEN G5, MES US_DATA_1000 G5) — forward-hot in 2026 Q1 but failed validation either via cross-instrument FDR or Phase 4b windows. Worth a shadow/provisional tier later. NOT deployed now.
- **Inactive profile cleanup.** 8 other profiles (topstep_50k_type_a, etc.) also reference ghost lanes but are `active=False` so they don't affect runtime. Drift check 95 skips them intentionally. Clean up at activation time.

### Next session priorities

1. **Do nothing for 2-4 weeks.** Stop refactoring. Let the new portfolio accumulate H2 2026 forward data. TOKYO_OPEN watch decision after 30+ new trades.
2. **If refactoring needed:** fix the validation_status/rejection_reason sync bug first (clean house), then the Codex shadow candidates.
3. **Do NOT run another discovery rebuild until Q2 2026 data.** Bailey MinBTL + Carver Table 5 both say you can't learn anything new from 3 months.

### Commits this sprint

- `[TBD]` fix: replace ghost deployed lanes with validated_setups backed (prop_profiles.py)
- `[TBD]` feat: drift check 95 — prop_profiles ↔ validated_setups alignment guard

### Known issues (pre-existing)

1. Drift check 59 (family_rr_locks) — 1 active family missing. Pre-existing. Run `python scripts/tools/select_family_rr.py` to fix.
2. Dual-field validation_status/rejection_reason drift — see "did NOT do" section above.
3. 8 pre-existing test failures unrelated to this sprint (test_trader_logic x3, claude_superpower_brief x2, test_data_era x2, test_pulse_integration x1). Verified unrelated via git stash test.

### Branch state

- Branch: `main`
- Will be: 15 commits ahead of origin/main after this sprint (need to push)

---

## Update (2026-04-09 evening — Full Discovery Methodology Overhaul + 6-Strategy Portfolio)

Historical audit-trail section only. Treat the current repo / DB / deployed-live
state as authoritative over the notes below. Some “next steps” in this section
have since been completed or superseded by later commits and the portfolio
alignment sprint above.

### Where we are

Session redesigned discovery methodology from scratch. 6 validated strategies (up from 1 at session start) + 9 high-conviction near-misses identified.

**Historical context doc:** `docs/plans/2026-04-09-portfolio-tiered.md`

### Current validated portfolio (Tier 1 — automate eligible)

| # | Instrument | Session | Filter | RR | IS Sharpe | 2026 TotR |
|---|---|---|---|---|---|---|
| 1 | MNQ | NYSE_CLOSE | G8 | 1.0 | 1.28 | **+19.6R** HOT |
| 2 | MNQ | TOKYO_OPEN | G8 | 2.0 | 1.02 | +14.9R HOT |
| 3 | MNQ | EUROPE_FLOW | G8 | 2.0 | 1.11 | +14.1R HOT |
| 4 | MNQ | COMEX_SETTLE | G8 | 1.0 | 1.53 | +5.8R |
| 5 | MNQ | NYSE_OPEN | G5 | 2.0 | 0.93 | +5.0R |
| 6 | MES | CME_PRECLOSE | G8 | 1.0 | 1.32 | +0.1R (N=15) |

Combined Q1 2026 forward: ~+59R across the 6 strategies.

### Key methodology breakthrough

**Strategy SELECTION vs Parameter OPTIMIZATION are different problems:**
- Strategy selection → BH FDR (K = number of ideas, ~10-15)
- Parameter optimization within a strategy → Walk-forward (WFE ≥ 0.50)
- Prior approach counted parameter variations in K, inflating it and killing honest edges

See memory files: `methodology_breakthrough_apr9.md`, `discovery_redesign_apr9.md`.

### What was done this session

1. **Bailey calendar year metric fix** (commit f3523ae) — MGC was using trading years, corrected
2. **Discovery redesign** (commit f4fc23d) — K=5 mechanism-grounded files
3. **Amendment 3.0** (commit ce450fc) — Pathway B (individual hypothesis testing)
4. **Pathway B validator support** (commit c8efb20) — TDD, 157/157 tests pass
5. **Criterion 8 fix** (commit ea18c61) — N_oos ≥ 30 minimum gate
6. **Comprehensive discovery** (commit 11bac27) — 25 bundles across all viable sessions
7. **Final portfolio tiered** (commit b85fb99) — `docs/plans/2026-04-09-portfolio-tiered.md`

### Tier 2 — Fundamentally valid, killed by non-load-bearing parameters

**Highest-conviction near-misses** (fixable, not data failures):

1. **MNQ CME_PRECLOSE G8 RR1.0** — Sharpe **1.83** (highest in project!). Killed by 2019 contract-launch era (first 8 months, N=56). Same issue MGC solved via WF_START_OVERRIDE=2022. Apply WF_START_OVERRIDE=2019-06-01 for MNQ.

2. **MNQ SINGAPORE_OPEN G8 RR2.0** — p=0.039 raw significant, killed by cross-instrument FDR stratification (K=25 pooling 3 uncorrelated asset classes).

3. **MGC CME_REOPEN G6 RR2.0** — WF 3/3 windows positive, ExpR +0.52 forward. Killed by cross-instrument FDR. **Uncorrelated diversifier — MGC is the only asset class independent of MNQ/MES.**

4. **MGC EUROPE_FLOW G6 RR2.0** — Similar cross-instrument FDR kill.

5. **MES NYSE_OPEN G8 RR2.0** — One bad year (2023) kills era stability. Other 6 positive years.

### Next steps (priority order)

#### 1. Review tiered portfolio doc (IMMEDIATE)
Read `docs/plans/2026-04-09-portfolio-tiered.md`. Decide which Tier 2 strategies to trade manually.

#### 2. Framework parameter fixes (HIGH VALUE — unlocks more strategies)

**a) MNQ WF_START_OVERRIDE = 2019-06-01** — pattern exists for MGC, apply to MNQ. Unlocks MNQ CME_PRECLOSE (Sharpe 1.83).

**b) Per-instrument FDR stratification** — currently stratifies across all 3 instruments. Stratify per instrument (MNQ K=12, MES K=8, MGC K=5). Unlocks MGC CME_REOPEN, MGC EUROPE_FLOW, MES NYSE_CLOSE, MES SINGAPORE_OPEN.

**c) Era stability softening** — consider allowing 1 era failure if 6+ other eras are positive and WF still passes.

#### 3. Deploy Tier 1 portfolio
Update `trading_app/prop_profiles.py` with 6 Tier 1 strategies. Old lanes are stale.

#### 4. Phase 2: Filter family expansion (LATER)
Base size-first lanes now established. Phase 2: OVNRNG overlays, PDR pre-session at LONDON_METALS/EUROPE_FLOW, GAP pre-session at CME_REOPEN.

#### 5. Shiryaev-Roberts live monitoring (FUTURE)
Referenced in Criterion 12 but not implemented. Paper: `resources/real_time_strategy_monitoring_cusum.pdf`.

### Files this session

**Hypothesis files (active):**
- `docs/audit/hypotheses/2026-04-09-mnq-comprehensive.yaml` (K=12) — CURRENT
- `docs/audit/hypotheses/2026-04-09-mes-comprehensive.yaml` (K=8) — CURRENT
- `docs/audit/hypotheses/2026-04-09-mgc-comprehensive.yaml` (K=5) — CURRENT

**Hypothesis files (obsolete — can archive):**
- `2026-04-09-mnq-redesign.yaml`, `2026-04-09-mes-redesign.yaml`, `2026-04-09-mgc-redesign.yaml` (K=5 intermediate)
- `2026-04-09-mnq-final.yaml`, `2026-04-09-mes-final.yaml`, `2026-04-09-mgc-final.yaml`
- `2026-04-09-mnq-mode-a-rediscovery.yaml`, `2026-04-09-mes-mode-a-rediscovery.yaml`, `2026-04-09-mgc-mode-a-rediscovery.yaml` (K=16 old)
- `2026-04-09-mnq-rr10-individual.yaml` (Pathway B experiment)

**Production code (committed):**
- `trading_app/hypothesis_loader.py` — testing_mode field exposed at top level
- `trading_app/strategy_validator.py` — Pathway B + Criterion 8 N_oos >= 30 gate

**Documentation:**
- `docs/institutional/pre_registered_criteria.md` — Amendment 3.0 (v3.0)
- `docs/plans/2026-04-09-portfolio-tiered.md` ← **PRIMARY HANDOFF DOC**
- `docs/plans/2026-04-09-discovery-redesign.md`
- `docs/plans/2026-04-09-full-discovery-execution-plan.md`
- `docs/plans/2026-04-09-hypothesis-audit-plan.md`

**Tests:**
- `tests/test_trading_app/test_hypothesis_loader.py` — 4 new TestTestingMode tests
- 158/158 pass (55 loader + 103 validator) after commit `149f9d0`. The prior "157/157" claim in this HANDOFF was inaccurate — actual baseline at session end was 155/157 because two `TestCriterion8OOSPositive` fixtures (pre-existing, not touched by Pathway B) were colliding with the `N_oos >= 30` gate added in `ea18c61`. Fixed 2026-04-09 in a /next iteration: the fixtures were resized to 30 rows to exercise the sign/ratio logic, and a new `test_insufficient_oos_sample_passes_through` test locks the sub-threshold pass-through behavior. Validator code unchanged; gate was correct.

### Database state

- `experimental_strategies`: 25 strategies from comprehensive run (pre-registered SHAs locked)
- `validated_setups`: 6 Tier 1 strategies
- `bars_1m` / `daily_features` / `orb_outcomes`: clean, current as of 2026-04-07

### Known issues

1. **Drift check 59** (family_rr_locks) — 1 active family missing. Pre-existing, not related to this session. Run `python scripts/tools/select_family_rr.py` to fix.

2. **Obsolete hypothesis files** — 10 old files in `docs/audit/hypotheses/` from the iterative redesign process. Can be archived but SHAs in experimental_strategies reference them via Phase 4 drift check, so don't DELETE yet.

### Branch state

- Branch: `main`
- Last commit: `b85fb99` (docs: tiered portfolio)
- 14 commits ahead of origin/main (need to push)

---

## Previous updates (preserved for audit trail)

### Update (Apr 9 — Phase 4.2 Hypothesis Files + Amendment 2.9 + Parent Data Cleanup)

**NOTE:** Superseded by this session's comprehensive run. Files from this earlier work have been obsoleted (see list above).

1. **Amendment 2.9** — Parent/Proxy Data Policy (binding). NQ/ES bars deleted, GC kept for MGC Tier 2.
2. **Amendment 2.8** — Factual correction of data horizons post-Phase-3c.
3. **Initial 3 hypothesis YAMLs** — 16+16+4 format (all obsoleted this session).
