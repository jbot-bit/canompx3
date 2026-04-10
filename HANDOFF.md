# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

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
