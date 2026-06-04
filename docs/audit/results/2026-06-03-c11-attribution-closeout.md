# C11 and Attribution Closeout - topstep_50k_mnq_auto

Date: 2026-06-03
Tool: Codex
Mode: evidence-first, no live trading, no allocation/live_config/profile writes, no schema edits

## Scope

The question: is `topstep_50k_mnq_auto` deployable now, and is the prior night's
attribution work closed out? Covers C11 current truth, execution attribution state,
live-readiness, and a model-parity audit of the live bracket-risk path. Out of scope:
any profile/allocation/live_config write, stopping the live `--signal-only` process,
and locking a remediation cap value.

## Verdict

MEASURED: `topstep_50k_mnq_auto` remains NO-GO for deployment.

MEASURED blockers:
- C11 does not clear. Current strict no-write run exits 2 with `strict max observed 90d DD = $2,039` versus strict budget `$1,600`.
- Execution attribution is not cleared. `paper_trade_logger --sync --dry-run` is runnable and finds no backfillable trades, but `project_pulse --fast --format json` still flags stale `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` attribution and cannot read `live_journal.db`.
- `live_journal.db` is locked by an active `--signal-only` Topstep process. This is unsafe to stop under this pass.
- Live-readiness exits nonzero. Criterion 11 state is fail/fingerprint mismatch, Criterion 12 is valid, telemetry maturity is 9/30 trading days and advisory for this express/funded profile.
- Replay parity has one BLOCKING mismatch: live bracket submission appears to multiply already-tight `event.risk_points` by `strategy.stop_multiplier` again.

No deployability claim is supported.

## Why This Can Differ From Last Night

MEASURED: `HANDOFF.md` contains prior ready/near-ready claims for this profile, including older notes that C11/C12 and live-readiness were green with telemetry advisory.

MEASURED: current commands from this pass contradict deployability now:
- no-write C11 exits 2;
- strict live-readiness exits 1;
- `project_pulse` exits 1 with stale execution attribution and unreadable `live_journal.db`;
- the repo is dirty and behind `origin/main`;
- replay/live bracket stop-distance parity has a blocking mismatch.

INFERRED: last night may have been close under an older readiness state, older code fingerprint, or stale saved state, but that is not current evidence. Current deployability must follow the measured state above.

## Phase 0 - Repo and Process Safety

MEASURED:
- cwd: `C:\Users\joshd\canompx3`
- branch: `main`
- divergence: ahead 3, behind 10 versus `origin/main`
- `git status --short --branch`: dirty
- conflict markers: none found by exact marker scan over `HANDOFF.md docs scripts trading_app research tests .codex .agents`
- worktrees: 28 total from `git worktree list --porcelain`
- active peer/process context includes:
  - `C:\Users\joshd\canompx3-live-launch-tokyo\.venv\Scripts\python.exe -m trading_app.live.bot_dashboard`
  - `.venv\Scripts\python.exe scripts\run_live_session.py --profile topstep_50k_mnq_auto --signal-only`
  - repo MCP servers
  - a peer diagnostic command in `C:\Users\joshd\canompx3-live-trade-diag`

Dirty files observed:
- `HANDOFF.md`
- `docs/audit/results/2026-06-02-mnq-single-leg-account-fit-replacement-v1.csv`
- `docs/audit/results/2026-06-02-mnq-single-leg-account-fit-replacement-v1.md`
- `research/mnq_single_leg_account_fit_replacement_v1.py`
- `tests/test_research/test_mnq_single_leg_account_fit_replacement_v1.py`
- `tests/test_trading_app/test_account_survival.py`
- `trading_app/account_survival.py`
- untracked report/stage/plan artifacts listed by `git status`

DB holder truth:
- MEASURED: `gold.db` exists and accepted both read-only and write-handle probes during this pass.
- MEASURED: `live_journal.db` exists but read-only and write-handle probes failed with DuckDB `IOException`.
- MEASURED holder: PID chain `27152 -> 15088 -> 81448`, command `.venv\Scripts\python.exe scripts\run_live_session.py --profile topstep_50k_mnq_auto --signal-only`.
- Classification: active signal-only, unsafe to stop in this pass.

## Phase 1 - Attribution Lock Closeout

Command:
`python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`

MEASURED:
- exit code: 0
- profile lanes: 3
- dry-run results:
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`: 0/0 trades pass filter
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`: 0/2 trades pass filter
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08`: 0/0 trades pass filter
  - total trades: 0

Command:
`python scripts/tools/project_pulse.py --fast --format json`

MEASURED:
- exit code: 1
- interpretation: pulse found blockers; command did not silently succeed
- pulse counts: broken 3, decaying 4, ready 1, unactioned 25, paused 6
- live journal status: unreadable, locked by active signal-only process
- missing execution count: 0
- stale attribution ids: `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`
- stale lane paper/live evidence:
  - COMEX: paper 75, live 0, last 2026-05-28, age 6
  - US_DATA: paper 53, live 0, last 2026-05-19, age 15
  - TOKYO: paper 73, live 0, last 2026-05-28, age 6

Attribution status:
- MEASURED: gold/paper dry-run path is not currently blocked.
- MEASURED: no new paper trades would be synced by the dry-run.
- MEASURED: stale US_DATA attribution remains unresolved in `project_pulse`.
- MEASURED: live journal read is blocked by active signal-only lock.
- INFERRED: the exact unblock path is to wait for the signal-only session to finish, or obtain explicit operator authorization to stop it via the canonical stop path, then rerun `project_pulse --fast --format json`.

## Phase 2 - C11 Current Truth

Command:
`python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`

MEASURED:
- account_survival exit code: 2
- as_of: 2026-06-03
- horizon: 90d
- MC paths: 10,000
- source days: 2,048, 2019-05-31 to 2026-06-03
- operational pass: 99.7%
- DD survival: 99.7%
- daily-loss breach probability: 0.0%
- trailing-DD breach probability: 0.3%
- scaling breach probability: 0.0%
- consistency breach probability: 0.0%
- historical DLL breach days: 0
- max observed 90d DD: `$2,039`
- strict DD budget: `$1,600`
- prop-account path safety: FAIL
- final deployability gate: FAIL
- no-write marker: `ACCOUNT_SURVIVAL_NO_WRITE_EXIT=2`

MEASURED: C11 does not clear because strict observed 90d DD exceeds the 80% budget, despite MC operational probability being above the 70% gate.

Additional read-only scenario measurement:
- source days: 2,048
- active lane count distribution: 0 lanes 847 days, 1 lane 716 days, 2 lanes 354 days, 3 lanes 131 days
- multi-lane days: 485
- zero-trade days: 847
- worst historical daily total: 2022-05-02, `$-320.82`, min intraday delta `$-355.13`, 3 lanes
- max 90d DD window: `$2,038.84`, from 2022-07-27 to 2022-10-25

## Phase 3 - C11 Model-Parity Audit

| Surface | Classification | Evidence |
| --- | --- | --- |
| SessionOrchestrator ORB cap / `max_orb_size_pts` | NOT A MISMATCH | MEASURED: live loads allocator/profile caps into `_orb_caps` and skips when `event.risk_points >= cap`. Account survival applies `max_orb_size_pts` with the same inclusive skip after tight-stop adjustment. |
| stop multiplier replay vs engine event risk | NOT A MISMATCH | MEASURED: account survival applies `apply_tight_stop` for profile/lane effective stop multiplier; execution engine moves the stop before computing `risk_points`. |
| broker bracket stop multiplier | BLOCKING | MEASURED: execution engine emits `event.risk_points` after applying the stop multiplier; live bracket code then computes `stop_dist = risk_pts * mult`. For `0.75`, this implies a live broker bracket around `0.5625x` raw ORB distance while replay models `0.75x`, unless another layer offsets it. This must be reconciled before any C11 remediation result can be trusted. |
| daily breaker semantics | ADVISORY | MEASURED: runtime `RiskManager` halts on realized R cap or realized dollar breaker; account_survival models daily loss by conservative day-min delta and stops path after a daily-loss breach. Current strict C11 has 0 historical DLL breach days, so this is not the current failing mechanism. |
| one-trade/day assumption | UNSUPPORTED | MEASURED: replay scenarios aggregate all same-day lane trades; no one-trade/day throttle exists in the inspected C11 path. Whether a throttle helps C11 is unmeasured. |
| lane stacking assumption | ADVISORY | MEASURED: 485 historical days have more than one active lane. `max_open_lots` in current scenarios is 1, but same-day sequential stacking still contributes to close-to-close DD. Needs report-only throttle replay before action. |
| HWM / loss-limit handling | ADVISORY | MEASURED: live path has HWM tracker checks before entry; C11 uses EOD trailing DD and strict close-to-close observed DD. The broker-equity live HWM path is not identical to C11's bootstrap/strict historical diagnostic, but both are fail-closed surfaces. |
| Topstep profile thresholds | NOT A MISMATCH | MEASURED: repo profile uses Topstep 50K MLL `$2,000`, strict budget fraction 80% = `$1,600`, and internal `daily_loss_dollars=$450`. Current official Topstep help still shows 50K XFA MLL starts at `$-2,000` and TopstepX DLL can be optional/configured separately. |
| `--no-write-state` behavior | NOT A MISMATCH | MEASURED: CLI accepts `--no-write-state`, evaluates current C11, prints `ACCOUNT_SURVIVAL_NO_WRITE_EXIT=2`, and does not refresh state. This explains why live readiness still reports Criterion 11 state/fingerprint mismatch. |

Primary/official rule checks:
- Topstep Maximum Loss Limit help center, crawled current by web search on 2026-06-03: 50K account MLL `$2,000`; XFA MLL starts at `-$2,000` and locks at `$0`.
- Topstep Daily Loss Limit help center, crawled current by web search on 2026-06-03: TopstepX Trading Combine/XFA DLL is a risk feature that can be optional/configured; checkout DLL parameters list 50K `$1,000`.
- Local repo official scrape: `docs/research-input/topstep/topstep_mll_article.md` and `trading_app/prop_profiles.py` encode 50K MLL `$2,000`.
- Local literature context: `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` supports the risk-lead stance that prop-firm drawdown caps force low effective vol targets and should not be rescued by post-hoc sizing without fresh measurement.

## Live-Readiness Evidence

Command:
`python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`

MEASURED:
- exit code: 1
- active lanes: 3
- criterion11: `state=fail`, age 0, message `BLOCKED: Criterion 11 state code fingerprint mismatch. Re-run account survival.`
- criterion12: `state=valid`, latest verdict `valid`, age 1
- paused lanes: 845, all with explicit reasons in the returned summary
- stale_lanes in live-readiness report: 0
- strict blockers: `Criterion 11 gate not OK`
- telemetry maturity: 9/30 trading days, `UNVERIFIED_INSUFFICIENT_TELEMETRY`, advisory for express/funded profile
- git dirty: true

INFERRED: live-readiness and project_pulse disagree on "stale" vocabulary because `project_pulse` flags stale execution attribution while live-readiness `stale_lanes` is a lane-allocation/readiness surface. This is not evidence that attribution cleared.

## Phase 4 - Remediation Options

| Option | Evidence available | What must be measured next | Overfit/post-hoc risk | Implementation risk | Capital safety | Expected C11 impact | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Runtime ORB cap parity | MEASURED: current account_survival and live ORB_CAP_SKIP are aligned on inclusive cap skip. | Resolve bracket stop-distance mismatch first; then rerun no-write C11. | Low if limited to parity. | Medium due live order path. | Positive if true parity. | Current cap parity still fails C11 at `$2,039/$1,600`. | PARK |
| B. Stop multiplier reduction | MEASURED: current 0.75 fails; prior/current code supports stop multiplier mechanics. | Fresh prereg and replay after bracket parity fix; include expectancy, costs, DSR, era splits, 2026 holdout descriptive only. | High if chosen only to rescue C11. | Medium/high because live bracket geometry currently suspect. | Potentially positive but untrusted until parity fixed. | UNSUPPORTED for current canonical runtime; cannot claim. | PARK |
| C. Daily dollar breaker / flat remainder of day | MEASURED: runtime has dollar breaker; C11 has 0 historical DLL breach days. | Report-only replay of true realized intraday breaker against current lane sequence. | Medium. | Medium. | Positive for tail loss, but not enough evidence for current DD failure. | Likely limited because C11 fails on rolling DD, not DLL breaches. | PARK |
| D. One-loss-per-day throttle | MEASURED: not present in current C11; multi-lane days exist. | Report-only throttle replay with fixed prereg and no lane/window/RR tuning. | High. | Medium. | Positive if it reduces adverse same-day stacking. | UNSUPPORTED. | PARK |
| E. One-trade-per-day throttle | MEASURED: not present; 485 multi-lane days. | Report-only throttle replay plus opportunity cost and scheduling priority rule. | High. | Medium. | Positive for drawdown, negative for diversification/edge capture. | UNSUPPORTED. | PARK |
| F. Session scheduling / priority | MEASURED: current profile has 3 active lanes; same-day sequential stacking contributes to DD. | Predeclare priority rule before replay; no tuning to 2026. | High. | Medium. | Potentially positive if deterministic and conservative. | UNSUPPORTED. | PARK |
| G. Lane removal/replacement | MEASURED: replacement audit rows are dirty/uncommitted; current C11 still fails. | Fresh bounded prereg or finish existing replacement artifact truth; no promotion ranking while C11 blocked. | High. | Medium. | Unknown. | UNSUPPORTED. | PARK |
| H. Larger/different prop account | MEASURED: Topstep 100K/150K have larger MLLs in repo and official rules. | Separate account-fit run with official current rules, costs, activation/payout constraints, and C11 per account. | Medium. | Low/medium if report-only. | Could improve DD budget; does not fix attribution/parity. | Likely positive on strict DD budget only. | CONTINUE only after parity/attribution clear |
| I. Staged buffer before automation | MEASURED: XFA MLL trails from negative balance to lock; current strict DD needs `$2,039` observed vs `$1,600` internal budget. | Define buffer rule and replay path from buffer state; verify Topstep payout/MLL effects current. | Medium. | Low if report-only; high if operationalized prematurely. | Positive if it reduces breach risk before automation. | Potentially positive, but not a C11 pass today. | PARK |
| J. Accept current profile NO-GO | MEASURED: C11 fail, attribution blocked, live-readiness fail, dirty repo. | None for deployment; preserve blockers and stop promotion talk. | Low. | Low. | Highest immediate safety. | Honest current result. | CONTINUE |

## Phase 5 - Decision

Chosen next action: fix replay parity.

MEASURED rationale:
- The live broker bracket path appears to apply `stop_multiplier` to `event.risk_points` after the execution engine has already computed `risk_points` from the tight stop.
- This is a direct model/runtime mismatch, so any stop-sizing remediation, throttle replay, or account-fit conclusion would be contaminated until this is reconciled.

Decision constraints:
- Do not deploy.
- Do not refresh C11 state yet.
- Do not change allocation/live_config/profile/schema in this pass.
- Do not stop the active signal-only holder.
- Keep current profile blocked.

Exact next command family after an explicit code-fix task:
1. Add a focused failing test that proves bracket `stop_dist` equals execution-engine `event.risk_points` for a 0.75x strategy.
2. Patch only the bracket stop/target computation or the event contract so replay, engine, and broker bracket use one canonical risk-distance convention.
3. Rerun:
   - `python -m pytest tests/test_trading_app/test_account_survival.py tests/test_trading_app/test_session_orchestrator.py -q`
   - `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`
   - `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`
4. Only after signal-only finishes or operator authorizes stop, rerun:
   - `python scripts/tools/project_pulse.py --fast --format json`

## Rejected Paths

- Deployment: rejected; C11, attribution, live-readiness, and dirty repo do not clear.
- Allocation ranking/promotion: rejected; C11 failed.
- Stop multiplier rescue now: rejected; bracket parity mismatch makes this untrusted and post-hoc.
- Lane/session/RR tuning: rejected by mission rules.
- Stopping signal-only to read `live_journal.db`: rejected; holder is active signal-only and unsafe to stop without explicit operator authorization.
- Stateful C11 refresh: rejected in this pass; command was intentionally run with `--no-write-state`.

## Files Changed

MEASURED:
- Added this report: `docs/audit/results/2026-06-03-c11-attribution-closeout.md`

Pending:
- `HANDOFF.md` should be updated only with a narrow pointer to this report and the parity blocker if the final conflict-marker and diff checks remain clean.

## Tests and Commands Run

MEASURED commands:
- `git status --short --branch`
- `git rev-list --left-right --count origin/main...HEAD`
- `git log --oneline -10`
- exact conflict-marker scan
- `git worktree list --porcelain`
- process scan for live/session/dashboard/MCP holders
- DuckDB read/write-handle probes for `gold.db` and `live_journal.db`
- `python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run`
- `python scripts/tools/project_pulse.py --fast --format json`
- `python -m trading_app.account_survival --profile topstep_50k_mnq_auto --no-write-state`
- read-only scenario diagnostic using `trading_app.account_survival` helpers
- `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`
- targeted source inspections of `trading_app/account_survival.py`, `trading_app/live/session_orchestrator.py`, `trading_app/execution_engine.py`, `trading_app/risk_manager.py`, `trading_app/prop_profiles.py`, and `trading_app/config.py`

No live trading command was run. No allocation/live_config/profile/schema write was made.

## Limitations

What this audit does NOT establish:
- It does not prove the bracket-risk-parity mismatch is fully fixed — the fix (`9b3fc530`)
  was authored later and still owes an independent reviewer per the adversarial-audit gate.
- C11 was run with `--no-write-state`; the persisted survival/SR state was not refreshed in
  this pass, so live-readiness fingerprint mismatch is expected, not a separate defect.
- `live_journal.db` could not be read (locked by the active signal-only process), so
  attribution staleness is reported from `project_pulse` only, not from the journal directly.
- No remediation cap value is validated here; the sensitivity figures are emulation, and the
  drawdown-magnitude gate remains failing at the current 0.75 stop.

## Follow-Up Prompt

Operate as canompx3 Code Guardian + institutional quant/risk lead. Fix only the replay/live bracket stop-distance parity blocker for `topstep_50k_mnq_auto`: prove whether `ExecutionEngine` emits tight-stop `event.risk_points` and whether `SessionOrchestrator` broker bracket code double-applies `stop_multiplier`; add a focused failing test; patch the smallest canonical risk-distance convention; do not edit allocation/live_config/profile/schema; no live trading; then rerun focused tests, no-write C11, and strict live-readiness. Keep all claims labeled MEASURED/INFERRED/UNSUPPORTED and do not recommend deployment unless C11, attribution, readiness, dashboard/API state, and dirty repo risk all clear.
