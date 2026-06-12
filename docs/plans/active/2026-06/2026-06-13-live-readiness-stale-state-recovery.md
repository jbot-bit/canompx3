# Live Readiness Stale-State Recovery Plan - 2026-06-13

## Status

Active plan. This is a stale-state containment and live-readiness recovery plan,
not deployment authority.

## Purpose

Stop the bot workstream from operating on stale 3-lane/green-readiness claims.
Until the gates are remeasured green, the working baseline is:

- `topstep_50k_mnq_auto` is blocked for live launch.
- The current deployed book has 2 lanes.
- Tokyo is paused, not deployed.
- Criterion 11 and Criterion 12 must be refreshed and reverified before any
  live claim is trusted.

## Truth Protocol

Labels in this plan are binding:

- **MEASURED**: observed from code, canonical files, or commands in this session.
- **INFERRED**: conclusion from measured evidence; must be rechecked before use.
- **UNSUPPORTED**: stale, contradicted, or not checked in this session.

Primary sources used here are local canonical repo files and fresh command
outputs. No external firm pages were reverified in this pass.

## Current Measured State

Measured in WSL at `/home/joshd/canompx3` on local date 2026-06-13.

- **MEASURED**: `git status --short --branch` showed clean `main`, ahead of
  `origin/main` by 7 commits.
- **MEASURED**: `git log --oneline -10` had HEAD `b7980acb`.
- **MEASURED**: `python3 scripts/infra/codex_local_env.py doctor --platform wsl`
  passed environment checks, with warnings for `HANDOFF.md` queue mismatch and
  14 active stage files.
- **MEASURED**: `docs/runtime/lane_allocation/topstep_50k_mnq_auto.json` has
  exactly 2 objects in `lanes`: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` and
  `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`.
- **MEASURED**: the same allocation file lists
  `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` under `paused` with an SR alarm reason
  and operator decision date 2026-06-11.
- **MEASURED**: `./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --format json`
  exited `1` with `broken=3`. Broken items are C11 profile fingerprint mismatch,
  C12 SR state mismatched/legacy, and two deployed lanes with zero execution rows.
- **MEASURED**: `./.venv-wsl/bin/python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only`
  exited `1`. It reported C11 `state=fail`, C12 `state=invalid`, active lanes
  equal to the 2-lane book, and strict blockers `Criterion 11 gate not OK` and
  `Criterion 12 invalid`.
- **MEASURED**: `live_journal.db` is missing and both deployed lanes have zero
  paper/live execution rows in `project_pulse`.
- **MEASURED**: `docs/institutional/pre_registered_criteria.md` Criterion 11
  requires a 90-day account-death Monte Carlo survival probability of at least
  70%, and Criterion 12 requires Shiryaev-Roberts monitoring of deployed
  strategy live R streams.

## Disconfirmed Or Quarantined Claims

- **UNSUPPORTED CURRENTLY**: "current book is a green 3-lane pilot." Older
  `HANDOFF.md` and decision-ledger entries say this, but current allocation and
  readiness commands contradict it.
- **UNSUPPORTED CURRENTLY**: "live readiness is green except telemetry." Fresh
  strict readiness exits nonzero on C11 and C12.
- **UNSUPPORTED CURRENTLY**: "Tokyo is deployed." Current allocation has Tokyo
  paused for SR alarm.
- **INFERRED**: C11/C12 may be pure control-state staleness, but this is not
  deployable evidence. The refresh is a capital-control write and needs explicit
  operator GO.

## Decisions

1. Treat the 2-lane blocked state as the only current operating baseline.
2. Do not start live, increase risk, add accounts, or switch prop profiles from
   this state.
3. Do not run `refresh_control_state.py` as part of background cleanup. It is a
   capital-control write and requires explicit operator GO.
4. Do not rewrite old historical handoff sections. Add superseding current
   entries so history remains auditable while stale claims are not reused.
5. Prop-firm ranking work remains research/integration planning only until the
   current Topstep live path is green or explicitly abandoned.

## Phase 0 - Stale-Guard Before Any Action

Run before any live-readiness, profile, prop-firm, dashboard, or account-routing
work:

```bash
git log --oneline -10
git status --short --branch
python3 scripts/tools/context_resolver.py --task "live readiness topstep_50k_mnq_auto stale state" --format markdown
nl -ba docs/runtime/lane_allocation/topstep_50k_mnq_auto.json | sed -n '1,75p'
./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --format json
./.venv-wsl/bin/python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only
```

Exit criteria:

- The branch, HEAD, dirty state, active lanes, C11, C12, and execution coverage
  are all freshly known.
- If these commands disagree with docs, commands and canonical runtime files win.

## Phase 1 - Contain Stale Docs

Actions:

- Add this plan.
- Add a top `HANDOFF.md` note pointing future tools at this plan.
- Add a new current decision-ledger entry superseding the old 3-lane pilot entry.

Exit criteria:

- A future reader sees the 2-lane blocked state before stale 3-lane history.
- Old history remains intact but is clearly superseded.

## Phase 2 - Read-Only Forensics Before Refresh

Actions:

```bash
./.venv-wsl/bin/python -m trading_app.paper_trade_logger --profile topstep_50k_mnq_auto --sync --dry-run
./.venv-wsl/bin/python scripts/audits/run_all.py --phase 7
```

Checks:

- Confirm why `live_journal.db` is missing.
- Confirm whether zero execution rows are expected after the lane change or are
  an attribution/sync break.
- Confirm phase-7 live controls still block launch while C11/C12 are invalid.

Exit criteria:

- No live write, broker order, webhook, kill, flatten, or account-routing action
  is triggered.
- A concrete cause is recorded for missing execution evidence.

## Phase 3 - Operator-Gated Control-State Refresh

Only after explicit operator GO:

```bash
./.venv-wsl/bin/python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto
./.venv-wsl/bin/python scripts/tools/project_pulse.py --fast --format json
./.venv-wsl/bin/python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn --format json --proof-pack-only
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --preflight
```

Exit criteria:

- C11 is valid and gate-ok under the current 2-lane profile.
- C12 is valid for the current 2-lane profile.
- The preflight report names any remaining blocker. No silent `FAIL ()`.
- If the refresh changes the lane set, restart Phase 0.

## Phase 4 - If Still Blocked

Branch by measured blocker:

- C11 still fingerprint-mismatched: trace `trading_app/account_survival.py`,
  `trading_app/prop_profiles.py`, and lifecycle-state fingerprint inputs.
- C11 valid but economics fail: do not relax thresholds; open a separate
  account-risk remediation prereg/design.
- C12 invalid: inspect SR state envelope via `trading_app/lifecycle_state.py`,
  `trading_app/sr_monitor.py`, and the refresh-control-state output.
- Zero execution rows remain: fix attribution/sync before calling telemetry or
  SR evidence meaningful.

Exit criteria:

- Each blocker has either a patch plan with tests or an explicit operator
  decision to keep live blocked.

## Phase 5 - Clean Half-Hanging Work

After live blocking state is understood:

- Reconcile `HANDOFF.md` with the canonical action-queue render.
- Reclassify the 14 active stage files; only archive stages that are
  `DONE_SAFE`.
- Add queue coverage for pulse findings that currently have no queue item:
  C11 mismatch, C12 invalid, MNQ/MES/MGC stale steps.
- Update stale 3-lane docs only where they are operationally current surfaces.
  Do not churn archive/history files.

## Phase 6 - Prop-Firm Ranking Integration

Do this after current live readiness is no longer ambiguous:

- Keep `docs/research/prop_firm_ranking_2026.md` and the EV scorecard as
  research artifacts unless fresh official firm rules and repo profiles are
  reconciled.
- Do not activate dormant Tradeify/MFFU/Bulenox profiles without C11/C12,
  profile-routing, broker/API, and compliance gates.
- Add a typed no-go rule for any firm/account path whose API, automation,
  device, VPS, copier, or payout rules are not official and non-conflicting.

## Reviewer And Live-Risk Pass

Applied locally as a targeted read-only review using:

- `.claude/agents/live-risk-auditor.md`
- `.codex/agents/canompx3_reviewer.toml`
- `canompx3-live-audit`
- `canompx3-deploy-readiness`

Findings:

- **MEASURED**: current readiness is BLOCKED, not GO.
- **MEASURED**: no live-risk path was exercised in this plan update.
- **SKIPPED**: broker account queries, order submission, cancellation, kill,
  flatten, and webhook checks. Residual risk: they remain unverified for launch,
  but C11/C12 already block launch.
- **SKIPPED**: external official prop-firm page recheck. Residual risk: prop-firm
  ranking remains research-only, not routing authority.

Decision: BLOCK live launch; VERIFY_MORE before any refresh or route change.

## Verification For This Plan Update

Required after doc edits:

```bash
git diff --check
git status --short --branch
```

No test suite is required for this documentation-only stale-state update. If any
runtime code or control-state file changes later, run the focused tests for that
path plus drift/phase-7 live audit.

## Second-Pass Critique

- This plan intentionally does not solve C11/C12. It prevents stale work and
  sets the safe order of operations.
- The riskiest false shortcut is refreshing C11/C12 and treating a green result
  as live authority without rerunning preflight, pulse, phase 7, and execution
  attribution checks.
- The simplest next action is read-only: dry-run execution attribution sync, then
  decide whether an operator-approved control-state refresh is justified.
