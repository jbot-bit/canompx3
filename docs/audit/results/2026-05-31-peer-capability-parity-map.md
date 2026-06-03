# Peer capability parity map

Date: 2026-05-31
Phase: Competitive benchmark follow-through, Phase 1 research capture
Status: COMPLETE for research capture; implementation remains Phase 2+

## Purpose

This executes the promised Phase 1 deliverable from
`docs/audit/results/2026-05-02-competitive-landscape-canompx3.md`: fetch and
pin primary-source evidence from peer trading frameworks, map capability gaps,
and state where canompx3 is weaker.

This is not a strategy-validation artifact. No trading edge is claimed here and
no live/capital behavior is authorized.

## Source basis

External evidence was refreshed from official project documentation on
2026-05-31. Internal evidence was read from this checkout.

External primary sources:

- QuantConnect LEAN engine docs:
  https://www.quantconnect.com/docs/v2/lean-engine/getting-started
- QuantConnect LEAN CLI backtest docs:
  https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-backtest
- QuantConnect LEAN CLI report docs:
  https://www.quantconnect.com/docs/v2/lean-cli/reports
- NautilusTrader overview:
  https://nautilustrader.io/docs/latest/concepts/overview/
- NautilusTrader backtesting docs:
  https://nautilustrader.io/docs/latest/concepts/backtesting/
- Freqtrade backtesting docs:
  https://www.freqtrade.io/en/stable/backtesting/
- Freqtrade hyperopt docs:
  https://www.freqtrade.io/en/stable/hyperopt/
- VectorBT getting-started docs:
  https://vectorbt.dev/
- Backtrader features:
  https://www.backtrader.com/home/features/
- Backtrader live trading intro:
  https://www.backtrader.com/docu/live/live/

Internal source files:

- `RESEARCH_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `scripts/tools/session_preflight.py`
- `scripts/tools/project_pulse.py`
- `scripts/tools/live_readiness_report.py`
- `trading_app/account_survival.py`
- `trading_app/sr_monitor.py`
- `docs/audit/results/2026-05-02-competitive-landscape-canompx3.md`

## Capability parity map

Scoring:

- Impact: 1 low, 5 high expected value if closed.
- Complexity: S small doc/script packaging, M targeted implementation, L broad
  architecture or data-work dependency.
- Evidence: MEASURED when directly sourced; INFERRED when a repo-local need is
  inferred from the comparison.

| Peer | Peer capability from primary docs | canompx3 current equivalent | Gap | Evidence | Impact | Complexity | EV lane |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| LEAN | Local backtest command runs through Docker and writes timestamped results; report command creates a shareable backtest/live report artifact. | `scripts/tools/live_readiness_report.py` exists and is read-only; many research runners emit markdown artifacts, but there is no single external-comparison benchmark report command. | Add a standard report-pack command for benchmark/live-readiness artifacts with fixed output paths and source manifest. | MEASURED | 5 | M | EV-2, EV-3 |
| LEAN | Modular engine pieces for datafeed, transaction processing, result handling, realtime events, and setup are explicit extension points. | canompx3 has strong domain-specific modules and fail-closed gates, but fewer stable public adapter seams for external users. | Define the internal "extension seams" that are allowed to be reused: data input, benchmark run, readiness report, broker/preflight surface. | INFERRED | 3 | M | EV-3 |
| NautilusTrader | Common core is shared across backtest, sandbox, and live contexts; backtest/live are formal environment contexts. | canompx3 separates research, validation, and live doctrine well, but user-facing commands are still spread across preflight, pulse, account survival, SR monitor, and readiness scripts. | Keep the separation, but wrap it in profile-level commands so operators do not have to know every underlying surface. | MEASURED | 5 | M | EV-1, EV-2 |
| NautilusTrader | High-level backtest API can run multiple configured backtests, while low-level API supports fine-grained control. | canompx3 has many specialized research runners and prereg front doors; no small standard "benchmark pack" for naive/trend/core-lane comparison. | Build one bounded benchmark harness with fixed configs rather than another flexible research runner. | INFERRED | 4 | M | EV-3 |
| NautilusTrader | Backtesting docs emphasize data integrity invariants, sorting requirements, reset behavior, and streaming/large-data handling. | canompx3 has strong canonical-layer and holdout rules, plus drift checks, but these are doctrine-heavy and not packaged as a benchmark-run contract. | Translate canonical-layer rules into a machine-readable benchmark contract: inputs, split, leakage guards, outputs. | MEASURED | 4 | M | EV-3 |
| Freqtrade | Backtesting can run from CLI or webserver mode, exports result bundles, and includes strategy/config copies for reproducibility assumptions. | canompx3 result docs are detailed, but artifact bundles vary by runner and do not consistently include an auto-written source manifest. | Standardize result bundles: command, git SHA, DB path/fingerprint, config/profile, source files, warnings, unsupported fields. | MEASURED | 5 | S/M | EV-2, EV-3 |
| Freqtrade | Hyperopt is a first-class workflow using Optuna over defined parameter spaces. | canompx3 intentionally resists broad parameter mining via preregistration and multiple-testing controls. | Do not copy hyperopt broadly. Borrow only the operator ergonomics: named search spaces, fixed trial accounting, exported results. | MEASURED | 3 | M | EV-3 |
| VectorBT | Vectorized NumPy/Pandas/Numba/Rust approach is designed for very large parameter throughput. | canompx3 is slower but more domain-governed; throughput is not the main weakness if finite-data discipline holds. | Avoid chasing VectorBT speed first. Use it as a pressure test for "how quickly can we run fixed baselines," not for larger unfenced sweeps. | MEASURED | 2 | L | EV-3 |
| VectorBT | Notebook-style charts/dashboards support rapid inspection. | canompx3 emits many markdown docs and JSON states; visual/report ergonomics are weaker. | Add lightweight HTML/markdown summary packs after command contracts exist. Do not prioritize visuals before reproducible source manifests. | INFERRED | 2 | M | EV-2 |
| Backtrader | Event-driven broker/trading logic, vectorized indicators where possible, live feeds, analyzers, broker model, slippage/commission, and observers are packaged as common features. | canompx3 has stronger project-specific futures cost/session logic and fail-closed policy, but less general analyzer/observer packaging. | Add a tiny observer/analyzer contract for profile reports: lane status, C11, C12, allocation, telemetry, git/data provenance. | MEASURED | 4 | M | EV-2 |
| Backtrader | Docs explicitly call out look-ahead-safe indexing and event-only mode. | canompx3 doctrine bans leakage and protects holdout, but benchmark artifacts should show the leakage guard instead of assuming readers know the doctrine. | Each benchmark artifact should include "leakage guard used" and "future data excluded" fields. | MEASURED | 4 | S | EV-3 |

## Disconfirming notes: where canompx3 is worse

1. canompx3 is weaker than peers on operator packaging.

   LEAN, Freqtrade, NautilusTrader, and Backtrader all present recognizable
   high-level user surfaces. canompx3 has many correct pieces, but a user still
   has to know which of `session_preflight`, `project_pulse`,
   `live_readiness_report`, `account_survival`, and `sr_monitor` to run.

2. canompx3 is weaker than Freqtrade/LEAN on reproducible result bundles.

   The repo writes detailed markdown research results, but there is no uniform
   guarantee that every Phase 2/3 artifact carries command, git SHA, DB
   fingerprint, profile/config, source file list, warnings, and unsupported
   fields in one schema.

3. canompx3 is weaker than NautilusTrader on explicit environment-context
   ergonomics.

   The project has a real research/validation/live separation, but it is encoded
   across doctrine and scripts. Nautilus exposes backtest/live contexts as
   first-class platform concepts. canompx3 should not copy the architecture, but
   should expose the profile-level context more cleanly.

4. canompx3 is weaker than VectorBT on sweep throughput, but this is not the
   highest-EV gap.

   The repo's finite-data and preregistration discipline intentionally blocks
   broad unfenced search. Faster sweeping is useful only after the benchmark
   harness has hard trial accounting and leakage guards.

5. canompx3 is stronger than most peers on anti-bias doctrine, but that strength
   is discounted when the operator has to reconstruct the command sequence.

   This is the central finding. The project is not lacking seriousness. It is
   leaking EV through fragmented proof packaging.

## Updated EV ranking

1. EV-2 moves to joint-first with EV-1 if `live_readiness_report.py` stays green.

   The old report called EV-2 missing. Current repo truth shows
   `scripts/tools/live_readiness_report.py` already exists as a one-command
   read-only aggregation over deployment truth, Criterion 11, Criterion 12,
   lane state, and rebalance provenance. The next task is not "invent EV-2"; it
   is to harden its artifact schema and make it the standard profile proof pack.

2. EV-1 remains mandatory, but it is a reliability gate, not the whole program.

   `session_preflight.py` and `project_pulse.py` already exist. The missing
   parity feature is a clean bootstrap/health contract that produces a stable
   result artifact and does not require the user to know the internals.

3. EV-3 is still the clearest peer-comparison gap.

   The peers make it easy to run comparable backtests or reports. canompx3 needs
   a fixed benchmark harness with naive/trend/core-lane baselines, not a new
   open-ended optimizer.

## Recommended Phase 2 specs

### Spec A: profile proof pack wrapper

Goal: turn `live_readiness_report.py` into the standard EV-2 artifact.

Minimum output fields:

- profile id
- git SHA and dirty flag
- DB path and DB fingerprint or snapshot timestamp
- active lanes and paused/stale reasons
- Criterion 11 state and age
- Criterion 12 state and latest verdict
- telemetry maturity
- strict warnings split into blocking vs advisory
- command line used
- unsupported or missing evidence

Verification:

- targeted tests for JSON schema and advisory-vs-blocking warning split
- `python scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --format json --strict-zero-warn`

### Spec B: bootstrap health proof

Goal: make EV-1 emit a small stable artifact rather than console-only output.

Minimum output fields:

- expected interpreter
- actual interpreter
- repo root
- git branch/SHA/dirty state
- canonical DB path and presence
- active mutating-session claim status
- pulse broken/startup blockers
- next command if blocked

Verification:

- targeted tests around missing interpreter, missing DB, dirty tree, and claim collision
- `python scripts/tools/session_preflight.py --context codex-wsl`
- `python scripts/tools/project_pulse.py --fast --format json`

### Spec C: bounded benchmark harness

Goal: implement EV-3 with fixed scope and anti-mining controls.

Minimum benchmark set:

- naive baseline
- simple trend baseline
- current core lane family baseline

Hard constraints:

- fixed date split declared before run
- fixed cost model
- no parameter rescue after seeing results
- no 2026 holdout selection leakage
- all failed controls included in the artifact

Verification:

- unit tests for split contract and leakage guard
- generated markdown + JSON artifact under `docs/audit/results/`

## What was tested

- Official peer docs were refreshed on 2026-05-31.
- Repo-local command surfaces were inspected for EV-1 and EV-2 existence.
- Research doctrine was checked for preregistration, holdout, Criterion 11, and
  Criterion 12 constraints.

## What was not tested

- No peer framework was installed or executed locally.
- No canompx3 benchmark harness was implemented.
- No live-readiness command was run against `gold.db` in this artifact.
- No trading result, edge, or lane promotion was evaluated.

## What failed

- The earlier Phase 1 deliverable did not exist before this file.
- The old EV-2 wording was stale: the one-command live-readiness report now
  exists, but it still needs standardized proof-pack treatment.
- The peer-comparison gap remains rhetorical until EV-3 emits a real comparable
  benchmark artifact.

## What remains unsupported

- Any claim that canompx3 is quantitatively better or worse than LEAN,
  NautilusTrader, Freqtrade, VectorBT, or Backtrader.
- Any claim about execution quality, slippage realism, or fill-model superiority
  versus peers.
- Any claim that the current live book is ready to trade from this artifact.

## What would change the decision

- EV-2 emits a reproducible profile proof pack with complete provenance and no
  hidden missing fields.
- EV-3 emits a fixed-scope benchmark report with naive/trend/core-lane baselines
  and explicit leakage controls.
- A later run executes the same benchmark contract through at least one peer
  engine or a peer-style independent implementation.

## Decision

Worth continuing.

The highest-value next work is not more broad research. It is Phase 2 design
and Phase 3 implementation for:

1. EV-2 profile proof pack hardening.
2. EV-1 bootstrap health proof artifact.
3. EV-3 bounded benchmark harness.

Do not start a new strategy-discovery thread from this comparison. The peer
research says the project needs cleaner proof packaging first.
