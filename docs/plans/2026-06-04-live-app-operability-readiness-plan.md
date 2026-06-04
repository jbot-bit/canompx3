# Live App Operability, Readiness, and Lane Deployment Plan

Date: 2026-06-04
Owner: Codex
Status: PLAN / NO-GO until measured gates pass
Profile in scope: `topstep_50k_mnq_auto`
Primary goal: make the dashboard open reliably on demand while preserving fail-closed live trading controls.

## Decision summary

The plan separates three states that must not be conflated:

1. **Dashboard open** — the control room should open quickly even when trading is blocked.
2. **Signal / read-only usable** — the app can render cached or live-read-only state and show exactly what is blocked.
3. **Live execution allowed** — real-money trading remains fail-closed behind strict launch gates.

The immediate target is not to bypass blockers. The target is to prevent blockers such as dirty git state, stale leases, DuckDB contention, stale evidence, or deep preflight runtime from making the operator blind at the trade window.

## Evidence posture

### Measured from repository state

- The live runbook currently says status is **NO-GO** until live preflight is green with no blocking strict-readiness warnings.
- The live runbook pins the pilot to `topstep_50k_mnq_auto`, `MNQ`, and `--copies 1`; the dashboard starts signal-only, and real-money launch is via `HOLD TO GO LIVE`.
- The hard blocker list includes dirty/ahead/behind git state, lane state, live stage acceptance, journal DB availability, broker/session checks, `copies>1` without per-shadow protection, and degraded router state.
- The active profile is `topstep_50k_mnq_auto`, active, Topstep 50k, one copy, daily loss belt `$450`, max slots `7`, MNQ-only, and dynamic lanes loaded from allocation JSON.
- The active allocation JSON for `topstep_50k_mnq_auto` contains three active MNQ lanes and rebalance date `2026-05-30`.
- The C11 closeout says strict C11 diagnostic failures are visible non-launch-blocking warnings; the binding C11 gate remains operational Monte Carlo survival with current profile fingerprint.
- `scripts/tools/worktree_guard.py` is the canonical worktree lease authority and uses a heartbeat/session/parent-process model rather than relying on an ephemeral subprocess file lock.
- `scripts/tools/live_readiness_report.py` already builds a report containing profile, git, allocation, C11/C12, telemetry, automation health, active lanes, strict warnings, and a proof pack.

### Official external documentation used

- DuckDB official concurrency documentation: in-process DuckDB supports one read-write process, while multiple processes may read in read-only mode; multi-process writing requires a different architecture such as a server/protocol pattern. Source: <https://duckdb.org/docs/current/connect/concurrency.html>
- Git official `git worktree` documentation: a repository can support multiple linked working trees with separate working-tree metadata, enabling a clean runtime checkout alongside a dirty development tree. Source: <https://git-scm.com/docs/git-worktree.html>
- Kubernetes official probe documentation: startup, liveness, and readiness are distinct concepts; readiness failure should not be treated as application startup failure. Source: <https://kubernetes.io/docs/concepts/workloads/pods/probes/>
- Twelve-Factor App official methodology: backing services and disposability support treating runtime services as replaceable/observable units rather than hidden singleton assumptions. Source: <https://12factor.net/>

### Explicit unknowns / no-claim zones

- No live database was inspected in this session.
- No broker account, ProjectX session, order router, or live journal state was inspected.
- No claim is made that C11/C12 currently pass in the operator environment.
- No claim is made that the dashboard currently starts on Windows.
- No claim is made that the current three-lane book is economically optimal; only the committed allocation file was inspected.
- No claim is made that any 100k account is superior; that requires a measured survival-and-take-home frontier.

## Operating doctrine

### Principle 1 — dashboard fail-open, execution fail-closed

The dashboard is an observability and command surface. It should open in degraded mode whenever possible. Real-money execution is a capital-risk action and must remain blocked until strict live gates pass.

### Principle 2 — dirty development state is not equivalent to unsafe observability

Dirty git state should block live execution from that tree. It should not prevent the operator from opening a read-only dashboard. The runtime execution tree should be clean and pinned; the development tree may be dirty.

### Principle 3 — DB lock contention is a design smell for dashboard refresh

Dashboard refresh should not require taking a write-capable handle on the research DB. Research/validated reads should be read-only, and dashboard state should be snapshot-first with bounded refresh timeouts.

### Principle 4 — preflight must be time-budgeted

The project needs separate startup, fast readiness, and deep verification checks. A 10:00 trade window cannot depend on starting a long deep audit at 10:00.

### Principle 5 — lane deployment must optimize a frontier, not one scalar

A lane/account selection decision should compare survival probability, drawdown distribution, expected take-home, opportunity count, correlation, operational load, and account rule constraints. Drawdown survival alone is insufficient.

## Workstream A — App opens when needed

### Goal

The dashboard opens quickly, renders current or stale/degraded state, and explains blockers without requiring the operator to debug git, leases, DB locks, or preflight scripts during a trade window.

### Plan

1. Add a documented three-tier launch contract:
   - `STARTUP_OK`: web server can start and serve status.
   - `SIGNAL_OK`: read-only / cached state is usable enough for signal monitoring.
   - `LIVE_OK`: strict live launch gates pass.
2. Ensure `trading_app/live/bot_dashboard.py` starts even when live readiness is red.
3. Render live blockers as dashboard cards rather than startup exceptions.
4. Add startup recovery output with:
   - git runtime state
   - lease state
   - DB state
   - snapshot state
   - readiness state
   - last auto-fix attempt
5. Add a bounded refresh timeout. On timeout, serve last known snapshot and mark it stale.

### Acceptance criteria

- Dashboard process starts even when git is dirty, provided no critical import/environment failure exists.
- Dashboard displays `LIVE BLOCKED` instead of failing to render.
- Refresh returns a response within a fixed budget.
- Stale data is clearly labeled with age and source.
- Live launch remains impossible while blockers exist.

### Implementation tasks

- Create `trading_app/live/startup_recovery.py` for deterministic startup diagnosis.
- Add dashboard cards for startup, signal, live, git, lease, DB, C11, C12, broker, and lane state.
- Add tests for dirty git, stale snapshot, unavailable DB, and live-readiness failure.

## Workstream B — Clean runtime worktree and git-state sorting

### Goal

Normal development dirt should not prevent opening the dashboard. Live launch should use a known clean runtime checkout.

### Plan

1. Define a runtime worktree, for example `canompx3-app-runtime`, managed by existing worktree tooling.
2. Launcher path:
   - detect dev tree state
   - locate or create runtime worktree
   - verify runtime worktree clean state
   - launch dashboard from runtime worktree when possible
3. If runtime worktree cannot be prepared:
   - dashboard may open from current tree in `DEGRADED / LIVE DISABLED`
   - live launch remains blocked
4. Extend worktree launch reporting to show:
   - descriptor
   - worktree path
   - branch/head
   - clean/dirty
   - lease holder
   - heartbeat age
   - reclaimability

### Acceptance criteria

- Dirty dev tree no longer blocks dashboard observability.
- Dirty runtime tree blocks live launch.
- Stale dead lease can be reclaimed safely.
- Live peer lease blocks live launch and displays holder details.

### Implementation tasks

- Extend `scripts/tools/worktree_launch_preflight.py` to emit a richer JSON diagnosis.
- Add a launcher wrapper that prefers the runtime worktree.
- Add tests for clean runtime / dirty dev, dirty runtime, stale lease, live lease, malformed lease.

## Workstream C — DB and refresh resilience

### Goal

Dashboard refresh cannot hang or fail hard because a DuckDB writer or journal lease is active.

### Plan

1. Inventory all dashboard and preflight DB access.
2. Use `read_only=True` for research/validated shelf reads.
3. Split state into:
   - research DB read-only facts
   - live journal write state
   - cached readiness snapshot
   - broker/runtime live state
4. Build a snapshot-first dashboard path:
   - serve latest atomic JSON snapshot immediately
   - trigger background refresh when allowed
   - mark snapshot stale if age exceeds threshold
5. Keep journal writes in one live writer path with explicit health checks.

### Acceptance criteria

- Dashboard still renders when research DB is locked/unavailable.
- DB status is visible and specific: readable, locked, unavailable, stale snapshot, journal writable.
- Refresh has a bounded timeout and never blocks the UI indefinitely.

### Implementation tasks

- Create a live readiness snapshot writer with atomic file replace.
- Update dashboard refresh to read snapshot first.
- Add tests with a simulated locked or missing DB.

## Workstream D — Preflight split and evidence freshness

### Goal

Preflight becomes predictable and time-budgeted.

### Plan

1. Classify checks:
   - startup checks: imports, server port, config files
   - fast live checks: clean runtime tree, profile match, fresh snapshot, no known blockers
   - deep checks: audits, drift, project pulse, C11/C12 proof rebuilds, lane rebalance
2. Add evidence age and commit SHA to each deep check result.
3. Run deep verification before trade windows, not during the final launch click.
4. At `HOLD TO GO LIVE`, run fast checks and reject if deep evidence is stale.

### Acceptance criteria

- Fast preflight has a bounded runtime target.
- Deep evidence records command, head SHA, profile fingerprint, and timestamp.
- Live launch blocks on stale deep evidence but the dashboard still opens.

### Implementation tasks

- Version `scripts/tools/live_readiness_report.py` output schema.
- Add fields for check class, evidence age, and launch impact.
- Add tests proving startup failure, readiness failure, and live failure are separate states.

## Workstream E — C11/C12 clarity

### Goal

C11 and C12 must be visible, machine-readable, and unambiguous about launch impact.

### Plan

1. Split C11 into:
   - operational survival gate
   - strict diagnostic warning state
2. Keep C12 as a separate validity gate.
3. Dashboard cards must show:
   - status
   - launch impact
   - evidence age
   - profile fingerprint
   - required remediation
4. Live launch must enforce only the binding policy, not ambiguous text.

### Acceptance criteria

- Strict C11 diagnostic warning is not silently hidden.
- Strict C11 diagnostic warning does not block if policy says advisory.
- Operational C11 failure blocks live.
- C12 failure blocks live when policy marks it binding.

### Implementation tasks

- Add/verify schema fields in `live_readiness_report.py` for C11 operational, C11 diagnostic, and C12.
- Add dashboard rendering and tests for advisory-vs-blocking behavior.

## Workstream F — Lane/account frontier and resource-grounded sizing

### Goal

Decide what to trade by measuring the economic frontier, not by optimizing only drawdown survival or only ExpR.

### Plan

1. Compare current three-lane `topstep_50k_mnq_auto` book against candidate account/profile books.
2. Evaluate each candidate on:
   - expected R
   - expected dollars
   - expected take-home after payout policy
   - breach probability
   - max drawdown distribution
   - daily-loss hit probability
   - time under water
   - trade frequency and missed opportunity
   - lane correlation and session concentration
   - operational complexity
3. Include at least these candidate families:
   - current 3-lane Topstep 50k pilot
   - 50k expanded candidates within existing max slots
   - Topstep 100k candidate profiles
   - inactive Type-A profiles if activation policy permits
   - self-funded candidates only if separated from prop-firm cap assumptions
4. Output a Pareto frontier, not a single recommendation.

### Acceptance criteria

- No lane promotion is based on in-sample performance alone.
- No account recommendation is based only on drawdown survival.
- Every candidate reports sample size, validation status, evidence date, account constraints, and launch blockers.
- The selected book has an explicit tradeoff statement: why this point on the frontier rather than safer/larger alternatives.

### Implementation tasks

- Extend or wrap `scripts/tools/optimal_lanes.py`, `scripts/tools/score_lanes.py`, and `scripts/tools/rebalance_lanes.py`.
- Add result docs under `docs/audit/` or `docs/plans/`.
- Add tests that self-funded sizing is not capped by prop-firm rules and prop profiles use their own tier constraints.

## Workstream G — Recent commit integration audit

### Goal

Merged work must be proven reachable and aligned across dashboard, preflight, runner, docs, and tests.

### Plan

1. Bucket recent commits by subsystem:
   - dashboard/startup
   - live runner/session orchestrator
   - worktree/preflight
   - DB/journal
   - lane allocation
   - C11/C12/risk
   - docs/runbook
   - tests/drift
2. For each commit or feature, trace:
   - changed files
   - intended behavior
   - runtime caller
   - dashboard surface
   - live launch impact
   - tests
   - docs/runbook alignment
3. Classify each as:
   - integrated
   - code present but not called
   - called but not surfaced
   - surfaced but not enforced
   - enforced but not tested
   - doc-only / stale / superseded

### Acceptance criteria

- No live-impacting feature is accepted because it merely exists in code.
- Every live-impacting feature has a runtime caller and a dashboard/report surface.
- Every blocker appears in operator-facing language.

### Implementation tasks

- Create `docs/audit/2026-06-04-recent-live-integration-audit.md`.
- Add a trace table from `git log` to runtime paths.
- Convert gaps into implementation tickets.

## Workstream H — Operator runbook and auto-fix boundaries

### Goal

The operator knows what the app did automatically, what remains manual, and what will never be auto-fixed.

### Safe automatic recovery

Allowed:

- reclaim stale dead leases with clear evidence
- refresh read-only snapshots
- restart dashboard server if old holder is dead
- prune temporary runtime files
- fall back to degraded read-only mode
- display broker unavailable state

Not allowed:

- force-reset dirty worktree
- delete live journal data
- bypass C11/C12
- switch profile/account silently
- increase copies
- start live mode
- flatten broker exposure unless invoked through explicit kill/flatten control

### Acceptance criteria

- Every auto-fix is logged.
- Every manual blocker has one clear next action.
- Unsafe recovery is documented as prohibited.

## Phasing

### Phase 0 — Plan and state contract

Deliverables:

- This plan.
- A versioned live-state schema proposal.
- Updated handoff state.

Exit criteria:

- Workstreams are taskable independently.
- Unknowns and no-claim zones are explicit.

### Phase 1 — Dashboard-open unblock

Deliverables:

- Dashboard degraded mode.
- Startup recovery report.
- Snapshot-first refresh skeleton.

Exit criteria:

- App opens and renders blocker cards even when live is blocked.

### Phase 2 — Runtime worktree and lease recovery

Deliverables:

- Dedicated runtime worktree launch path.
- Rich lease diagnostics.
- Safe stale-lease reclaim.

Exit criteria:

- Dirty dev tree cannot prevent dashboard observability.
- Dirty runtime tree still blocks live.

### Phase 3 — DB-safe refresh and preflight split

Deliverables:

- Atomic readiness snapshot.
- Fast/deep preflight classification.
- Evidence age and freshness gates.

Exit criteria:

- Refresh is bounded.
- Deep evidence is not recomputed at the trade click unless explicitly requested.

### Phase 4 — Live launch hardening

Deliverables:

- Strict `HOLD TO GO LIVE` gate backed by the live-state schema.
- Operator confirmation payload.
- Audit log of launch attempts and blockers.

Exit criteria:

- Degraded dashboard mode cannot launch live.
- Live launch fails closed on any binding blocker.

### Phase 5 — Lane/account frontier

Deliverables:

- Survival-plus-take-home frontier report.
- Candidate account/profile comparison.
- Explicit promotion/no-promotion decision.

Exit criteria:

- Account/lane selection has measured economic and risk tradeoffs.

### Phase 6 — Integration audit closeout

Deliverables:

- Recent-commit integration audit.
- Missing wiring/tests/docs backlog.
- Final runbook update.

Exit criteria:

- Live-impacting changes are reachable, surfaced, enforced, and tested.

## Work packages for parallel agents

### Agent 1 — Dashboard/startup

Scope: `trading_app/live/bot_dashboard.py`, startup recovery, UI cards, degraded mode.

Output: patch plus tests proving dashboard opens when live readiness is red.

### Agent 2 — Worktree/git/lease

Scope: `scripts/tools/worktree_guard.py`, `scripts/tools/worktree_launch_preflight.py`, runtime worktree launcher.

Output: patch plus tests for dirty dev tree, clean runtime tree, stale lease, and live peer lease.

### Agent 3 — DB/snapshot

Scope: dashboard refresh paths, snapshot writer, read-only DB access, journal health display.

Output: patch plus tests for locked/missing DB and stale snapshot.

### Agent 4 — Readiness schema/C11/C12

Scope: `scripts/tools/live_readiness_report.py`, schema docs, C11/C12 dashboard contract.

Output: patch plus tests for advisory vs blocker states.

### Agent 5 — Lane/account frontier

Scope: lane scoring and account profile analysis tooling.

Output: research/audit report with measured frontier and explicit go/no-go recommendations.

### Agent 6 — Integration audit

Scope: recent commits to runtime traceability.

Output: audit doc with gaps and task list; no code changes unless directed.

## Final live go/no-go checklist

Live remains **NO-GO** unless all binding gates are measured green:

- Runtime worktree clean and approved.
- Profile explicitly selected: `topstep_50k_mnq_auto` unless a new profile is separately approved.
- Instrument explicitly selected: `MNQ` for this pilot.
- Copies remain `1` unless per-shadow loss belts are implemented and verified.
- Allocation profile matches runtime profile.
- Active lanes are not stale, paused, blocked, or SR alarmed.
- C11 operational survival gate passes on current profile fingerprint.
- C12 validity gate passes if binding under current policy.
- Strict C11 diagnostic warnings are visible and reviewed.
- Broker account/mode/session is verified.
- Journal DB is writable and has no unresolved incomplete trade for the target day.
- Router is not degraded.
- Kill/flatten path is available.
- Operator confirms profile, account, mode, instrument, and copies in the dashboard before launch.

## Immediate next actions

1. Implement Phase 1 dashboard degraded mode and startup recovery report.
2. Implement Phase 2 runtime worktree launch path.
3. Implement Phase 3 snapshot-first refresh.
4. Create the live-state schema and bind dashboard + live launch to it.
5. Run recent-commit integration audit before changing lane/account deployment.
6. Run lane/account frontier only after app operability is stable enough to avoid mixing infrastructure failures with strategy conclusions.
