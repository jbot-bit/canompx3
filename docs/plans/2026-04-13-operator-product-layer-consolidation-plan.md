# Operator Product Layer Consolidation Plan

**Date:** 2026-04-13
**Status:** APPROVED DESIGN TARGET
**Priority:** HIGH
**Goal:** Finish the existing operator app into a professional trading cockpit without creating a second dashboard, second launcher, or second source of truth.

---

## 1. Executive Decision

The project already has an operator-facing app:

- `trading_app/live/bot_dashboard.py`
- `trading_app/live/bot_dashboard.html`
- `scripts/run_live_session.py`
- `trading_app/pre_session_check.py`

This stack is the **canonical operator product layer**.

From this point forward:

- Do **not** build a second dashboard shell
- Do **not** create `ui_v2/` or a parallel operator frontend
- Do **not** reintroduce Streamlit for live trading operations
- Do **not** create a separate "operator console" app
- Do **not** create chart-first work before operator reliability is finished

The problem is **not** "missing app."
The problem is that the current app is still a **thin UI wrapper over script-oriented workflows**.

This plan finishes that layer instead of spawning another one.

---

## 2. Current-State Audit

### 2.1 What already exists in code

#### A. Operator shell exists

`trading_app/live/bot_dashboard.py` already provides:

- FastAPI server
- single-page local web UI host
- account/profile APIs
- broker connection APIs
- preflight action endpoint
- stop/kill endpoint
- live journal + bot state readers

`trading_app/live/bot_dashboard.html` already provides:

- account selector
- mode badge
- preflight button
- kill button
- profile cards
- lane summaries by session
- `Alerts` / `Paper` / `Live` buttons
- connections tab

#### B. Launch/orchestration exists

`scripts/run_live_session.py` already provides:

- profile-based launch
- `--signal-only`
- `--demo`
- `--live`
- `--preflight`
- live dashboard auto-launch

#### C. Safety gate exists

`trading_app/pre_session_check.py` already provides:

- manual halt checks
- data freshness checks
- DD tracker checks
- daily equity checks
- account/HWM checks
- hard-gate style session validation

#### D. Backend trading orchestration exists

`trading_app/live/session_orchestrator.py` and related live modules already own:

- session timing
- broker integration
- lane execution
- state updates
- journal writes
- runtime risk / DD logic

### 2.2 What is missing

The missing layer is **not launch controls**.
The missing layer is **professional operator trust**:

- clear app-level health state
- structured preflight results instead of raw subprocess text
- alerting when the operator is not watching
- feed/broker/session liveness surfaced in-app
- restart recovery / stale state recovery clarity
- one obvious "what do I do next?" operator path
- removal of CLI-shaped seams from the product experience

### 2.3 Why the current UX still feels wrong

The dashboard docstring says:

> "Control buttons shell out to CLI."

That is the exact seam the operator feels.

The shell exists, but the operator experience still behaves like:

`dashboard -> subprocess -> script -> logs`

instead of:

`dashboard -> operator action -> supervised system state`

That is why the current product feels unfinished even though many backend pieces already exist.

---

## 3. Canonical Surfaces

These components are now declared canonical for operator use.

### 3.1 Operator UI

- `trading_app/live/bot_dashboard.py`
- `trading_app/live/bot_dashboard.html`

**Ownership:**
- user-facing operator workflow
- account/broker/session visibility
- operator actions
- alert/history surface
- health/readiness summary

### 3.2 Operator launcher backend

- `scripts/run_live_session.py`

**Ownership:**
- session process startup
- mode selection
- orchestration bootstrap
- controlled process lifecycle

This is a backend launch surface, **not** the primary operator UX.

### 3.3 Session safety gate

- `trading_app/pre_session_check.py`

**Ownership:**
- authoritative session readiness validation
- reasons to block launch
- fail-closed pre-session rules

This remains the canonical validation engine.
Its outputs must be translated into operator-facing UI language, not replaced.

### 3.4 Runtime truth surfaces

- `data/bot_state.json`
- `live_journal.db`
- state files under `data/state/`

**Ownership:**
- operator state read model
- health, PnL, lane, trade, and risk state

### 3.5 Live orchestration

- `trading_app/live/session_orchestrator.py`
- broker adapters / connections
- live feed / journal / trackers

**Ownership:**
- actual trading behavior
- runtime state changes
- event production

---

## 4. Anti-Duplication Rules

These rules are mandatory. Violating them creates operator debt.

### Rule 1: One cockpit only

All operator-facing live-trading UX goes into the existing bot dashboard.

Do not create:

- `ui_v2/`
- alternate web shell
- Streamlit live cockpit
- separate operator desktop app
- chart-only control surface

### Rule 2: One launcher only

`scripts/run_live_session.py` remains the only canonical launch backend.

The dashboard may call it or share its internals, but must not create a second launch stack with different semantics.

### Rule 3: One preflight authority only

`trading_app/pre_session_check.py` remains the canonical session readiness authority.

Do not implement a second parallel preflight engine inside the dashboard.

### Rule 4: UI translates backend truth; it does not re-encode it

The dashboard must consume existing state and verdicts.
It must not invent separate rules for:

- DD limits
- lane eligibility
- session timing
- broker readiness
- validation status

### Rule 5: Old plans are not new scopes

Older design docs may contain good ideas, but they do **not** authorize a parallel implementation surface.

Port ideas into the current dashboard or leave them parked.

### Rule 6: Productize seams; do not hide them with more docs

If the pain is:

- subprocess output
- stale status
- missing alerting
- confusing launch flow

the answer is code/product integration, not another spec tree.

---

## 5. Duplication Map

### 5.1 Plans that are now informationally superseded for operator-shell architecture

These docs may still contain useful ideas, but they are **not** the active shell direction:

- `docs/plans/2026-03-06-dashboard-copilot.md`
- `docs/plans/2026-03-07-dashboard-v2-design.md`
- `docs/plans/2026-03-07-dashboard-v2-phase1-plan.md`
- `docs/plans/2026-03-07-dashboard-v2-phase4-6-design.md`

Reason:

- they reflect earlier Streamlit / `ui_v2` / alternate-shell thinking
- the repo has already converged on `bot_dashboard.py/html`

### 5.2 Plans that remain directly relevant

- `docs/plans/2026-03-25-bot-dashboard-design.md`
- `docs/plans/2026-03-06-go-live-deployment.md`
- `docs/plans/2026-03-07-live-observability.md`
- `docs/plans/2026-03-16-feed-liveness-monitor-design.md`
- `docs/plans/2026-04-04-bot-dashboard-alerting.md`
- `docs/superpowers/specs/2026-04-11-dashboard-embedded-chart-signal-overlay.md`

Reason:

- they build on the existing shell
- they identify missing runtime trust / observability
- the chart spec explicitly says to extend the existing dashboard, not create a new app

---

## 6. Product Problem Statement

The operator should be able to:

1. Open the app
2. See whether the system is healthy, blocked, stale, or ready
3. See the next sessions in Brisbane time
4. See which profile/account/broker is active
5. Run `Alerts`, `Paper`, or `Live` from the app
6. Watch the session lifecycle and trade lifecycle in-app
7. Receive alerts when something fails
8. Recover safely if the system dies or restarts

Today, the project partially supports these actions, but not with a coherent operator-grade path.

That gap is the operator product layer.

---

## 7. Desired End State

### 7.1 Professional operator workflow

The app becomes the default and sufficient workflow:

- open app
- inspect readiness
- choose account/profile
- run preflight
- start mode
- observe live state
- respond to alerts
- stop safely

No terminal should be required for normal operation.

### 7.2 Clear state model

Every operator-visible state must be explicit:

- `READY`
- `BLOCKED`
- `RUNNING_SIGNAL`
- `RUNNING_DEMO`
- `RUNNING_LIVE`
- `STALE`
- `DEGRADED`
- `STOPPING`
- `STOPPED`
- `ERROR`

The current product leaks implementation detail instead of exposing a stable operating model.

### 7.3 One source of runtime truth

The dashboard becomes the single place to inspect:

- broker connection health
- session countdown / active lane state
- account state and DD
- alerts
- recent activity
- process heartbeat
- trade log summary

---

## 8. Scope In

### Phase A: Consolidate and clarify existing shell

Objective:

- remove ambiguity about what the operator app is

Deliverables:

- dashboard is the declared operator shell
- old alternate-shell plans are treated as archival/informational
- no new frontend root is created

### Phase B: Structured readiness and health

Objective:

- make the app immediately answer "can I run this safely right now?"

Deliverables:

- structured preflight summary in UI
- explicit blocked/warn/pass statuses by category
- broker/feed/account/session health summary
- stale-state detection surfaced in UI

### Phase C: Runtime observability and alerts

Objective:

- remove the need to stare at the dashboard constantly

Deliverables:

- alert panel in dashboard
- persisted alert history
- critical liveness alerts
- DD/risk/position/feed warnings
- fail-open alert integration

### Phase D: Start/stop reliability

Objective:

- make launching and stopping predictable

Deliverables:

- mode transitions reflected clearly in UI
- single active-session ownership model
- restart/stale recovery visibility
- explicit "session running under profile X / account Y / broker Z"

### Phase E: Supervised operator flow

Objective:

- support watched signal/demo validation without terminal use

Deliverables:

- next-session countdown
- session lane cards
- live event feed
- obvious operator prompts: waiting / forming / armed / signal / in trade / done

### Phase F: Unattended trust layer

Objective:

- make unattended demo/live operationally credible

Deliverables:

- feed silence detection
- broker disconnect detection
- stale bot heartbeat detection
- alert persistence
- restart recovery state

### Phase G: Visual upgrades only after trust layer

Objective:

- only after the operator path is finished

Possible deliverables:

- embedded chart
- richer overlays
- visual lane drill-down

This phase is intentionally last.

---

## 9. Scope Out

Not part of this plan:

- new operator shell
- React rewrite
- Streamlit resurrection
- alternate `ui_v2` live app
- TradingView integration as a prerequisite
- new research authority
- new backtest / discovery features
- new governance file tree

---

## 10. Implementation Sequence

### Step 1: Canonicalization pass

Make the code/docs state unambiguous:

- document that `bot_dashboard.py/html` is the operator shell
- document that `run_live_session.py` is the backend launcher
- document that `pre_session_check.py` is the readiness authority
- explicitly mark alternate dashboard-shell plans as non-canonical for implementation

### Step 2: Structured preflight in dashboard

Current issue:

- preflight is launched from dashboard but returned as raw script text

Target:

- dashboard shows category-by-category readiness:
  - auth
  - broker
  - profile
  - features/data
  - contract resolution
  - component self-tests
  - DD/risk blockers

### Step 3: Runtime health model

Add app-level health summary:

- bot heartbeat age
- feed heartbeat age
- broker connection status
- account fetch status
- session process state
- stale-state banner if state is old

### Step 4: Alert engine integration

Implement the alerting path already scoped in the alerting plan:

- critical alerts
- warning alerts
- advisory alerts
- dashboard alert panel
- persisted alert log

### Step 5: Operator event feed

Dashboard should expose human-readable live activity:

- session started
- ORB forming
- lane armed/disqualified
- signal fired
- order submitted
- fill confirmed
- exit complete
- stop/kill acknowledged
- feed stale / reconnect / auth failure

### Step 6: Start/stop lifecycle hardening

Improve operator confidence around launch control:

- clear current mode
- clear active profile/account
- explicit stop-in-progress state
- stale subprocess cleanup surfaced as state, not hidden behavior
- "already running" / lock-state handling translated in UI

### Step 7: Remove operator dependence on terminal

Acceptance:

- no normal watched signal/demo session requires shell use
- shell remains for debugging and development only

---

## 11. File Ownership Plan

### Keep and extend

- `trading_app/live/bot_dashboard.py`
- `trading_app/live/bot_dashboard.html`
- `scripts/run_live_session.py`
- `trading_app/pre_session_check.py`
- `trading_app/live/session_orchestrator.py`

### Likely additions

Additions are allowed only if they support the existing shell and do not create a second surface.

Examples:

- `trading_app/live/alert_engine.py`
- small dashboard helper modules
- structured preflight/result adapters
- event serialization helpers

### Explicitly forbidden additions

- `ui_v2/` as a separate operator shell
- second dashboard server
- second launch backend
- second preflight engine

---

## 12. Operator UX Requirements

The dashboard must make these answers obvious:

### 12.1 Am I safe to start?

Show:

- pass / warn / block
- exact blocking reason
- recommended next action

### 12.2 What is going to happen next?

Show:

- upcoming sessions in Brisbane time
- which lanes belong to the selected profile
- whether a session is waiting / forming / running / done

### 12.3 What is running now?

Show:

- mode
- active profile
- active broker connection
- active account
- current session status
- heartbeat freshness

### 12.4 What broke?

Show:

- alert panel
- most recent critical issue
- whether the issue is acknowledged
- whether the system is still safe to continue

### 12.5 What should I do?

Show:

- start mode
- stop mode
- fix connection
- rerun preflight
- do not trade

The operator must not need to infer the next action from raw logs.

---

## 13. Acceptance Criteria

This plan is complete when all of the following are true:

1. The operator can run a watched `signal-only` session from the dashboard without terminal use.
2. The operator can run a watched `demo` session from the dashboard without terminal use.
3. The dashboard shows structured readiness, not only raw preflight text.
4. The dashboard shows broker/feed/process health in one place.
5. Critical failures produce visible alerts without the operator reading logs.
6. A stale/dead session is visibly stale/dead in the app within a bounded time.
7. There is still exactly one operator shell, one launch backend, and one preflight authority.
8. No new dashboard root or parallel operator app is created.

---

## 14. Decision Heuristics

When deciding whether to add something, use this test:

### Add it if:

- it reduces operator dependence on terminal
- it surfaces backend truth more clearly
- it improves launch / stop / health / alert / recovery
- it reuses the existing dashboard shell

### Do not add it if:

- it creates a second UI root
- it duplicates readiness logic
- it duplicates runtime truth
- it is mostly visual and does not improve operator reliability
- it belongs to research rather than operations

---

## 15. Immediate Next Step

The next implementation slice should be:

**Dashboard Readiness + Health Consolidation**

Meaning:

- keep current dashboard shell
- upgrade preflight from raw subprocess dump to structured operator status
- surface runtime health/liveness
- wire alerting into the existing shell

That is the shortest path from "engine with buttons" to "professional operator product."
