---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Operator UX Principles and Flows

**Date:** 2026-04-13
**Status:** APPROVED DESIGN TARGET
**Priority:** HIGH
**Depends on:** `docs/plans/2026-04-13-operator-product-layer-consolidation-plan.md`
**Goal:** Define the operator experience standard for the existing dashboard so it is understandable, safe, and low-friction for a new user, an interrupted user, and an ADHD user.

---

## 1. Executive Standard

The app must be operable by someone who:

- has never used the project before
- is tired
- is interrupted
- has low working memory in the moment
- cannot be expected to remember hidden commands, prompts, rituals, or launch order

The operator experience standard is:

**Open app -> understand state immediately -> take the next safe action**

Anything that depends on remembering terminal commands, mode flags, hidden sequencing, or tribal knowledge fails this standard.

---

## 2. Product Standard: "Bloomberg-Level" for This Project

"Bloomberg-level" here does **not** mean copying Bloomberg visuals.
It means adopting the same operator qualities:

- information-dense but legible
- stateful, not decorative
- deterministic status semantics
- fast situational awareness
- clear action hierarchy
- no ambiguity about what is live, stale, blocked, or simulated
- keyboard- and interruption-friendly
- designed for repeat professional use under time pressure

For this project, that means:

- zero-terminal operation for normal use
- single operator shell
- explicit status model
- plain-language system explanations
- strong guardrails around live trading
- persistent, queryable recent activity
- no fake calm when the system is stale or broken

---

## 3. Scope and Relationship to Existing Plans

This document is a **UX companion** to the consolidation plan.

It does **not** authorize:

- a new dashboard shell
- `ui_v2/`
- React rewrite
- Streamlit resurrection
- a chart-first redesign

It applies only to the existing operator shell:

- `trading_app/live/bot_dashboard.py`
- `trading_app/live/bot_dashboard.html`

This document supersedes older operator-UX direction where it conflicts with:

- alternate dashboard roots
- multi-shell ideas
- terminal-first workflows

---

## 4. Operator Personas

### 4.1 New Operator

Characteristics:

- understands trading basics
- does not know this codebase
- does not know internal script names
- does not know mode semantics yet

Needs:

- obvious safe starting point
- plain English status and actions
- confidence that `Paper` is not `Live`
- no reliance on docs during operation

### 4.2 Builder-Operator

Characteristics:

- knows the system deeply
- has context contamination risk
- overestimates what is "obvious"

Needs:

- app that does not assume memory
- app that surfaces silent failures instead of relying on instinct
- friction against casual live actions

### 4.3 ADHD / Interrupted Operator

Characteristics:

- attention can fragment
- working memory is scarce during action
- interruptions are normal
- state must be recoverable at a glance

Needs:

- one obvious next action
- stable visual semantics
- persistent state summaries
- low recall burden
- interruption-safe recovery

---

## 5. UX Principles

### Principle 1: Recognition over recall

The app must show what to do next.
The operator must not have to remember:

- commands
- run order
- mode meanings
- hidden prerequisites
- which subsystem is failing

### Principle 2: One primary action at a time

At any moment, the UI should make the next best action obvious:

- `Run Preflight`
- `Fix Broker Connection`
- `Start Alerts`
- `Start Paper`
- `Stop Session`
- `Do Not Trade`

Not all actions should compete equally for attention.

### Principle 3: Stable state semantics

Colors, badges, and labels must mean the same thing everywhere.

Example:

- green = safe / ready / healthy
- amber = caution / degraded / review needed
- red = blocked / live risk / critical failure
- blue = informational / simulated / paper
- grey = stopped / inactive / no current action

Do not overload colors with multiple meanings.

### Principle 4: The app carries the mental model

The system should explain:

- what mode is running
- what account is selected
- what broker is connected
- what session is next
- why something is blocked
- what changed recently

The operator should not reconstruct this from logs.

### Principle 5: Interruptions must be survivable

If the operator walks away and comes back, the app must make it obvious:

- whether anything is running
- whether anything failed
- whether a signal fired
- whether the session is stale
- whether trading is still safe

### Principle 6: Hard distinction between signal, paper, and live

The operator must never confuse:

- observing signals
- placing simulated orders
- sending real orders

These modes need different visual treatment, wording, and confirmation flow.

### Principle 7: Quiet when healthy, loud when broken

Healthy systems should feel calm.
Broken or stale systems should be unmistakable.

Do not bury critical failures in passive text or secondary tabs.

### Principle 8: The dashboard is an operating instrument, not a brochure

Every element must answer one of these:

- What is happening?
- What is next?
- What is wrong?
- What should I do?

If it does not, it is likely noise.

---

## 6. Information Architecture

The dashboard should be understood as five layers of information.

### Layer A: Global system state

Always visible.

Shows:

- active mode
- active profile
- active broker/account
- heartbeat freshness
- highest current risk/health status

### Layer B: Readiness and blockers

Shown prominently when not running, and still accessible when running.

Shows:

- pass / warn / blocked by category
- plain-English explanation
- recommended next action

### Layer C: Timeline and sessions

Shows:

- next sessions in Brisbane time
- countdown where appropriate
- active session state
- lane ownership under selected profile

### Layer D: Live activity and alerts

Shows:

- event feed
- alerts
- latest critical issues
- whether an alert is acknowledged

### Layer E: Supporting detail

Collapsible / secondary.

Shows:

- broker connection details
- account details
- trade log detail
- expanded diagnostics

The operator should not need Layer E for normal use.

---

## 7. Required State Model

These top-level states must be explicit and mutually understandable:

- `STOPPED`
- `READY`
- `BLOCKED`
- `DEGRADED`
- `RUNNING_ALERTS`
- `RUNNING_PAPER`
- `RUNNING_LIVE`
- `STOPPING`
- `STALE`
- `ERROR`

### State definitions

#### `STOPPED`

Nothing is running.
No active bot session.

Primary action:

- `Run Preflight`

#### `READY`

All required checks passed for the selected profile/broker/account.

Primary action:

- `Start Alerts`
- secondary: `Start Paper`

#### `BLOCKED`

The system must not start.

Primary action:

- the specific fix action for the highest-priority blocker

Examples:

- connect broker
- refresh auth
- data stale
- DD halt active

#### `DEGRADED`

The system is usable with caution, but not ideal.

Examples:

- optional checks warn
- component self-test warnings
- stale advisory data

Primary action:

- review warning or continue in safe mode only

#### `RUNNING_ALERTS`

Signal-only mode.
No orders sent.

Primary action:

- `Stop Session`

#### `RUNNING_PAPER`

Simulated order flow is active.

Primary action:

- `Stop Session`

#### `RUNNING_LIVE`

Real money or combine-impacting order flow is active.

Primary action:

- `Stop Session`

#### `STOPPING`

Shutdown command has been accepted but completion is pending.

Primary action:

- none except emergency escalation if stop does not complete

#### `STALE`

The dashboard has old state and cannot safely claim current truth.

Examples:

- heartbeat too old
- feed silence beyond threshold
- session expected but no runtime updates

Primary action:

- recover/restart path

#### `ERROR`

The app hit an unexpected failure and cannot guarantee safe interpretation.

Primary action:

- show operator-safe fallback and explicit non-safety

---

## 8. Core Screen Layout

### 8.1 Top Bar

Must always show:

- product name
- selected profile/account
- mode badge
- current high-level state
- global health badge
- clock in Brisbane time

This is the "at a glance" anchor.

### 8.2 Primary Action Zone

Directly below top bar.

Must show the one most relevant action based on state.

Examples:

- `Run Preflight`
- `Connect Broker`
- `Start Alerts`
- `Start Paper`
- `Type LIVE to unlock live mode`
- `Stop Session`

This zone exists to reduce choice paralysis.

### 8.3 Readiness Card

Shows structured readiness categories:

- broker/auth
- account/profile
- data/features
- session safety
- DD/risk
- component self-test

Each row must show:

- status
- short explanation
- optionally: fix action

### 8.4 Session Timeline

Shows:

- upcoming sessions
- Brisbane times
- currently active session
- session progression

The next session should be visually dominant.

### 8.5 Live Event Feed

Shows recent events in human language:

- preflight passed
- broker connected
- session started
- ORB forming
- signal fired
- order submitted
- fill confirmed
- stop placed
- target hit
- session stopped

This should be persistent enough to survive short interruptions.

### 8.6 Alert Panel

Shows:

- critical alerts first
- warnings second
- advisories last

Must include:

- timestamp
- severity
- category
- message
- current status if still active

### 8.7 Connections / Details

Secondary area for:

- broker credentials status
- connection enable/disable
- account fetch detail
- diagnostics

This should not be the primary operating surface.

---

## 9. Language Standards

### 9.1 Use operator language, not implementation language

Prefer:

- `Broker not connected`
- `Data is stale`
- `Daily drawdown halt is active`
- `Paper session running`

Avoid exposing raw internal jargon as the first explanation:

- `subprocess failed`
- `instance lock collision`
- `bot_state parse issue`
- `component self-test failure`

Internal detail can appear in expandable diagnostics.

### 9.2 All blocking messages need action guidance

Bad:

- `Contract resolution failed`

Good:

- `Cannot resolve front-month contract. Refresh broker connection or rerun preflight.`

### 9.3 Avoid hidden mode semantics

Do not expect users to infer:

- alerts = signal-only
- paper = simulated orders
- live = real orders / combine-impacting orders

Each label must be self-explaining or paired with a sublabel.

---

## 10. New User Flow

### Goal

A first-time operator can reach a safe watched session without docs or terminal.

### Flow

1. Open app
2. App auto-shows current profile/account/broker health
3. If blocked, app says why and what to do
4. Operator clicks `Run Preflight`
5. App shows structured pass/warn/block results
6. Operator selects `Alerts` or `Paper`
7. App shows session status and live event feed
8. Operator can stop from the same screen

### Acceptance test

A new user should be able to explain:

- whether the system is safe to start
- what mode is active
- what the next session is
- how to stop it

within 30 seconds of opening the app.

---

## 11. ADHD-Friendly Design Rules

### Rule 1: Minimize memory burden

Never require the user to remember:

- hidden sequences
- command syntax
- status meanings unique to one panel
- "did I already do preflight?"

### Rule 2: Persistent summaries beat transient logs

Important information should remain visible until replaced or acknowledged.

### Rule 3: One dominant action

At every major state, highlight one primary safe action.

### Rule 4: Avoid mode ambiguity

Signal/paper/live must be visually and verbally distinct.

### Rule 5: Use progressive disclosure

Show:

- essential truth first
- detail second
- raw diagnostics last

### Rule 6: Recoverability over cleverness

When the operator returns after interruption, the app must help them recover context fast.

### Rule 7: Calm default, urgent exception

Healthy idle state should not feel like an alarm.
Broken state should not feel calm.

---

## 12. Mode-Specific UX

### 12.1 Alerts / Signal-Only

Visual treatment:

- green or safe informational styling
- explicit text: `No orders are sent`

UI focus:

- next session
- signal feed
- status of qualification / arming / signal

### 12.2 Paper

Visual treatment:

- blue / simulated styling
- explicit text: `Practice orders only`

UI focus:

- order lifecycle
- fill visibility
- trade outcome

### 12.3 Live

Visual treatment:

- red-accent risk styling
- explicit text: `Real orders`

UI requirements:

- stronger unlock path
- stronger confirmation language
- persistent live-mode badge
- impossible to confuse with paper

---

## 13. Confirmation Design

### 13.1 Safe modes need low friction

`Run Preflight`, `Alerts`, and `Paper` should be easy to start.

### 13.2 Live mode needs intentional friction

Live mode should require:

- unmistakable live context
- explicit unlock gesture
- final confirmation with consequences stated

But the friction must be in the app, not in remembered shell syntax.

---

## 14. Error and Failure Handling

### 14.1 Stale state

If state is stale, the app must say:

- that it is stale
- how stale it is
- what that means
- what to do next

Never present stale state as current truth.

### 14.2 Feed failure

If the feed is silent during an active session, the app must escalate visually and in alerts.

### 14.3 Broker disconnect

Broker disconnect must appear as an operator problem, not a hidden backend detail.

### 14.4 Restart recovery

After restart, the app should say:

- whether it recovered prior state
- whether recovery is trustworthy
- whether a session is still running or presumed dead

---

## 15. Observability Requirements

The operator must be able to answer these without opening logs:

- Is the bot alive?
- Is the feed alive?
- Is the broker connected?
- Is the selected account usable?
- Is a session running?
- Did a signal fire?
- Did an order get placed?
- Did something fail?

If any answer requires terminal logs, the operator UX is incomplete.

---

## 16. Keyboard and Speed Standards

The app should be usable quickly under pressure.

Preferred:

- obvious tab order
- keyboard activation for primary actions
- no hidden hover-only controls
- no tiny click targets for important actions

This does not require Bloomberg-style hotkey density on day one.
It does require low-friction operator control.

---

## 17. Visual Standards

### 17.1 Dense, not cluttered

Show high-signal information.
Avoid decorative cards that repeat the same truth.

### 17.2 Contrast and legibility first

This is an operations tool, not a brand page.

### 17.3 Important information must survive peripheral vision

The operator should be able to notice:

- mode
- blocker
- live risk
- critical alerts

without reading every panel.

### 17.4 Consistency beats novelty

Once a state, badge, or color is assigned a meaning, keep it stable.

---

## 18. Anti-Patterns

Do not do the following:

- hide readiness behind a modal only
- bury failures inside raw logs
- require a docs read before normal operation
- make users guess whether state is current
- show multiple equally prominent start buttons with no recommendation
- rely on color alone for live vs paper distinction
- add major visual complexity before the trust layer is finished
- create a second operator shell for a "cleaner UX"

---

## 19. Acceptance Criteria

This UX spec is satisfied when:

1. A new user can open the app and understand the current state within 30 seconds.
2. A user with no terminal use can run a watched alerts session from the app.
3. A user with no terminal use can run a watched paper session from the app.
4. The app explicitly explains blocked, degraded, stale, ready, and running states.
5. The difference between alerts, paper, and live is unmistakable.
6. A short interruption does not destroy situational awareness.
7. Critical failures are visible without reading logs.
8. The app presents one obvious safe next action in each top-level state.

---

## 20. Immediate Build Implications

This spec implies the next build work should prioritize:

1. structured readiness cards
2. explicit state model and top-level badges
3. health/liveness summary
4. alert panel and persistent activity feed
5. clearer start/stop action hierarchy

This spec does **not** justify:

1. new app shells
2. chart-first expansion
3. research feature growth
4. more operator docs as a substitute for product behavior

---

## 21. Final Rule

If the operator must remember something important, the product is still unfinished.

The dashboard must carry the memory, the state model, the sequencing, and the warnings.
That is the bar.
