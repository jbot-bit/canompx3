---
status: active
owner: codex
last_reviewed: 2026-05-02
superseded_by: ""
---

# Official-Doc Decision Packet

## Purpose

This workstream exists to verify the external operator architecture against
official documentation before any platform-specific integration work proceeds.

This is a documentation and decision packet only. It does not authorize code
integration with Quantower, TopstepX, MotiveWave, Sierra, or any other external
surface.

## Required Inputs

Use only:

- repo-canonical code and docs
- official Topstep / TopstepX documentation
- official ProjectX API documentation
- official candidate-platform connection / replay / SDK documentation

Mark every conclusion as:

- `VERIFIED`
- `INFERRED`
- `NEEDS VERIFICATION`

## Questions To Answer

1. What is allowed in `Combine`, `XFA`, and `Live Funded`?
2. Does a read-only/assist sidecar remain clean under official policy text?
3. Does any candidate platform introduce a second truth surface for order or
   account state?
4. Can a candidate consume the canonical export contract without re-encoding
   the ORB/research brain?
5. Which assumptions remain unproven and must block implementation?

## Mandatory Output

The final packet must include:

- stage-split policy matrix
- candidate matrix
- unsupported assumptions register
- kill criteria
- explicit "implementation still blocked" line unless all blockers are cleared

## Non-Negotiable Rules

- No platform code in this worktree
- No scripting-layer migration plans as a first step
- No "likely allowed" claims without an official citation or explicit
  `NEEDS VERIFICATION` label
- No collapsing `Combine/XFA/Live` into one blended judgment

## Repo-Grounded Constraints

These are the local invariants any external operator path must respect:

- `VERIFIED` `bot_state.py` is the current projection layer. The export surface
  is now richer, but external platforms must still consume repo-exported truth
  rather than infer ORB, stop, or target from charts.
- `VERIFIED` `build_state_snapshot()` in `trading_app/live/bot_state.py`
  produces the canonical operator contract and `write_state()` is the atomic
  export boundary.
- `VERIFIED` `session_orchestrator.py` owns feed status aggregation, router
  degraded state, and live session context.
- `VERIFIED` `_publish_state()`, `_feed_status_payload()`,
  `_router_status_payload()`, and `_broker_status_payload()` in
  `trading_app/live/session_orchestrator.py` are the canonical runtime-health
  projection points.
- `VERIFIED` `projectx/data_feed.py` owns stale/reconnect semantics.
- `VERIFIED` `ProjectXDataFeed` defines `_STALE_TIMEOUT`,
  `_MAX_STALE_BEFORE_RECONNECT`, and `on_stale` callback semantics that any
  external surface must treat as authoritative.
- `VERIFIED` `projectx/order_router.py` owns live order semantics, native
  brackets, price-collar enforcement, and order-status polling.
- `VERIFIED` `ProjectXOrderRouter` enforces price-collar checks and native
  bracket placement; any external shell that bypasses it creates a second
  authority for live trade semantics.
- `VERIFIED` `copy_order_router.py` owns cross-account divergence detection.
- `VERIFIED` `CopyOrderRouter.is_degraded()` and
  `CopyOrderRouter.degraded_accounts()` are already the canonical multi-account
  failure signals.

Implication:

- `VERIFIED` any external surface that cannot live as a thin consumer of this
  contract is the wrong integration shape.

## Required External Contract

Any external operator surface must consume, at minimum:

- `VERIFIED` signal identity and lifecycle: `lane_key`, `strategy_id`,
  `status`, `status_detail`, `direction`
- `VERIFIED` trade geometry: `entry_price`, `stop_price`, `target_price`,
  `risk_points`
- `VERIFIED` ORB geometry: `orb_high`, `orb_low`, `orb_size`, `orb_complete`,
  `orb_break_direction`, `orb_break_time_utc`, `orb_complete_time_utc`
- `VERIFIED` timing and freshness: `signal_time_utc`, `entry_time_utc`,
  `exit_time_utc`, `heartbeat_utc`
- `VERIFIED` runtime health: `feed_status`, `router_status`, `broker_status`
- `VERIFIED` account context: `account_id`, `account_name`, `instrument`,
  `session_name`, `trading_day`

Implication:

- `VERIFIED` if a candidate needs to infer any of the above from charts,
  account UI state, or platform-local script logic, it fails the integration
  shape test.

## Official-Doc Evidence

### Topstep / TopstepX

- `VERIFIED` New Trading Combines became TopstepX-only on 2025-07-07, and
  practical reset continuity became TopstepX-only on 2025-08-01.
  Source:
  <https://help.topstep.com/en/articles/8284149-tradovate-connection-instructions>
- `VERIFIED` Topstep Tradovate accounts include charting/replay add-ons but do
  **not** include API access.
  Source:
  <https://help.topstep.com/en/articles/8284149-tradovate-connection-instructions>
- `VERIFIED` TopstepX API supports automated strategies, third-party tools,
  custom dashboards/tools, market data, and direct order execution.
  Sources:
  <https://help.topstep.com/en/articles/11187768-topstepx-api-access>
- `VERIFIED` TopstepX API use must originate from the trader's own device; VPS,
  VPNs, and remote servers are prohibited.
  Source:
  <https://help.topstep.com/en/articles/11187768-topstepx-api-access>
- `VERIFIED` TopstepX API is linked from inside TopstepX through ProjectX
  linking, not as a free-floating separate stack.
  Source:
  <https://help.topstep.com/en/articles/11187768-topstepx-api-access>
- `VERIFIED` In `Live Funded`, automated strategies are allowed generally, but
  automated trading through the ProjectX API is prohibited.
  Source:
  <https://help.topstep.com/en/articles/10657969-live-funded-account-parameters>
- `VERIFIED` TopstepX cannot connect to TradingView and does not currently offer
  custom indicators.
  Source:
  <https://help.topstep.com/en/articles/14434175-topstepx>
- `VERIFIED` Topstep permits Micro Gold (`MGC`).
  Source:
  <https://help.topstep.com/en/articles/8284224-permitted-products-per-exchange>

### Quantower

- `VERIFIED` Quantower documents direct connections for Topstep, ProjectX, and
  Rithmic.
  Sources:
  <https://help.quantower.com/quantower/connections/connection-to-topstep>
  <https://help.quantower.com/quantower/connections/connecting-to-projectx>
  <https://help.quantower.com/quantower/connections/connection-to-rithmic>
- `VERIFIED` Quantower's Topstep connection uses separate Trading Combine
  credentials, not Topstep dashboard login credentials.
  Source:
  <https://help.quantower.com/quantower/connections/connection-to-topstep>
- `VERIFIED` Quantower says Topstep accounts get Premium features like DOM,
  charting, and copy trading under one login, but do **not** get Market Replay,
  Strategy Manager, Strategy Runner, or local order strategies without extra
  Quantower licensing.
  Source:
  <https://help.quantower.com/quantower/connections/connection-to-topstep>
- `VERIFIED` Quantower Algo exposes custom indicators and strategy tooling.
  Source:
  <https://help.quantower.com/quantower/quantower-algo>
- `VERIFIED` Quantower Market Replay exists, but the Topstep connection page
  explicitly says it is not included free for Topstep accounts and requires an
  Advanced Features license.
  Sources:
  <https://help.quantower.com/quantower/trading-panels/market-replay>
  <https://help.quantower.com/quantower/connections/connection-to-topstep>

### ProjectX

- `VERIFIED` ProjectX markets its API for custom tools, third-party software,
  automation, real-time data, and trade management.
  Source:
  <https://www.projectx.com/api>

## Operational Constraints From Official Docs

- `VERIFIED` Any TopstepX API-assisted path must run from the trader's own
  device. Cloud-hosted sidecars, VPS-based relays, or remote-server automation
  are dead on arrival under current Topstep policy.
- `VERIFIED` A Quantower path does not remove Topstep policy dependence. It
  only changes the operator shell and, in some cases, the licensing burden.
- `VERIFIED` Quantower Topstep accounts do not include replay or local order
  strategies for free. That weakens the naive "Quantower gives everything"
  assumption materially.
- `INFERRED` A read-only sidecar remains the cleanest architecture only if it
  can stay local-device, non-executing, and state-consumer-only.

## Stage-Split Policy Matrix

| Path | Combine | XFA | Live Funded | Status |
|---|---|---|---|---|
| TopstepX native manual | `VERIFIED` allowed | `VERIFIED` allowed | `VERIFIED` allowed | clean baseline |
| TopstepX API full automation | `VERIFIED` allowed in product docs | `VERIFIED` allowed in product docs | `VERIFIED` prohibited via ProjectX API | blocked as end-state |
| TopstepX read-only / assistive sidecar | `INFERRED` plausible | `INFERRED` plausible | `NEEDS VERIFICATION` no explicit read-only carve-out found | not cleared yet |
| Quantower manual shell to Topstep/ProjectX | `VERIFIED` documented connection exists | `VERIFIED` documented connection exists | `NEEDS VERIFICATION` no Topstep source explicitly blesses this lifecycle path | challenger only |
| Quantower Algo / platform-native automation | `NEEDS VERIFICATION` docs show tooling exists | `NEEDS VERIFICATION` docs show tooling exists | `NEEDS VERIFICATION` likely collides with Live policy if routed through ProjectX API | blocked pending proof |

## Candidate Assessment

### Candidate A — TopstepX-core + read-only/assist sidecar

- `VERIFIED` Fits repo architecture best because the live brain already exports
  signal, ORB, feed, and router truth externally.
- `VERIFIED` Avoids platform scripting lock-in if kept read-only/assistive.
- `VERIFIED` Must remain local-device only if it touches TopstepX API or any
  linked ProjectX account tooling.
- `NEEDS VERIFICATION` No official Topstep article found in this pass that
  explicitly says passive read-only overlays/tools are allowed in `Live Funded`
  when using TopstepX API / ProjectX-linked tooling.
- `INFERRED` Still the best-looking path, but not yet policy-cleared.

### Candidate B — TopstepX + Quantower shell

- `VERIFIED` Quantower clearly supports connection and operator tooling.
- `VERIFIED` Quantower Topstep feature access is tiered; replay and local order
  strategies are not included free for Topstep accounts.
- `VERIFIED` Quantower does not inherit repo-native order authority. If the
  human executes there directly, `ProjectXOrderRouter` is bypassed unless an
  explicit feedback bridge is built.
- `NEEDS VERIFICATION` No official Topstep source found here that says this is a
  first-class, future-proof `Combine -> XFA -> Live` path under current TopstepX
  platform policy.
- `INFERRED` Strong challenger for operator UX, weaker on lifecycle clarity and
  lock-in risk.

### Candidate C — API-driven Live automation through ProjectX

- `VERIFIED` Dead as an end-state `Live Funded` architecture because official
  Topstep policy prohibits automated trading through the ProjectX API there.

## Unsupported Assumptions Register

- `NEEDS VERIFICATION` "Read-only sidecar is policy-neutral in Live Funded."
- `NEEDS VERIFICATION` "Quantower manual shell is explicitly acceptable for the
  exact TopstepX-era lifecycle the user wants."
- `NEEDS VERIFICATION` "External passive overlays do not count as prohibited
  connected tools in Live Funded when coupled to account/order data."
- `NEEDS VERIFICATION` "Quantower execution can feed back enough canonical
  account/order truth to avoid a second authority surface."

## Kill Criteria

- Kill any path that requires moving ORB/session/filter/risk logic into
  platform scripting.
- Kill any path that depends on ProjectX API automation in `Live Funded`.
- Kill any path that requires off-device hosting, VPS, or remote execution to
  be viable under normal use.
- Kill any path that cannot surface canonical feed stale / router degraded state.
- Kill any path that turns account/order truth into a split-brain problem.
- Kill any path justified only by charts or replay aesthetics.

## Decision Dashboard

| Candidate | Connection reality | Operator fit | Overlay path | Failure handling | Lock-in risk | Current read |
|---|---|---|---|---|---|---|
| TopstepX-core + read-only/assist sidecar | `VERIFIED` TopstepX-first lifecycle fit; `NEEDS VERIFICATION` for explicit Live read-only allowance | `INFERRED` strong for manual-assist workflows | `VERIFIED` best match for repo contract shape | `VERIFIED` can consume canonical stale/degraded signals directly | `VERIFIED` low if non-executing | **lead candidate, still gated** |
| TopstepX + Quantower shell | `VERIFIED` connection docs exist; `NEEDS VERIFICATION` for full Topstep lifecycle acceptance | `INFERRED` strong shell ergonomics, weaker policy clarity | `INFERRED` viable only if it stays render-only | `INFERRED` adds second failure domain and potential split-brain | `INFERRED` medium-high | **challenger only** |
| ProjectX API-driven Live automation | `VERIFIED` technically marketed by ProjectX | `INFERRED` could be powerful in non-Live stages | `VERIFIED` not needed for current repo shape | `VERIFIED` blocked by Live policy regardless | `INFERRED` high | **dead as end-state** |

## Next Verification Tasks

These are the only questions worth spending more time on before any platform
implementation:

1. `NEEDS VERIFICATION` Ask or find official written guidance on whether a
   passive read-only tool that consumes TopstepX/ProjectX-linked account and
   order data, but never places/cancels orders, is permitted in `Live Funded`.
2. `NEEDS VERIFICATION` Confirm whether Topstep treats Quantower as a supported
   long-run shell for new TopstepX-era traders all the way through `Live`, or
   only as a documented connection path.
3. `NEEDS VERIFICATION` Confirm whether any candidate shell can feed back enough
   order/account truth to avoid bypassing `ProjectXOrderRouter` as the
   canonical live-order authority.

## Provisional Ranking

1. `INFERRED` TopstepX-core + read-only/assist sidecar
2. `INFERRED` TopstepX + Quantower shell
3. `VERIFIED` ProjectX API-driven Live automation path is blocked

## Current Verdict

`VERIFIED` Implementation remains blocked at the platform layer.

What is cleared:

- repo-local canonical export contract work
- repo-local operator-state projection work

What is **not** cleared:

- any external platform integration
- any Live Funded API-assisted execution path
- any assumption that a read-only sidecar is officially allowed in Live

The next valid step from this branch is to close the `NEEDS VERIFICATION` items
with additional official documentation or explicit written support guidance.
