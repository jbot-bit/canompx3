---
status: active
owner: codex
last_reviewed: 2026-05-02
superseded_by: ""
---

# Topstep Verification Questions

## Purpose

This document turns the remaining `NEEDS VERIFICATION` items in the official
decision packet into explicit, narrow questions for Topstep and/or ProjectX.

Use this before any outbound support request. Do not improvise the ask.

## Already Verified From Official Docs

- New Topstep Trading Combines are TopstepX-only and reset continuity is
  TopstepX-only.
- TopstepX API supports third-party tools, custom dashboards/tools, alerts,
  monitoring systems, and direct order execution.
- TopstepX API activity must originate from the trader's own device. VPS, VPNs,
  and remote servers are prohibited.
- Live Funded accounts prohibit automated trading through the ProjectX API.
- Topstep says a TopstepX account cannot be connected to other trading
  platforms.
- ProjectX official docs expose real-time account, order, and position updates
  via WebSocket/SignalR and REST.

## Open Questions

### Q1 — Live read-only assistive tooling

**Question**

Is a local-device tool permitted in `Live Funded` if it:

- connects through the official TopstepX / ProjectX API
- consumes account / order / position / market-data updates
- displays dashboards, alerts, and operator warnings
- does **not** place, cancel, or modify orders
- does **not** automate execution
- runs only on the trader's own device

**Why this matters**

This is the lead architecture candidate. If the answer is yes, the path stays
clean. If the answer is no or ambiguous, the candidate is blocked.

**What we need back**

- explicit yes / no
- whether any additional restrictions apply in `Live Funded`
- whether read-only monitoring/alerts are treated differently from automated
  trading

### Q2 — Scope of “connected tools” in Live

**Question**

When TopstepX API docs say traders can connect third-party tools, create custom
dashboards/tools, and set up alerts/monitoring systems, does that permission
apply to `Live Funded` accounts so long as the tool is non-executing?

**Why this matters**

The docs clearly allow connected tools in general, but the Live policy only
explicitly bans automated trading through the ProjectX API. We need to know
whether passive tooling remains in-bounds.

**What we need back**

- explicit yes / no
- whether passive dashboards/alerts are considered permitted connected tools in
  `Live Funded`

### Q3 — Order-authority feedback expectations

**Question**

If a trader executes manually in TopstepX while using a separate local tool for
monitoring and risk overlays, are there any restrictions on consuming order and
position updates from the API for reconciliation and alerting?

**Why this matters**

The repo's canonical brain wants one truth surface. We need to know whether
passive reconciliation tooling is allowed even when order entry stays manual in
TopstepX.

**What we need back**

- explicit yes / no
- any restrictions on order/account state consumption for passive monitoring

## Questions That No Longer Matter

Do **not** ask these unless the architecture direction changes:

- "Can TopstepX accounts be connected directly to Quantower?"
  This is already answered by Topstep's supported-platform article: no.
- "Can ProjectX API automation run in Live Funded?"
  Already answered: no.

## Suggested Support Packet

If sending to Topstep or ProjectX, keep it short and binary:

1. State that the tool is local-device only.
2. State that it is non-executing.
3. State that it consumes API data for dashboards/alerts/monitoring only.
4. Ask whether this is permitted in `Live Funded`.
5. Ask whether any additional restrictions apply.

## Decision Rule

- If Q1/Q2 return a clear yes, the sidecar path is materially de-risked.
- If Q1/Q2 return no, the sidecar path is blocked.
- If Q1/Q2 return ambiguous marketing language instead of a policy answer, keep
  the architecture blocked and do not treat that as approval.
