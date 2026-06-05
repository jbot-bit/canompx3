# Live Pilot Gap Plan - Bias-Grounded Checklist

Date: 2026-05-31

Purpose: keep the Topstep/ProjectX live pilot honest. This is a design and
research control plan, not a claim that the execution system is mature.

## Evidence Labels

- MEASURED: directly verified in code, tests, or current command output.
- INFERRED: supported by repo structure or docs, but not yet proven by a live
  execution record.
- UNSUPPORTED: plausible industry expectation or design risk that still needs
  official-doc, broker, or live-journal proof.

## Current Bottleneck

- MEASURED: ProjectX credentials exist in the canonical runtime `.env`, and the
  linked Codex worktree can now load that canonical `.env` without reading
  `.env.example`.
- MEASURED: telemetry maturity is advisory for the express/funded prop-firm
  profile; it is not the hard capital blocker for this profile.
- INFERRED: the biggest live bottleneck is context hygiene plus real
  execution/fill attribution. Readiness can be green while actual order, fill,
  account, copy, and session attribution remain unproven.
- MEASURED: new live journal rows now have fields for `profile_id`,
  `account_id`, `copy_id`, `runtime_session_id`, `contract_id`, `session_id`,
  `orb_minutes`, entry/exit broker order IDs, and entry/exit client order IDs.
  The session orchestrator populates these for signal-only rows and the current
  single primary broker copy.
- MEASURED: `scripts/tools/reconcile_live_fills.py` is a read-only gate that
  reconciles `live_journal.live_trades` to `data/broker_fills.jsonl` by
  `(account_id, order_id)` so overlapping order IDs across accounts do not
  merge.
- MEASURED: `scripts/tools/start_topstep_live_pilot.py` now includes a
  post-session operator gate: after a successful live runner exit it fetches
  TopstepX broker fills, then runs live-fill reconciliation scoped to the
  current trading day so stale historical/test rows do not mask the session
  under review.
- UNSUPPORTED: multi-copy scaling is not ready until every shadow account has
  independent journal rows, drawdown, HWM, order, fill, and reconciliation
  evidence.

## Missing Before Institutional Confidence

- MEASURED PARTIAL: independent identity fields now exist in the live journal
  for every `profile_id`, `account_id`, `copy_id`, `instrument`, `contract`,
  `strategy_id`, `session_id`, `trading_day`, `client_order_id`, and broker
  order id. Broker fill IDs are supported by the journal API but still require
  a broker fill poll/readback path that supplies them.
- MEASURED PARTIAL: fill reconciliation now proves journaled order IDs against
  fetched broker fills at `(account_id, order_id)` granularity. It does not yet
  fetch broker data, attach fill IDs during live runtime, or prove shadow-copy
  fills.
- UNSUPPORTED: per-shadow-account live journal rows. The current orchestrator
  journals the primary broker copy and signal-only records; demo/live
  multi-copy must remain blocked until shadow fills are independently persisted
  and reconciled.
- Broker-side readback after every execution path: order status, positions,
  account state, fills/trades, and rejected/cancelled order handling.
- Per-account and per-copy risk belts: daily loss, trailing drawdown / HWM,
  max contracts, lockout state, and prop-firm rule state.
- Official-doc parity checklist for authentication, connection URLs, rate
  limits, account search, contract search, order placement, order status,
  fills/trades, market/user hub behavior, token lifetime, and reconnect rules.
- Replayable audit artifact per live session: config snapshot, env-key
  presence only, profile fingerprint, lane allocation, broker account map,
  order/fill ledger, and post-session reconciliation summary.

## Do Not Reinvent

Use official broker/API docs and existing repo authorities before adding logic:

- ProjectX Gateway API auth: `POST /api/Auth/loginKey` with `userName` and
  `apiKey`, returning a session token.
  Source: https://gateway.docs.projectx.com/docs/getting-started/authenticate/authenticate-api-key/
- ProjectX connection URLs are firm/platform specific; TopstepX currently
  documents `https://api.topstepx.com`, `https://rtc.topstepx.com/hubs/user`,
  and `https://rtc.topstepx.com/hubs/market`.
  Source: https://gateway.docs.projectx.com/docs/getting-started/connection-urls/
- Topstep says the API key must be used with the TopstepX username, and that
  API access includes REST and WebSocket APIs, real-time market data,
  integration tools, docs, and dashboard management tools.
  Source: https://help.topstep.com/en/articles/11187768-topstepx-api-access

Repo authorities to keep using:

- `trading_app/prop_profiles.py` for prop-firm profile rules and copies.
- `trading_app/lifecycle_state.py` for C11/C12 state.
- `scripts/run_live_session.py` for live preflight gates.
- `trading_app/live/projectx/` for broker auth, contracts, router, positions,
  and feed adapters.
- `pipeline/paths.py` for DB/runtime artifact locations; env loading is now
  handled by `trading_app/live/env_bootstrap.py` to avoid linked-worktree
  `.env` drift.

## Continue / Stop Rule

Continue only as a single-copy pilot while:

- C11/C12 are valid for the active profile and lanes.
- Strict readiness has no blockers.
- Broker auth, contracts, bracket/fill probes, journal health, and pulse gates
  pass preflight.
- Telemetry advisory remains visible but is not misrepresented as a hard
  prop-firm blocker.
- Every trade is treated as independent across asset, session, account, and
  copy in calculation and attribution.
- Post-session reconciliation is run against freshly fetched broker fills and
  any `unmatched_*` or missing-account issue is treated as a stop-and-investigate
  item before the next live session.
- Full-history reconciliation may still surface old incomplete/test rows. Treat
  those separately from the current-session gate; do not let stale rows hide
  current-day attribution issues.

Do not continue to multi-copy or larger capital until the missing attribution
and per-copy risk gates above are measured, not inferred.
