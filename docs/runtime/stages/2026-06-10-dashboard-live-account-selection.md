---
task: STAGE 1 of multi-account control plane — single-account dashboard selector (Express/Combine) threaded through preflight + launch, PLUS no-menu startup. Dissolves live-launch blocker #13. (Stage 2 = per-account risk belts; Stage 3 = simultaneous independent multi-account. Both gated separately.)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - START_BOT.bat
  - tests/test_live/test_dashboard_account_selection.py
blast_radius: "bot_dashboard.py modifies the live-launch path (_live_pilot_cli_args signature plus its two callers, a new read-only /api/broker-accounts endpoint, an optional action_start param; no order-placement code touched). bot_dashboard.html adds UI only (dropdown plus LIVE gating). START_BOT.bat gets no-menu startup plus account-id synced to Express. Reads ProjectX Account search (enumeration only) and gold.db read-only via the preflight subprocess. No new gold.db writes. Per-file detail follows below."
---

## Staged architecture (operator decision 2026-06-11)

Operator expanded scope from a selector to a multi-account control plane, then
chose to STAGE it and ship the safe foundation first.

- **Stage 1 (THIS stage):** single-account dashboard selector (pick Express OR
  Combine per-launch) + no-menu startup (`.bat` boots SIGNAL+dashboard; mode &
  account chosen in UI). Dissolves blocker #13 now. ONE account live at a time —
  the existing single-account engine assumption holds, no new risk surface.
- **Stage 2 (future, capital-gated):** per-account daily-loss kill belts — the
  "per-shadow loss belt" `prop_profiles.py:761` flags as missing. HARD
  PREREQUISITE for Stage 3 (operator: "belts are a hard prerequisite").
- **Stage 3 (future, capital-gated):** simultaneous INDEPENDENT multi-account
  (Express + Combine, different lanes each, one orchestrator multiplexing N
  accounts per the best-practices research). Per-account C11, concentration cap.

Research grounding (official ProjectX docs, 2026-06-11): one token routes orders
to any accountId; rate-limit is per-CREDENTIAL (200/60s shared) → one process
multiplexing N accounts, NOT N processes. Full brief in session.

### GOVERNING CONSTRAINT — everything is an OPTION, nothing hardcoded (operator 2026-06-11)

Operator: "i want options right, i dont like hardcoding shit that i need to
change." Mirror-vs-independent is NOT a build-time fork — it is a RUNTIME option:

- Each account carries a **lane-set assignment** (UI-selectable).
- Same lane-set on both accounts = MIRROR. Different lane-sets = INDEPENDENT.
- ONE mechanism (account→lane-set mapping) expresses BOTH modes; no code fork.
- Research's "independent for Express+Combine" is the recommended DEFAULT, not a
  hardcoded pairing. Operator: mirror is "possibly, not always."

Applies to all selectable axes: account(s) active, mode (signal/demo/live),
lanes-per-account, mirror-vs-independent. The Stage 2/3 data model MUST express
these as config/UI levers, never baked-in pairings. (Sizing already follows this
doctrine: self-funded-sizing-doctrine.md — risk-first, never hardcoded caps.)

## Context

Capital-path change. The broker account the bot trades is hardcoded `23055112`
(50K Combine) in three places (`bot_dashboard.py:76`, `START_BOT.bat:137,187`).
Operator wants to pick Express (`21944866`) vs Combine at the dashboard with no
CLI/code edits. Permanently dissolves live-launch blocker #13. Plan:
`docs/plans/` (in-session). Front-end + back-end ship together
(doctrine_bot_changes_must_be_front_and_back_end).

## Pre-implementation gates (BOTH PASSED against live broker truth, 2026-06-11)

- **Gate #0 (FATAL-IF-FALSE) — PASS.** Live `resolve_all_account_ids()` with the
  current credential (`TopStepX (from .env)`, conn `env-projectx-88aeb4`)
  returned BOTH `(21944866, 'EXPRESS-V2-451890-53179846')` AND
  `(23055112, '50KTC-V2-451890-29512053')`. One credential spans both accounts —
  the design is viable (NOT a multi-credential feature). Names self-label
  `EXPRESS-V2` (funded) vs `50KTC-V2` (combine) — canonical label source.
- **Gate #1 (C11 per-account) — PASS for Express.** Profile
  `topstep_50k_mnq_auto` resolves `is_express_funded=True`, firm=topstep,
  size=50000 — i.e. the profile IS the Express/XFA tier. Current C11 lifecycle
  state: `valid=True gate_ok=True`, "operational 100.0%, as_of=2026-06-10,
  strict_account=PASS". So C11 was computed against the Express floor; routing to
  `21944866` runs on the account C11 validates.
  **Nuance (carried to UI):** C11 binds to the PROFILE, not a broker account-id.
  Combine (`23055112`) is a different product (TC starts at `account_size`; XFA
  starts at $0 broker equity → different trailing-DD anchor). So the selector is
  fully C11-matched for Express; **selecting Combine must surface a disclosure**
  that C11 was validated on the Express tier. Operator decision 2026-06-11:
  default = Express, warn on Combine.
- **Gate #2 (connected-state probe) — confirmed.** `list_connections()` rows
  carry `status` (`broker_connections.py:153`); `"connected"` is the live flag,
  already consumed by `_collect_broker_status:987` / `_connection_readiness:1007`.

## Approach

Single source of truth for "which account" = the FE selection, carried as ONE
`account_id` param through ONE backend builder (`_live_pilot_cli_args`) into BOTH
preflight and launch — parity by construction (wiring-contract row #6).

### Backend (`bot_dashboard.py`)
- `_live_pilot_cli_args(profile, account_id: int | None = None)` → emits
  `--account-id (account_id or LIVE_PILOT_ACCOUNT_ID)`. Single injection point.
- Callers updated to pass `account_id`: `_run_preflight_subprocess(profile, mode,
  account_id)` (:768), launch path in `action_start` (:2921),
  `_prepare_profile_for_start(profile, mode, account_id)` (:838).
- `action_start` (:2792) gains `account_id: int | None = None` query param.
  Validation stays downstream (`_select_primary_and_shadow_accounts:929` hard-fails
  on bad id — surfaced to UI, not re-implemented).
- NEW `GET /api/broker-accounts` (mirror `/api/broker/list:2276`): one
  `/api/Account/search` round-trip via `connection_manager.get_auth(conn_id)` →
  `ProjectXContracts(auth).resolve_all_account_ids()`, enriched from the SAME rows
  (balance/canTrade/isVisible — NOT N×`query_account_metadata`, which would be N+1
  identical calls). 3-state, NO silent fallback:
  - `connecting` (broker auth not yet complete) → `{accounts:[], state, ...}`,
    no `default_id`, LIVE blocked.
  - `auth_failure` → explicit error, LIVE blocked.
  - `ok` → real accounts + `default_id=21944866` (Express).
  ~5min module-level cache. Default `LIVE_PILOT_ACCOUNT_ID` stays as the
  zero-arg fallback (comment repointed to the selector; NOT deleted).
- Handoff-restart (`_set_handoff`/`_handoff_state`:570) threads `account_id` so an
  auto-restart leg uses the chosen account, not the default (finding #3).
- Running-state visibility: chosen `account_id` persists so `/api/status` /
  bot_state surface the live account (watch-out #9).

### Front-end (`bot_dashboard.html`)
- Account `<select>` near LIVE control, populated from `/api/broker-accounts`,
  default Express, Combine shows C11-tier disclosure banner, LIVE blocked until
  `state==ok`, `account_id` appended to launch AND preflight POSTs.

### `START_BOT.bat`
- Reconcile `:137`/`:187` so the `.bat` boot account-id does not contradict the
  dashboard selection (default Express; comment synced to selector).

## Blast Radius

- `trading_app/live/bot_dashboard.py` — MODIFIES live-launch path. `_live_pilot_cli_args` signature change (2 callers: `_run_preflight_subprocess`, `action_start` launch). NEW `/api/broker-accounts` endpoint (additive, read-only broker enumeration). `action_start` gains optional param (backward-compat: None → existing default). Reads `connection_manager` (already loaded at startup :144). NO order-placement code touched.
- `trading_app/live/bot_dashboard.html` — additive UI (dropdown + banner + 2 POST param appends). No existing control removed.
- `START_BOT.bat` — reconciles hardcoded account-id at 2 lines; preflight↔launch parity preserved (both must agree or engine check [13] fails).
- `trading_app/live/projectx/contract_resolver.py`, `positions.py` — READ-ONLY reuse (`resolve_all_account_ids`, account metadata fields). No edits.
- `trading_app/live/preflight.py` `_select_primary_and_shadow_accounts:929` — UNCHANGED; relied on as the hard-fail validation backstop.
- `scripts/run_live_session.py` — UNCHANGED; already accepts `--account-id` (argparse:278, default None).
- Reads: ProjectX `/api/Account/search` (enumeration only, no orders); gold.db via existing preflight subprocess (read-only). Writes: none new to gold.db; bot_state/handoff in-process state only.
- Tests: NEW `tests/test_dashboard/test_broker_account_selector.py`. Reuses existing `tests/test_scripts/test_run_live_session_account_selection.py` as validation backstop.

## Adversarial-audit gate

MANDATORY post-implementation `evidence-auditor` pass on the live-path commit
before "done" (`.claude/rules/adversarial-audit-gate.md`) — this commit is
[judgment] + touches `trading_app/live/` + capital-path. Verdict PASS required,
or findings closed.

## Done criteria

Tests pass (show output) + dead code swept (`grep -r`) + `check_drift.py` passes
+ self-review + evidence-auditor gate PASS.
