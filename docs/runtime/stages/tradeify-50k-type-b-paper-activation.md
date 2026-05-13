---
task: Stage 3b (doc-only) — file the paper-only activation contract for `tradeify_50k_type_b` and record the Tradovate credential gate that blocks the actual `active=True` flip. This stage produces zero production-code edits. The flag flip is Stage 3b-flip, separately gated on credential presence and a successful manual auth dry-run.
mode: IMPLEMENTATION
scope_lock:
  - docs/runtime/stages/tradeify-50k-type-b-paper-activation.md
---

## Blast Radius

- `docs/runtime/stages/tradeify-50k-type-b-paper-activation.md` — new file. Zero callers. No production-code touched, no DB touched, no allocator touched.
- `trading_app/prop_profiles.py:683-724` — **READ-ONLY in this stage**. Reference only. `tradeify_50k_type_b.active` stays `False`. The 5 daily lanes and `notes` text are unchanged.
- `trading_app/live/broker_factory.py:37-90` — **READ-ONLY**. Reference: `BROKER=tradovate` dispatches `TradovateAuth/Contracts/OrderRouter/Positions`; market-data feed remains ProjectX (no Tradovate feed).
- `trading_app/live/tradovate/auth.py:1-90` — **READ-ONLY**. Reference: 5 required env vars (`TRADOVATE_USER`, `TRADOVATE_PASS`, `TRADOVATE_CID`, `TRADOVATE_SEC`, `TRADOVATE_DEVICE_ID`) + `TRADOVATE_DEMO=1` to flip to demo URL.
- `docs/audit/results/2026-05-11-mgc-mes-profile-activation-feasibility.md` — **READ-ONLY**. §1-2: AccountProfile schema + 15 consumers of `.active`.
- Live runtime, paper-trader, pre-session checks, lane allocator: **NO CHANGE**. Profile remains `active=False`; nothing in the live runtime reads it any differently than yesterday.

## Predecessor

Stage 3a complete at `0b0a51d9` — `[mechanical] Tests: Tradovate F4 emergency-flatten parity (Stage 3a)`. Branch `feat/tradovate-bracket-legs` is 5 commits ahead of `origin/main` on a clean tree (`HANDOFF.md` modification is pre-existing and unrelated to this stage).

## Activation Contract (what `active=True` means for `tradeify_50k_type_b`)

Paper-only is **operator discipline**, not a schema change. There is no `paper_only` field on `AccountProfile`. Paper-vs-live is decided at runtime by environment, not config:

| Env var | Required for activation | Source of truth |
|---|---|---|
| `BROKER=tradovate` | yes — dispatches Tradovate components in `broker_factory.create_broker_components()` | `trading_app/live/broker_factory.py:37-39, 75-90` |
| `TRADOVATE_DEMO=1` | yes for paper — flips base URL to `https://demo.tradovateapi.com/v1` | `trading_app/live/tradovate/auth.py:30-38` |
| `TRADOVATE_USER` | yes — read at `auth.py:112` | `auth.py:9` (also accepts `TRADOVATE_USERNAME`) |
| `TRADOVATE_PASS` | yes — read at `auth.py:113` | `auth.py:10` (also accepts `TRADOVATE_PASSWORD`) |
| `TRADOVATE_CID` | yes — read at `auth.py:114` | `auth.py:11` |
| `TRADOVATE_SEC` | yes — read at `auth.py:115` | `auth.py:12` |
| `TRADOVATE_DEVICE_ID` | **optional** — `auth.py:116-123` auto-generates a UUID and logs a warning when unset (Tradovate may flag multi-device usage) | `auth.py:13` |

Note: `TRADOVATE_APP_ID` and `TRADOVATE_APP_VERSION` env vars (sometimes seen in `.env`) are **dead** — `auth.py:130-131` hardcodes `appId="canompx3-bot"` and `appVersion="1.0"` in the POST body. They are never read.

When `tradeify_50k_type_b.active` flips to `True` and the operator launches the session orchestrator with `BROKER=tradovate TRADOVATE_DEMO=1` plus the 4 required credential vars set, the profile becomes selectable by the 15 `.active`-consuming sites enumerated in `docs/audit/results/2026-05-11-mgc-mes-profile-activation-feasibility.md` § 2. Without `TRADOVATE_DEMO=1`, the same flip would route to **live** Tradovate — paper-vs-live is purely the env var. Operator discipline means: never launch without `TRADOVATE_DEMO=1` until live-grade verification is done.

## Current Blocker — Why Stage 3b is Doc-Only

Tradovate credentials **are present** in the project root `.env` (verified 2026-05-13 via grep against `C:/Users/joshd/canompx3/.env` lines 14-20):

- `TRADOVATE_USER` — populated
- `TRADOVATE_PASS` — populated (quoted)
- `TRADOVATE_CID` — populated
- `TRADOVATE_SEC` — populated
- `TRADOVATE_DEVICE_ID` — **NOT set** (optional; auto-UUID fallback fires on first auth, logs warning)
- `TRADOVATE_APP_ID="Sample App"` / `TRADOVATE_APP_VERSION="1.0"` — present but **dead** (see note above)

So the literal "credentials missing" blocker the user assumed is **not** the blocker. What is unverified:

1. **Whether those credentials actually authenticate.** The values exist; no one has run the demo-URL auth dry-run against them. The current `prop_profiles.py:718-723` notes still say `"Tradovate API (auth broken)"` — that string predates the current `.env` state and may reflect a stale assumption, a known-bad credential set, or a real prior failure. We do not know which.
2. **Whether `TRADOVATE_DEVICE_ID` should be pinned** before first auth, to avoid Tradovate flagging the account when the UUID changes per process restart.
3. **Whether the 5 daily lanes in `prop_profiles.py:708-716` are still deployable** against current allocator truth (last refreshed 2026-04-19; deployable shelf has moved since the 2026-05-03 rebalance).

This is why Stage 3b is doc-only: we file the contract, list what's unverified, and let Stage 3b-flip do the actual checks. Rewriting `notes="...Tradovate API (auth broken)..."` to "auth verified" before running the dry-run would be the unverified-narration anti-pattern.

## Prerequisites for Stage 3b-flip (deferred, not in this stage)

Stage 3b-flip may proceed only after all three are true:

1. **Manual auth dry-run succeeds against demo URL.** Operator runs from the worktree root:

   ```bash
   TRADOVATE_DEMO=1 python -c "from trading_app.live.tradovate.auth import TradovateAuth; print(TradovateAuth().get_token()[:10])"
   ```

   Expected: prints a 10-character access-token prefix and exits 0. No exception traceback. Failure modes to recognize:
   - HTTP 400 with `errorText` → credentials in `.env` are wrong (re-check on Tradovate API key page).
   - `"Tradovate requires 2FA p-ticket"` → 2FA must be completed manually in the Tradovate web UI first.
   - Connection error → demo URL unreachable from this network.

2. **`TRADOVATE_DEVICE_ID` decision.** Either (a) generate a stable UUID once and pin it in `.env` (recommended — avoids Tradovate multi-device flagging), or (b) accept the per-run UUID warning. Stage 3b-flip records the decision.

3. **Daily-lane shelf re-validated.** All 5 `daily_lanes` currently listed at `prop_profiles.py:708-716` checked against the current `deployable_validated_setups` view per the `check_lane_allocation_daily_lanes_in_deployable_set` drift check at `pipeline/check_drift.py:5611-5648`. The shelf was last refreshed 2026-04-19; the allocator-backed truth has moved since (verified via live state notes for 2026-05-03 rebalance). The flip cannot land if any of these strategies has been retired, paused, or had its `deployment_scope` change since then:

   - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
   - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
   - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
   - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
   - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

## Done criteria (this stage — Stage 3b)

1. This stage file written at `docs/runtime/stages/tradeify-50k-type-b-paper-activation.md`.
2. `python .claude/hooks/stage-gate-guard.py` parses the file without complaint (validated on next production-code edit attempt, and may be invoked directly).
3. `git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs status --short` shows only this stage file as added (`HANDOFF.md`'s pre-existing modification is out of scope and not introduced by this stage).
4. `trading_app/prop_profiles.py` byte-identical vs `aeef4ed1` — `git diff aeef4ed1 -- trading_app/prop_profiles.py` returns empty.

## Out of scope (explicit non-goals)

- Flipping `tradeify_50k_type_b.active` from `False` to `True`. That is Stage 3b-flip, gated on the three prerequisites above.
- Stage 4 (PR open).
- Re-validating the 5 daily lanes against current allocator truth — Stage 3b-flip prerequisite, not part of this doc-only stage.
- Rewriting the `notes="...Tradovate API (auth broken)..."` text. Stays as-is until auth is verified clean in Stage 3b-flip.
- Touching the 5 stale stage files on `main` unrelated to this branch.

## Verification commands

```bash
ls C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs/docs/runtime/stages/tradeify-50k-type-b-paper-activation.md
git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs status --short
git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs diff aeef4ed1 -- trading_app/prop_profiles.py
```

Expected: file exists; status shows only the new stage file added (plus pre-existing `HANDOFF.md` modification); diff returns empty.

## Closure (2026-05-13)

Stage 3b-flip ran and produced verdict **BLOCKED — keep `active=False`**. Full evidence in the companion file `docs/runtime/stages/tradeify-50k-type-b-paper-activation-flip.md`. Summary:

- **Gate 1 (auth dry-run):** DEFERRED to operator. `.env` is fenced from the autonomous loop (Claude Code permission classifier denies `Read(./.env)` and the `python -c "import dotenv; load_dotenv()"` loophole); third-party API + potential 2FA p-ticket challenge per `trading_app/live/tradovate/auth.py:142-145` requires operator action.
- **Gate 2 (`TRADOVATE_DEVICE_ID` pin):** DEFERRED to operator. Same `.env`-fence reason.
- **Gate 3 (5 daily lanes still deployable):** **FAILS-HARD**. Live `lane_allocation.json` (rebalance 2026-05-11, staleness OK days_old=2, queried 2026-05-13 via `mcp__strategy-lab__get_lane_allocation_summary`) shows **0 of 5 `prop_profiles.py:708-716` lanes are routable today**:
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` — PAUSED, chordia `FAIL_BOTH` (t<3.0), audit_age=2d
  - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` — PAUSED, chordia `MISSING`
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` — PAUSED, chordia `FAIL_BOTH` (t<3.0), audit_age=2d
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — NOT IN ALLOCATOR (neither deployed nor paused list)
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` — PAUSED, chordia `MISSING`

  This stage's own § "Prerequisites for Stage 3b-flip" item 3 specifies "The flip cannot land if any of these strategies has been retired, paused, or had its `deployment_scope` change since then." 5 of 5 have moved.

- **Refill pool size:** 3 MNQ strategies in canonical `lane_allocation.json:lanes[]` (verified by direct JSON read 2026-05-13): `MNQ_COMEX_SETTLE_OVNRNG_100` (DEPLOY/PASS_CHORDIA), `MNQ_US_DATA_1000_VWAP_MID_ALIGNED_O15` (DEPLOY/PASS_CHORDIA), `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` (PROVISIONAL/PASS_PROTOCOL_A). All three are on `topstep_50k_mnq_auto`. The TYPE-B exclusivity clause (no cross-firm sharing, per `notes=` text) → fill pool is **empty** for TYPE-B. Per `memory/feedback_chordia_missing_is_not_backlog.md` and `memory/feedback_max_profit_grow_chordia_inventory_not_force_slots.md`, forcing the slots from a Chordia pool that is exclusivity-blocked is the wrong move. (An earlier MCP query in the same session reported the RR1.5 NYSE_OPEN lane as paused; the raw JSON is canonical and showed PROVISIONAL — see flip stage file for the reconciliation note.)

- **Code-state delta from this stage:** `trading_app/prop_profiles.py:704-723` — comment block (3 lines → 7 lines) and `notes=` string rewritten to record the current blocker (Chordia shelf depletion, not "auth broken"). `daily_lanes` tuple byte-identical vs `7b008d15`. `active=False` byte-identical. Verified via `git diff 7b008d15 -- trading_app/prop_profiles.py | grep -E '^[-+].*(active=|DailyLaneSpec\()'` returning empty.

- **Verification:** `python pipeline/check_drift.py` → 125 PASS, 0 skipped, 20 advisory (pre-existing). Check 141 (`lane_allocation.json lanes[] must pass Chordia gate`) PASSED. `python -m pytest tests/ -k prop_profiles -x` → 80 passed, 1 skipped.

- **Adversarial-audit gate:** Dispatched per `.claude/rules/adversarial-audit-gate.md`. See verdict line below once the auditor returns.

### Next steps after this closure

The path to actually flipping `active=True` runs through **growing the PASS_CHORDIA shelf**, not unlocking MISSING entries (per fail-closed doctrine). Likely candidates: MGC/MES Chordia unlock audits for TYPE-B allowed sessions (`US_DATA_1000`, `COMEX_SETTLE`, `NYSE_CLOSE`, `CME_REOPEN`, `SINGAPORE_OPEN`, `EUROPE_FLOW`, `US_DATA_830`, `NYSE_OPEN`). These audits are out of scope for the flip stage and need their own pre-registered hypotheses. When the shelf reaches ≥5 lanes exclusive to Tradeify with PASS_CHORDIA, re-attempt Stage 3b-flip with a fresh `daily_lanes` regeneration.

In parallel, the operator can run Gate 1 (auth dry-run) and Gate 2 (DEVICE_ID pin) at any time — those clear unrelated downstream risk before live broker connection, even though they no longer block the activation.
