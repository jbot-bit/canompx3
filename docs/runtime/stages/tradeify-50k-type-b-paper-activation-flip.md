---
task: Stage 3b-flip — execute the `tradeify_50k_type_b.active=False -> True` flip per the Stage 3b activation contract. This stage either flips the flag (all 3 gates pass) or records a BLOCKED verdict (any gate fails) and refreshes the stale `notes` text to match current truth. No `daily_lanes` regeneration in this stage — see "Out of scope" below.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/prop_profiles.py
  - docs/runtime/stages/tradeify-50k-type-b-paper-activation-flip.md
  - docs/runtime/stages/tradeify-50k-type-b-paper-activation.md
  - HANDOFF.md
---

## Blast Radius

- `trading_app/prop_profiles.py:683-724` — narrow edit window. `active` boolean NOT flipped this stage (Gate 3 fails — see Verdict). `notes` text at lines 718-723 rewritten to record current verified blocker. `daily_lanes` tuple at lines 708-716 UNCHANGED.
- `docs/runtime/stages/tradeify-50k-type-b-paper-activation-flip.md` — this new stage file.
- `docs/runtime/stages/tradeify-50k-type-b-paper-activation.md` — append Closure section recording Stage 3b-flip verdict.
- `HANDOFF.md` — update Last Session pointer.
- Live runtime, paper-trader, lane allocator, session orchestrator, kill-switch: **NO CHANGE** — `active` stays `False`, so the 15 consumers of `.active` enumerated in `docs/audit/results/2026-05-11-mgc-mes-profile-activation-feasibility.md` § 2 continue to filter `tradeify_50k_type_b` out of routing. The only file-state delta inside `trading_app/` is a `notes=` string literal — semantically dead text consumed by no runtime code path.

## Predecessor

Stage 3b complete at `7b008d15` — `docs(stage-3b): file paper-activation contract + credential-gate verification plan (doc-only)`. HANDOFF stamp at `74553f87`. Branch `feat/tradovate-bracket-legs`, 7 commits ahead of `origin/main` after this stage's commits land.

## Gate Verification (live truth as of 2026-05-13)

### Gate 1 — Manual auth dry-run against demo URL: **DEFERRED to operator**

Stage 3b contract § "Prerequisites for Stage 3b-flip" specifies the operator runs:

```bash
TRADOVATE_DEMO=1 python -c "from trading_app.live.tradovate.auth import TradovateAuth; print(TradovateAuth().get_token()[:10])"
```

This stage does not invoke that command. Rationale: (a) third-party API touch; (b) any 2FA p-ticket challenge per `trading_app/live/tradovate/auth.py:142-145` requires manual completion in the Tradovate web UI — outside the autonomous-loop's reach; (c) per the no-auto-run-of-cred-dependent-broker-calls posture in this worktree, the `.env` read path is fenced (Claude Code permission classifier denies `Read(./.env)` and the `python -c "import dotenv; load_dotenv()"` loophole alongside it). The operator runs Gate 1 once, reports pass/fail, and the stage Closure section records the result. **Gate 1 status: NOT BLOCKING THIS STAGE** — independent of Gates 2/3 because flip cannot proceed anyway (see Gate 3).

### Gate 2 — `TRADOVATE_DEVICE_ID` decision: **DEFERRED to operator**

Per the Stage 3b plan, `auth.py:117-123` falls back to a fresh per-process UUID when the env var is unset. Recommended action: pin a stable UUID via `python -c "import uuid; print(uuid.uuid4())"` written to `.env` as `TRADOVATE_DEVICE_ID=<uuid>`. Operator-owned (`.env` is fenced from the autonomous loop). **Gate 2 status: NOT BLOCKING THIS STAGE** for the same reason as Gate 1.

### Gate 3 — 5 daily lanes still deployable: **FAILS-HARD**

Cross-checked against `docs/runtime/lane_allocation.json` (rebalance `2026-05-11`, profile `topstep_50k_mnq_auto`, staleness OK, `days_old=2`) via `mcp__strategy-lab__get_lane_allocation_summary` on 2026-05-13:

| Strategy (`prop_profiles.py:708-716`) | Live allocator state | Chordia verdict |
|---|---|---|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | PAUSED | `FAIL_BOTH` (t<3.0), audit_age=2d |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | PAUSED | `MISSING` |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | PAUSED | `FAIL_BOTH` (t<3.0), audit_age=2d |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | NOT IN ALLOCATOR (neither deployed nor paused list) | unknown |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | PAUSED | `MISSING` |

**0 of 5 lanes are currently routable.** Per direct read of the canonical `docs/runtime/lane_allocation.json` (verified 2026-05-13), `lanes[]` carries exactly 3 entries — all on `topstep_50k_mnq_auto`:

- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` (status=DEPLOY, chordia=PASS_CHORDIA, age=1d)
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (status=DEPLOY, chordia=PASS_CHORDIA, age=1d)
- `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` (status=PROVISIONAL, chordia=PASS_PROTOCOL_A, age=4d)

All three are deployed/provisional on `topstep_50k_mnq_auto`. The TYPE-B exclusivity clause (cross-firm sharing forbidden) lives in `notes=` ("Bot must be exclusive to Tradeify (no cross-firm sharing)") → these 3 lanes are not available to populate `tradeify_50k_type_b`.

**Reconciliation note:** an earlier `mcp__strategy-lab__get_lane_allocation_summary` reply in this session reported the RR1.5 NYSE_OPEN lane in `paused[]` with reason "MECHANISM_FALSIFIED + ALARM_STILL_LIVE ... fail-closed pause per 2026-05-12 SR-alarm diagnostic". The raw JSON file is the canonical truth per `.claude/rules/integrity-guardian.md` § 2 — and it lists the lane as PROVISIONAL in `lanes[]`, not paused. The MCP-vs-JSON divergence appears to be a derived-view discrepancy in the strategy-lab MCP and is worth a separate audit, but does not affect Gate 3's outcome: none of the 3 entries in `lanes[]` is one of the 5 `prop_profiles.py:708-716` `daily_lanes` strategies, so the refill pool for TYPE-B remains empty under exclusivity.

**Stage 3b § "Prerequisites for Stage 3b-flip" item 3 explicitly states:** "The flip cannot land if any of these strategies has been retired, paused, or had its `deployment_scope` change since then." 5 of 5 have moved. Gate 3 fails by the stage contract's own terms.

## Verdict: BLOCKED — keep `active=False`

The Stage 3b activation contract permits the `active=True` flip only when all three gates pass. Gate 3 fails by the contract's explicit "any of these strategies has been retired, paused, or had its `deployment_scope` change" trigger. Per `memory/feedback_chordia_missing_is_not_backlog.md`: **Chordia MISSING is a fail-closed verdict, not a backlog to be unlocked.** Per `memory/feedback_max_profit_grow_chordia_inventory_not_force_slots.md`: **filling empty slots from a depleted Chordia pool by absolute-threshold sibling is the exact anti-pattern that was caught on 2026-05-12.**

Three resolution paths exist; only Path I is in-scope for this stage:

- **Path I — record BLOCKED, refresh stale `notes`, defer flip until shelf is replenished.** This is the institutional-rigor-compliant default.
- **Path II — widen TYPE-B to MGC/MES Chordia audits.** Out of scope (matches Stage 3b "Out of scope" item: "Re-validating the 5 daily lanes against current allocator truth — Stage 3b-flip prerequisite, not part of this doc-only stage" — implicitly accepts that the result may be BLOCKED).
- **Path III — Option A strict-reading flip with stale lanes.** Rejected: would write a known-wrong `active=True` state into a real-broker-routable profile (`TRADOVATE_DEMO=1` is operator discipline, not a schema-level safety belt; `auth.py:30-38` flips the URL but nothing prevents an operator from launching live with the flag set). Violates institutional-rigor rule 8 ("Verify before claiming") and rule 6 ("No silent failures" — `active=True` would claim routability that the allocator denies).

## Code edit (this stage)

Single file, single field, semantically inert change to documentation text:

```diff
--- a/trading_app/prop_profiles.py
+++ b/trading_app/prop_profiles.py
@@ -704,7 +704,11 @@
-        # Rebuilt from current allocator-backed deployable shelf on 2026-04-19.
-        # Prior config had 8 ghost lanes and 1 valid incumbent displaced by the
-        # current liveness-aware allocator. Keep inactive until explicit account
-        # activation review.
+        # daily_lanes rebuilt 2026-04-19 from allocator-backed shelf. Shelf has
+        # moved since (verified 2026-05-13): 0 of 5 lanes are currently routable
+        # — 3 paused (Chordia FAIL_BOTH/MISSING), 1 absent from allocator,
+        # 1 paused 2026-05-12 SR-alarm. Do NOT refresh from MNQ Chordia-PASS
+        # pool (size=2, both on topstep_50k_mnq_auto, exclusivity-blocked).
+        # Activation requires growing PASS_CHORDIA shelf to >=5 lanes exclusive
+        # to Tradeify (likely via MGC/MES audits — out of scope for this flip).
@@ -718,9 +722,11 @@
-            "TYPE-B auto inactive profile rebuilt 2026-04-19 from current allocator-backed shelf. "
-            "Current recommendation = 5 lanes, MNQ-led. Tradovate API (auth broken). "
-            "Bot must be exclusive to Tradeify (no cross-firm sharing). "
-            "Keep inactive pending explicit activation review."
+            "TYPE-B auto profile. Active=False pending Chordia-PASS shelf depth >= 5 "
+            "lanes exclusive to Tradeify (verified 2026-05-13: 0 of 5 listed daily_lanes "
+            "currently routable). Tradovate credentials present in .env; demo-URL auth "
+            "dry-run pending operator. Bot must be exclusive to Tradeify (no cross-firm "
+            "sharing). Re-evaluate after MGC/MES Chordia unlock audits expand the "
+            "deployable shelf for TYPE-B allowed sessions."
```

**`daily_lanes` tuple is byte-identical** vs `7b008d15`. **`active` boolean is byte-identical** (`False`). Only the comment block at 704-707 and the `notes=` string at 718-723 change.

Verification of byte-identicality on `daily_lanes` and `active`:

```bash
git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs diff 7b008d15 -- trading_app/prop_profiles.py | grep -E '^[-+].*(active|DailyLaneSpec)'
```

Expected: empty (no `active=` or `DailyLaneSpec(...)` lines in the diff).

## Done criteria (this stage)

1. `trading_app/prop_profiles.py:704-723` updated per the diff above. `active=False` and `daily_lanes` tuple byte-identical vs `7b008d15`.
2. `docs/runtime/stages/tradeify-50k-type-b-paper-activation.md` has a Closure section appended recording: gate-1 deferred (operator), gate-2 deferred (operator), gate-3 BLOCKED-shelf-depleted, evidence-auditor verdict (after dispatch).
3. `python pipeline/check_drift.py` exits 0. Specifically `check_lane_allocation_daily_lanes_in_deployable_set` must pass — `daily_lanes` unchanged, so this check's behavior is identical to its pre-stage state.
4. `python -m pytest tests/ -k prop_profiles -x` exits 0. (Profile-loading tests; `active` boolean unchanged so any `.active`-conditional path is unaffected.)
5. Adversarial-audit gate (`evidence-auditor` subagent) dispatched on the resulting commit per `.claude/rules/adversarial-audit-gate.md`. Verdict PASS recorded in Closure section.

## Out of scope (explicit non-goals)

- Flipping `tradeify_50k_type_b.active` from `False` to `True`. Gate 3 fails — see Verdict.
- Regenerating `daily_lanes` from the current allocator + ranked Chordia-PASS siblings. Pool is size-2, both already deployed on `topstep_50k_mnq_auto`, exclusivity-blocked. Per `feedback_max_profit_grow_chordia_inventory_not_force_slots.md`, forcing fill from depleted pool is the wrong move.
- Running Chordia unlock audits on MGC/MES candidates. Out of scope per the original Stage 3b plan ("Auto-batch-run Chordia audits ... out of scope for the flip").
- Pinning `TRADOVATE_DEVICE_ID` or running the Gate 1 auth dry-run. Operator-owned; `.env` is fenced.
- `tradeify_100k_type_b` (line 726+). Out of scope per Stage 3b.
- Stage 4 (PR open). Separate stage.

## Verification commands

```bash
# byte-identicality on the parts that must not change:
git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs diff 7b008d15 -- trading_app/prop_profiles.py | grep -E '^[-+].*(active|DailyLaneSpec)'
# expected: empty

# drift check (must pass — daily_lanes unchanged so check 23/lane-deployability check holds):
python pipeline/check_drift.py

# profile-loading tests:
python -m pytest tests/ -k prop_profiles -x

# scope check before commit:
git -C C:/Users/joshd/canompx3/.worktrees/tradovate-bracket-legs status --short
# expected: only the 4 scope_lock files modified
```
