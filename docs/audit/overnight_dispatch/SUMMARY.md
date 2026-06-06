# Overnight Capital Code-Review — SUMMARY

- **date:** 2026-06-06
- **scope:** 10 capital-at-risk code paths, each reviewed by a fresh-context
  live-risk-auditor subagent dispatched from a lean /loop session (state 100% on disk).
- **method:** adversarial confirm/refute of named findings + open review; grounded in
  quoted code. Full verdicts in `results/01.md`..`results/10.md`.

## Verdict table

| id | target | verdict | severity |
|----|--------|---------|----------|
| 01 | session_orchestrator.py:2780-2821 bracket-leg-missing | CONFIRMED | **HIGH — naked position** |
| 02 | contract_resolver.py resolve_account_id | PARTIAL | MED — wrong-account if 2+ accounts |
| 03 | account_survival.py:407/636 hardcoded 1-micro | CONFIRMED (dormant) | **P1 — blocks clamp-lift** |
| 04 | account_survival.py:220 fingerprint 2-file | CONFIRMED | **P1 — stale PASS (DSR-class)** |
| 05 | prop_profiles.py routing | PARTIAL (clean) | MED — no import-time MLL guard |
| 06 | run_live_session.py preflight chain | CONFIRMED | **HIGH — preflight NOT run on --live** |
| 07 | live sizing path (engine half D-3) | PARTIAL | clamp sound today; blocks clamp-lift |
| 08 | broker order/fill/reconnect | CONFIRMED | HIGH (=01) + 2 NEW MED |
| 09 | kill-switch / emergency-flatten | PARTIAL | flatten real; 2 MED silent-exposure gaps |
| 10 | cost_model.py / asset_configs.py specs | CLEAR | none — CME-correct, single source |

## The headline findings (do NOT arm live until addressed)

1. **[HIGH, row 06] Preflight gates are not on the arm path.** The 14-gate chain
   (C11/C12 survival, drift, pulse) runs ONLY under `--preflight`, a terminal
   `sys.exit` mode. The `--live` arm path never calls `_run_preflight`.
   **START_BOT.bat:52-53 falsely claims** the full preflight runs before live, but
   `:102` invokes `--live` with no `--preflight`. The operator-facing front-end
   asserts a safety guarantee the code does not deliver. *This is the single most
   important finding — it means every "preflight 14/14" result is decoupled from the
   actual arm.* Fix: enforce gates in-process on `--live` (front-ends can't be trusted to chain).

2. **[HIGH, rows 01 + 08] Naked-position on bracket-leg-missing.** When a native
   bracket's protective leg is unconfirmed, the code only alerts + stores partial IDs
   and continues — no kill/flatten. The F4 non-native path DOES flatten on the same
   failure class → asymmetric. Row 09 confirms `_emergency_flatten` genuinely closes at
   the broker, so the fix (call kill+flatten, mirroring F4) is sound.

3. **[P1, rows 03 + 04] C11 survival gate is fragile in two ways**, both dormant
   today but live blockers on the memory clamp-lift income plan: (a) DD hardcoded to
   1 micro, blind to max_contracts; (b) fingerprint covers only 2 files, so a stale
   PASS survives drift in prop_profiles/cost_model/scaling_plan (same bug class as the
   already-fixed DSR `_dsr_policy.py`). Both should be fixed before any clamp lift.

## Secondary (fix alongside the above)
- **Row 02:** bind `account_id` into AccountProfile; verify the API key has exactly 1 account before arming (else `accounts[0]` may route to the wrong account).
- **Row 08:** partial-fill bracket may size to intended not filled qty (verify ProjectX AutoBracket semantics); reconnect loop never re-syncs positions from broker truth.
- **Row 09:** `on_exit_filled` marks FLAT on submit not fill; no-event-loop flatten path silently drops the order.
- **Row 05:** add a drift check running `evaluate_profile_survival(write_state=False)` on all active profiles.

## Clean
- **Row 10:** cost/contract specs are CME-correct, friction conservative, single-sourced. The "wrong number mis-sizes gate" risk does not materialize.
- **Row 05:** is_express_funded classifier, self-funded de-leak, and None-default safety all fail closed.

## Cross-cutting theme
The unifying defect (consistent with `project_codex_3x_capital_review_findings_2026_06_06.md`):
**safety checks are decoupled from live behavior** — preflight runs in a separate mode
from arming (06), the survival gate's sizing assumption is decoupled from the engine's
(03/07), the gate's fingerprint is decoupled from the code it should track (04), and
kill-switch is decoupled from flatten (09). The fixes are all "re-couple the guard to
the thing it guards," not new features.

## Status
All 10 rows DONE. No production code was edited (this loop audits; capital fixes are
operator-gated Tier B). Recommended fix order: **06 → 01/08 → 03/04 → 02 → 09 → 05**.
