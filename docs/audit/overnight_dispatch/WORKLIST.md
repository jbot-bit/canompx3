# Overnight Capital Code-Review — Dispatcher Worklist

Each row is ONE fresh-context subagent review job. The dispatcher (a /loop in a
lean main session) picks the next `PENDING` row every 15m, spawns ONE subagent to
review that target adversarially for capital-wreck bugs, writes the verdict to
`results/<id>.md`, and flips the row to `DONE`. State lives 100% on disk so the
dispatcher session never grows.

**Focus:** capital-at-risk paths only (live trading, brokerage, sizing, survival
gates, kill/flatten, account routing, bracket legs). Not style. Wreck-the-account
bugs, silent failures, gate-vs-live decoupling, stale-PASS, relaxed capital gates.

**Priority order:** P0 first (already-found CRITICALs get a confirm pass), then the
canonical capital modules, then the broader live surface.

| id | priority | target | focus | status |
|----|----------|--------|-------|--------|
| 01 | P0 | trading_app/live/session_orchestrator.py:2787-2821 | B[P0] bracket-leg-missing only ALERTS, never kill/flatten → naked position. Confirm + propose fix. | DONE |
| 02 | P0 | trading_app/live/projectx/contract_resolver.py + trading_app/live/broker_base.py (resolve_account_id) | A[P0] returns accounts[0] → orders route to wrong account, voids C11 proof. Confirm + fix. (memory path was stale — real defs here.) | DONE |
| 03 | P1 | trading_app/account_survival.py:407 | C[P1] C11 DD hardcoded 1-micro → blocks clamp-lift; gate lies about DD when max_contracts raised. | DONE |
| 04 | P1 | trading_app/account_survival.py:220 | D[P1] C11 fingerprints only 2 files → stale PASS survives live-code drift (DSR-class). | DONE |
| 05 | P1 | trading_app/prop_profiles.py | is_express_funded classifier, lane caps, self-funded de-leak, telemetry waiver keying. Capital-routing correctness. | DONE |
| 06 | P1 | scripts/run_live_session.py | Live preflight chain (14 gates): order, fail-closed semantics, any gate that warns where it should fail. | DONE — CONFIRMED HIGH (preflight not run on --live; START_BOT comment falsely claims it is) |
| 07 | P1 | trading_app/live/session_orchestrator.py (sizing path) | Vol-scaled sizing vs max_contracts=1 clamp; D-3 seam (gate projects 1-micro DD, engine sizes from equity). | DONE |
| 08 | P2 | trading_app/live/ (broker/execution engine) | Order placement, fill polling, partial fills, reconnect/idempotency on capital orders. | DONE |
| 09 | P2 | trading_app/live/ (kill-switch / flatten) | Kill/flatten control: does it actually flatten? any path that no-ops or only logs? | DONE |
| 10 | P2 | pipeline/cost_model.py + asset_configs.py | Cost/contract specs feeding survival math — wrong number here silently mis-sizes the gate. | DONE |

## Legend
- PENDING — not yet reviewed
- INPROGRESS — a subagent is currently reviewing (set on dispatch, cleared on write)
- DONE — verdict written to `results/<id>.md`
- BLOCKED — target file missing or path stale; note in result

When all rows are DONE, the dispatcher writes `SUMMARY.md` and stops the loop.
