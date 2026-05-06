# feat/live-app-ux — operator smoke check

Browser-side verification that JS event wiring (Trade Book toggle, paused-lanes
badge, 60s polling) renders against today's live state. `TestClient` proves
JSON shapes — only a real browser proves the DOM wiring.

**Status:** TEMPLATE — operator must run procedure and fill in observed values.
Verification artefact #3 of 4 per the Stage A plan.

## Cross-check baseline (pre-recorded 2026-05-06 via canonical accessors)

Run on the working tree at HEAD = post-Task-1 commit on `feat/live-app-ux`:

| Source | Value |
|---|---|
| `lane_ctl.get_paused_strategy_ids("topstep_50k_mnq_auto")` | 1 entry |
| Paused strategy_id | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` |
| Reason | `SR alarm: stat=33.27 >= thr=31.96; baseline=paper_trades_first_50; stream=paper_trades` |
| Expires | `2026-05-23` |
| `live_journal.db.live_trades` row count | 2 |
| `gold.db.paper_trades` row count | 580 |

Observed UI values must match these (or be logged as a mismatch with explanation).

## Procedure

1. **Launch.** From repo root, `START_BOT.bat`. Confirm uvicorn binds 127.0.0.1:8080.
2. **Open browser** at `http://localhost:8080/`. Open DevTools → Network tab.
3. **Trade Book toggle.** Click the "Trade Book" topbar button.
   - Section expands.
   - Both tables render with column headers.
   - Network tab shows ONE call to `/api/trade-book` (200 OK).
4. **Paused-lanes badge.** Look at the Positions header.
   - Badge shows `1` (not `0`, not absent).
   - Hover/click reveals tooltip with `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`,
     reason text, and `2026-05-23` expiry.
5. **60s polling.** Wait 60 seconds with the page open.
   - Network tab shows a SECOND call to `/api/lane-status` (200 OK).
6. **Cross-check.**
   - Trade Book live count vs `live_trades=2` baseline above.
   - Trade Book paper count vs `paper_trades=580` baseline above.
   - Badge count vs `paused_count=1` baseline above.

## Recorded outcome

> Operator: fill in below after running. If any step fails, file a bug — do
> not merge. Report observed values verbatim, not "matches baseline" — the
> point of the cross-check is to catch silent drift.

| Step | Pass / Fail | Observed |
|---|---|---|
| 1. uvicorn binds 8080 | _ | _ |
| 2. localhost:8080 loads | _ | _ |
| 3. Trade Book toggle | _ | live count=__, paper count=__, network call=__ |
| 4. Paused-lanes badge | _ | badge count=__, tooltip strategy_id=__, reason=__, expires=__ |
| 5. 60s polling | _ | second `/api/lane-status` call observed at t=__s |
| 6. Cross-check live | _ | UI live count vs canonical 2 |
| 6. Cross-check paper | _ | UI paper count vs canonical 580 |
| 6. Cross-check paused | _ | UI badge vs canonical 1 |

**Operator name / date:** _

**Verdict:** _ (PASS / FAIL — any FAIL blocks merge)

## Why this can't be a TestClient test

`TestClient` calls the FastAPI route function directly. It cannot prove:

- The HTML button has an event handler attached.
- The handler actually fetches `/api/trade-book`.
- The response is rendered into DOM nodes the operator can see.
- The 60s `setInterval` polling fires.
- The badge tooltip reads the right field from the JSON.

Those failure modes only surface when a real browser parses the HTML, runs
the JS, and renders the DOM. 5 minutes of operator time covers what 100
unit tests cannot.
