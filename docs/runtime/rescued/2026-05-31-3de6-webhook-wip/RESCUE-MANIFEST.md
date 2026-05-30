# RESCUE MANIFEST — 3de6 worktree uncommitted webhook WIP (2026-05-31)

## Source
- Worktree: `C:/Users/joshd/.codex/worktrees/3de6/canompx3` (detached HEAD `2e6f8024`)
- `2e6f8024` (NYSE_PREOPEN prereg) IS an ancestor of main → committed tip already merged.
- The **uncommitted** WIP below was the only at-risk content.

## Rescued content
- `webhook-wip.patch` — uncommitted tracked-file diff (82 lines) over:
  - `trading_app/live/webhook_server.py`
  - `tests/test_trading_app/test_webhook_server.py`

## Assessment — REAL CAPITAL-PATH BUGFIX (NOT disposable)
- Relocates the `MAX_ORDER_QTY` cap from **before** `_resolve_execution_request()` to **after** it.
- Why it matters: profile execution-mapping (e.g. NQ-mini routing MNQ→NQ) divides the qty. The
  old code capped `req.qty` (the raw strategy qty) — so the *broker-side* qty that actually
  reaches `_place_order` was never re-checked against `MAX_ORDER_QTY`. An oversized request that
  divides down to a valid broker qty was being rejected wrongly, and (more dangerously) the cap
  semantics did not reflect the value sent to the router.
- Verified UNMERGED on main 2026-05-31: main `webhook_server.py:335-336` still caps `req.qty`
  before the resolve at `:356`. Test `test_profile_execution_map_applies_qty_cap_after_division`
  is absent from main.
- Companion regression test asserts the **mapped** qty (5), not the raw request qty (20), is what
  `_place_order` receives.

## Disposition
- Because this is a Tier B (live webhook / capital) path, it is **NOT merged autonomously**.
- Rescued onto branch `session/joshd-webhook-exec-qty-cap-rescue` and opened as a review-gated PR
  (see Phase A of the 2026-05-31 worktree-cleanup plan). This manifest is the redundant net.

## Provenance
- Rescued by main session 2026-05-31 (Phase A2 of worktree-cleanup plan).
- Related: `memory/project_nq_mini_stage2_wiring_shipped_2026_05_30.md` (NQ-mini execution mapping).
