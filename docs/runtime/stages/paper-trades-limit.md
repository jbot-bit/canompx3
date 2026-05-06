---
task: Add LIMIT 5000 to /api/trade-book paper_trades query + truncated-flag flow + UI hint (PR #248 LOW polish)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - tests/test_trading_app/test_bot_dashboard_routes.py
---

## Blast Radius

- `trading_app/live/bot_dashboard.py` — `/api/trade-book` route (`api_trade_book` ~`:1175-1277`). Adds `LIMIT 5000` to the paper_trades SELECT + emits a new boolean `paper_truncated` and integer `paper_total_count` in the response payload. Live_trades query is left as-is (live_journal is bounded by single-account fills/day; not the same growth concern).
- `trading_app/live/bot_dashboard.html` — `fetchTradeBook()` JS handler reads the new fields and appends a "showing first 5000 of N" suffix to `paperNoteEl` when truncated.
- `tests/test_trading_app/test_bot_dashboard_routes.py` — extends with two tests: (a) <5000 rows → `paper_truncated=False`, `paper_total_count=N`; (b) >5000 rows → only 5000 returned, `paper_truncated=True`, `paper_total_count` reflects full count.
- Reads: `gold.db.paper_trades` (read-only). Writes: none.
- Live order-route impact: NONE. Read-only dashboard route.

## Why this is IMPLEMENTATION (not TRIVIAL)

`trading_app/live/` is on the NEVER_TRIVIAL list (production live-trading path even when read-only — the dashboard is operator-facing). Diff is small (~15 LOC across 3 files) but stage protocol applies.

## Implementation

1. `bot_dashboard.py` — modify the SELECT to:
   ```sql
   SELECT trading_day, instrument, ...
   FROM paper_trades
   ORDER BY trading_day DESC, entry_time DESC
   LIMIT 5001
   ```
   `LIMIT 5001` instead of `5000` so we can detect truncation in one query without a second `COUNT(*)` call. If `len(rows) > 5000`, set `paper_truncated=True` and trim to 5000; otherwise `paper_total_count = len(rows)` and `paper_truncated=False`.

   Wait — that's wrong. If we LIMIT 5001, `paper_total_count` only knows ≥5001, not the true count. Pick one of two shapes:

   **Shape A (LIMIT 5001 + count probe):** if `len > 5000`, run a second `SELECT COUNT(*) FROM paper_trades` to get the true total. Two queries when truncated, one when not. Simple but hits DB twice on the truncated path.

   **Shape B (always count, then fetch):** issue `SELECT COUNT(*)` first, then issue the LIMIT'd fetch. Always two queries.

   **Decision: Shape A.** Saves a query in the common (untruncated) path; adds a query only when the user is in the >5k-trades regime where they're already paying for it. Net zero perf cost today (paper_trades=580 per smoke baseline).

2. `bot_dashboard.html` — in `fetchTradeBook()` after parsing `data`:
   ```js
   if (paperNoteEl) {
     const baseNote = data.paper_note || "";
     if (data.paper_truncated) {
       const total = data.paper_total_count;
       paperNoteEl.textContent = baseNote + (baseNote ? " · " : "") + `showing first 5000 of ${total}`;
     } else {
       paperNoteEl.textContent = baseNote;
     }
   }
   ```
   Existing `escapeHtml`/`fmt` not needed — `paperNoteEl.textContent` (not innerHTML) is the canonical safe write.

3. Companion tests — add `test_trade_book_paper_truncation_flag_under_limit` and `test_trade_book_paper_truncation_flag_over_limit`. The over-limit test seeds 5050 rows; assert `len(paper_trades) == 5000`, `paper_truncated == True`, `paper_total_count == 5050`. The under-limit test confirms `paper_truncated == False`, `paper_total_count == N`.

## Acceptance criteria — ALL required

1. `pytest tests/test_trading_app/test_bot_dashboard_routes.py -v` → all tests pass (existing 6 + 2 new = 8/8).
2. `python pipeline/check_drift.py` → 122 PASS, 0 FAIL.
3. `grep -n "LIMIT 5001" trading_app/live/bot_dashboard.py` → exactly one match.
4. `grep -n "paper_truncated" trading_app/live/bot_dashboard.py trading_app/live/bot_dashboard.html` → ≥3 matches (set in route, read in JS).
5. Self-review: PREMISE/TRACE/EVIDENCE/CONCLUSION on the two queries' total cost & correctness.

## Out of scope

- live_trades query (different growth profile).
- Pagination UI (single-page truncation note is the agreed scope).
- Any change to existing column ordering / fields.
