task: Fix live bar volume corruption — convert cumulative GatewayQuote.volume to per-tick delta at the ProjectX boundary (Defect A)
mode: IMPLEMENTATION
status: code-complete-verified

## Verification (2026-06-10)
- 49/49 tests pass (test_projectx_feed + test_bar_aggregator). Aggregator delta contract preserved.
- Drift: 1 violation / 183 checks, exit 0 — the violation is pre-existing Check 79 (MNQ daily_features
  data gap), independent of this diff (touches only data_feed.py + tests; Check 79 inspects neither).
- Ruff: clean.
- Adversarial-audit gate (evidence-auditor, independent context): CONDITIONAL → its highest-priority
  finding FIXED. parse_quote `0→1` coercion (data_feed.py:93) was wrong under cumulative semantics
  (silent off-by-one when session opens at cumulative 0). Now distinguishes explicit 0 (pass through)
  from missing field (default 1). Added test_zero_cumulative_baseline_no_off_by_one + rewrote
  test_zero_volume_becomes_one → test_explicit_zero_volume_passes_through. parse_quote has zero
  external callers (grep-verified) — contract change fully contained.

## OPEN (deferred, Tier B — NOT part of this code change):
1. Behavioral proof: after next live/demo session, query bar volume band for source_symbol='MNQ'
   — must be ≤ ~tens of thousands (the real-feed acceptance criterion; tests prove math only).
2. Backfill remediation: re-pull MNQ 2026-04-24→latest from Databento + rebuild daily_features
   (1,037 poisoned rows, rel_vol_NYSE_OPEN=92,198). Separate staged DB-write step.

## Scope Lock
- trading_app/live/projectx/data_feed.py
- tests/test_trading_app/test_projectx_feed.py

## Blast Radius
- data_feed.py — adds `_last_cum_volume` state + `_cum_to_delta()` helper; converts cumulative quote volume to delta in 4 handler sites (`_on_quote`, `_on_quote_sync`, `_on_trade`, `_on_trade_sync`) between extract and `on_tick`. `parse_quote` stays a PURE extractor (locked by TestParseQuote). Reads: nothing new; Writes: nothing (in-memory state only). Only callsite of BarAggregator.on_tick is this feed.
- test_projectx_feed.py — new TestCumulativeToDelta tests (delta math, session-reset clamp, realistic per-minute band, thread path). Existing TestParseQuote/TestFlush/TestSyncCallbackQueueBridge UNCHANGED (regression proof).
- bar_aggregator.py — UNCHANGED. Broker-agnostic delta contract + 7 volume tests preserved.
- NOT touching: deployed ORB/friction gate (reads backfill features), backtest/discovery pipeline (live-feed-only change).
- Capital path (trading_app/live/) → Tier B → adversarial-audit gate MANDATORY before done.
